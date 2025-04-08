use burn::{
    module::Module,
    optim::{decay::WeightDecayConfig, AdamConfig, Optimizer},
    prelude::Backend,
    record::{Record, Recorder},
    tensor::{cast::ToElement, Tensor, TensorData},
    train::checkpoint::{Checkpointer, FileCheckpointer},
};
use std::{
    collections::HashMap,
    fmt::Display,
    marker::PhantomData,
    sync::{Arc, Mutex},
    time::SystemTime,
};

use burn::grad_clipping::GradientClippingConfig;
use burn::{
    module::AutodiffModule,
    optim::AdamWConfig,
    record::{DefaultFileRecorder, FullPrecisionSettings},
    tensor::backend::AutodiffBackend,
};
use chrono::Local;
use tensorboard_rs::summary_writer::SummaryWriter;

use crate::{
    rl_algorithm::ppo::ppo_agent::PPO,
    rl_env::{
        env::MujocoEnv,
        env_sampler::{self, BatchTrajInfo},
        nd_vec::{tensor2vec2, NdVec2},
    },
};

use super::{
    memory::Memory,
    ppo::{self, config::PPOTrainingConfig},
};
use crate::rl_env::nd_vec::vec2tensor2;

pub struct TrainConfig {
    pub ppo_train_config: PPOTrainingConfig,
    pub n_env: usize,
    pub traj_length: usize,
    pub video_log_freq: usize,
    pub train_iter: usize,
    pub ckpt_save_path: String,
    pub resume_from_ckpt_path: Option<String>,
    pub save_model_freq: usize,
    pub grad_clip: Option<GradientClippingConfig>,
}

// struct LogInfo {
//     iter: usize,
//     actor_loss: f32,
//     baseline_loss: f32,
//     learn_time: f32,
//     collect_time: f32,
//     reward: f32,
//     q_vals: f32,
// }

struct CheckPoint<B: Backend, R: Record<B>> {
    actor_net_ckpter: FileCheckpointer<R>,
    baseline_net_ckpter: FileCheckpointer<R>,
    actor_opti_ckpter: FileCheckpointer<R>,
    baseline_opti_ckpter: FileCheckpointer<R>,
    backend: PhantomData<B>,
}

pub struct OnPolicyRunner<E: MujocoEnv + Send + 'static, B: AutodiffBackend> {
    device: B::Device,
    backend: PhantomData<B>,
    writer: SummaryWriter,
    env_name: String,
    eval_env: E,
    env_sampler: env_sampler::BatchEnvSample<E>,
    config: TrainConfig,
    exp_name: String,
}

fn create_n_env<ENV: MujocoEnv>(n_env: usize) -> Vec<Arc<Mutex<ENV>>> {
    let mut envs = vec![];
    for _ in 0..n_env {
        envs.push(Arc::new(Mutex::new(ENV::new(false))));
    }
    return envs;
}

fn batch_traj_to_memory<B: Backend>(
    mut batch_traj: BatchTrajInfo,
    device: &B::Device,
) -> Memory<B> {
    let batch_size = batch_traj.reward.shape()[0];
    let traj_length = batch_traj.reward.shape()[1];
    let num_element = batch_size * traj_length;

    let mut obs_vec = Vec::<f32>::with_capacity(num_element);
    let mut action_vec = Vec::<f32>::with_capacity(num_element);
    let mut reward_vec = Vec::<f32>::with_capacity(num_element);
    let mut done_vec = Vec::<bool>::with_capacity(num_element);

    for b in 0..batch_traj.reward.shape()[0] {
        for t in 0..batch_traj.reward.shape()[1] {
            let shape = (b, t);
            let done = *batch_traj.terminal.get(shape).unwrap() > 0;
            obs_vec.extend_from_slice(batch_traj.observation.slice_mut(shape).unwrap());
            action_vec.extend_from_slice(batch_traj.action.slice_mut(shape).unwrap());
            reward_vec.push(*batch_traj.reward.get(shape).unwrap());
            done_vec.push(done);

            if done {
                break;
            }
        }
    }
    return Memory::new(obs_vec, action_vec, reward_vec, done_vec, device);
}

impl<E: MujocoEnv + Send + 'static, B: AutodiffBackend> OnPolicyRunner<E, B> {
    pub fn new(device: B::Device, config: TrainConfig) -> Self {
        let env_name = std::any::type_name::<E>().rsplit("::").next().unwrap();
        let exp_name = format!("ppo_{}_{}", env_name, Local::now().format("%d-%m-%Y %H-%M"));
        let writer = SummaryWriter::new(format!("./logdir/{}", &exp_name));
        let env_sampler: env_sampler::BatchEnvSample<E> = env_sampler::BatchEnvSample::new(
            config.traj_length,
            4,
            create_n_env::<E>(config.n_env),
        );
        Self {
            device: device,
            backend: PhantomData,
            writer,
            env_name: env_name.to_string(),
            eval_env: E::new(true),
            env_sampler: env_sampler,
            config: config,
            exp_name,
        }
    }

    pub fn sample_to_memory<AM: ppo::model::ActorModel<B>>(
        &mut self,
        actor: &AM,
        iter: usize,
    ) -> Memory<B> {
        let policy = |obs: NdVec2<f64>| {
            let action = actor.forward(vec2tensor2(obs, &self.device)).sample();
            return tensor2vec2(&action).to_f64();
        };
        let trajs = self.env_sampler.sample_n_trajectories(&policy);
        if iter % self.config.video_log_freq == 0 {
            self.eval_env.run_policy(
                &format!("{}_iter{}.mp4", self.env_name, iter),
                self.config.traj_length,
                &policy,
            );
        }
        return batch_traj_to_memory(trajs, &self.device);
    }

    pub fn train_update<
        AM: ppo::model::ActorModel<B> + AutodiffModule<B> + Display,
        BM: ppo::model::BaselineModel<B> + AutodiffModule<B> + Display,
    >(
        &mut self,
        mut actor_net: AM,
        mut baseline_net: BM,
    ) {
        let mut actor_optimizer = AdamWConfig::new()
            .with_grad_clipping(self.config.grad_clip.clone())
            .init::<B, AM>();
        let mut baseline_optimizer = AdamWConfig::new()
            .with_grad_clipping(self.config.grad_clip.clone())
            .init::<B, BM>();

        // println!("actor_optimizer={:?}", actor_optimizer);
        // println!("baseline_optimizer={:?}", baseline_optimizer);
        (actor_net, baseline_net, actor_optimizer, baseline_optimizer) =
            self.resume_from_ckpt(actor_net, baseline_net, actor_optimizer, baseline_optimizer);
        let mut update_info: super::utils::UpdateInfo;
        let mut log_info = HashMap::<String, f32>::new();
        for iter in 0..self.config.train_iter {
            log_info.clear();
            let mut start = SystemTime::now();
            let memory = self.sample_to_memory(&actor_net, iter);

            let mean_reward =
                memory.reward().clone().sum().into_scalar().to_f32() / self.config.n_env as f32;
            log_info.insert("mean_reward".to_string(), mean_reward);
            log_info.insert("step_num".to_string(), memory.len() as f32);
            log_info.insert(
                "collect_time".to_string(),
                start.elapsed().unwrap().as_millis() as f32,
            );

            start = SystemTime::now();
            (actor_net, baseline_net, update_info) = PPO::train(
                actor_net,
                baseline_net,
                &memory,
                &mut actor_optimizer,
                &mut baseline_optimizer,
                &self.config.ppo_train_config,
                &self.device,
            )
            .unwrap();

            if iter > 0 && iter % self.config.save_model_freq == 0 {
                self.save_ckpt(
                    iter,
                    mean_reward,
                    actor_net.clone().into_record(),
                    baseline_net.clone().into_record(),
                    actor_optimizer.to_record(),
                    baseline_optimizer.to_record(),
                );
            }

            log_info.insert(
                "ppo_update_time".to_string(),
                start.elapsed().unwrap().as_millis() as f32,
            );
            log_info.insert("actor_loss".to_string(), update_info.actor_loss);
            log_info.insert("critic_loss".to_string(), update_info.critic_loss);
            log_info.insert("mean_q_val".to_string(), update_info.mean_q_val);
            log_info.insert(
                "actor_std".to_string(),
                actor_net.std_mean().into_scalar().to_f32(),
            );
            self.log(iter, &log_info);
        }
    }

    fn write_scalar(&mut self, main_tag: &str, sub_tag: &str, scalar: f32, step: usize) {
        let mut map = HashMap::<String, f32>::new();
        map.insert(sub_tag.to_string(), scalar);
        self.writer.add_scalars(main_tag, &map, step);
    }

    fn log(&mut self, step: usize, log_info: &HashMap<String, f32>) {
        println!("************iter={}************", step);
        for (tag, scalar) in log_info {
            self.write_scalar(format!("humanoid/{}", &tag).as_str(), &tag, *scalar, step);
            println!("{}={}", tag, scalar);
        }
        println!("************iter={}************", step);
        println!();
        println!();
        self.writer.flush();
    }

    fn save_ckpt<R1: Record<B>, R2: Record<B>, R3: Record<B>, R4: Record<B>>(
        &self,
        iter: usize,
        mean_reward: f32,
        actor_net_record: R1,
        baseline_net_record: R2,
        actor_opti: R3,
        baseline_opti: R4,
    ) {
        let path = format!(
            "{}/{}/iter{}_mean_reward{}",
            self.config.ckpt_save_path, self.exp_name, iter, mean_reward
        );
        let file_recorder = DefaultFileRecorder::<FullPrecisionSettings>::new();

        file_recorder
            .record(actor_net_record, format!("{}/actor_net", path).into())
            .unwrap();
        file_recorder
            .record(baseline_net_record, format!("{}/baseline_net", path).into())
            .unwrap();
        file_recorder
            .record(actor_opti, format!("{}/actor_opti", path).into())
            .unwrap();
        file_recorder
            .record(baseline_opti, format!("{}/baseline_opti", path).into())
            .unwrap();
    }

    fn resume_from_ckpt<
        AM: ppo::model::ActorModel<B> + AutodiffModule<B> + Display,
        BM: ppo::model::BaselineModel<B> + AutodiffModule<B> + Display,
        AO: Optimizer<AM, B>,
        BO: Optimizer<BM, B>,
    >(
        &self,
        mut actor_net: AM,
        mut baseline_net: BM,
        mut actor_optimizer: AO,
        mut baseline_optimizer: BO,
    ) -> (AM, BM, AO, BO) {
        if self.config.resume_from_ckpt_path.is_none() {
            return (actor_net, baseline_net, actor_optimizer, baseline_optimizer);
        }
        let path = self.config.resume_from_ckpt_path.as_ref().unwrap().clone();
        let file_recorder = DefaultFileRecorder::<FullPrecisionSettings>::new();
        actor_net = actor_net
            .load_file(format!("{}/actor_net", path), &file_recorder, &self.device)
            .unwrap();
        baseline_net = baseline_net
            .load_file(
                format!("{}/baseline_net", path),
                &file_recorder,
                &self.device,
            )
            .unwrap();
        let actor_opti_record = file_recorder
            .load(format!("{}/actor_opti", path).into(), &self.device)
            .unwrap();
        let baseline_opti_record = file_recorder
            .load(format!("{}/baseline_opti", path).into(), &self.device)
            .unwrap();
        actor_optimizer = actor_optimizer.load_record(actor_opti_record);
        baseline_optimizer = baseline_optimizer.load_record(baseline_opti_record);
        return (actor_net, baseline_net, actor_optimizer, baseline_optimizer);
    }
}
