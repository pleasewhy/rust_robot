use burn::{
    config::Config,
    module::Module,
    nn::LinearConfig,
    optim::{decay::WeightDecayConfig, AdamConfig, Optimizer},
    prelude::Backend,
    record::{Record, Recorder},
    tensor::{cast::ToElement, Tensor, TensorData},
    train::checkpoint::{Checkpointer, FileCheckpointer},
};
use ndarray::{Array2, ArrayView1, ArrayView2, MultiSliceArg};
use std::{
    collections::HashMap,
    fmt::Display,
    marker::PhantomData,
    path::Path,
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
use chrono::{Local, Utc};
use chrono_tz::Asia::Shanghai;
use tensorboard_rs::summary_writer::SummaryWriter;

use crate::{
    rl_algorithm::ppo::ppo_agent::PPO,
    rl_env::{
        env::{EnvConfig, MujocoEnv},
        env_sampler::{self, create_n_env, BatchTrajInfo, FlattenBatchTrajInfo},
    },
};

use crate::rl_algorithm::base::rl_utils::{ndarray2tensor2, tensor2ndarray2};

use super::{
    config::TrainConfig,
    model::{ActorModel, BaselineModel, RlTrainAlgorithm},
};

use super::memory::Memory;

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
    exp_base_path: String,
}

fn batch_traj_to_memory<B: Backend>(
    flatten_batch_traj: FlattenBatchTrajInfo,
    device: &B::Device,
) -> Memory<B> {
    return Memory::new(
        flatten_batch_traj.obs_vec,
        Vec::new(),
        flatten_batch_traj.action_vec,
        flatten_batch_traj.reward_vec,
        flatten_batch_traj.done_vec,
        device,
    );
}

impl<E: MujocoEnv + Send + 'static, B: AutodiffBackend> OnPolicyRunner<E, B> {
    pub fn new(device: B::Device, config: TrainConfig, algo_name: &str) -> Self {
        let env_name = std::any::type_name::<E>().rsplit("::").next().unwrap();
        let exp_name = format!(
            "{}_{}_{}",
            algo_name,
            env_name,
            Utc::now().with_timezone(&Shanghai).format("%m-%d %H:%M")
        );
        let writer = SummaryWriter::new(format!("./logdir/{}", &exp_name));
        let env_sampler: env_sampler::BatchEnvSample<E> = env_sampler::BatchEnvSample::new(
            config.env_config.traj_length,
            6,
            create_n_env::<E>(config.env_config.clone()),
        );
        let exp_base_path = format!("{}/{}", config.ckpt_save_path, exp_name);
        std::fs::create_dir_all(&exp_base_path);
        Self {
            device: device,
            backend: PhantomData,
            writer,
            env_name: env_name.to_string(),
            eval_env: E::new(true, config.env_config.clone()),
            env_sampler,
            config,
            exp_name,
            exp_base_path,
        }
    }

    pub fn sample_to_memory<AM: ActorModel<B>>(&mut self, actor: &AM, iter: usize) -> Memory<B> {
        let policy = |obs: Array2<f64>| {
            let action = actor.forward(ndarray2tensor2(obs, &self.device)).sample();
            return tensor2ndarray2(&action).map(|x| *x as f64);
        };
        let trajs = self.env_sampler.sample_n_trajectories(&policy);
        if iter % self.config.video_log_freq == 0 {
            self.eval_env.run_policy(
                &format!("{}_iter{}", self.env_name, iter),
                self.config.env_config.traj_length,
                &policy,
            );
        }
        return batch_traj_to_memory(trajs, &self.device);
    }

    pub fn train_update<
        AM: ActorModel<B> + AutodiffModule<B> + Display,
        BM: BaselineModel<B> + AutodiffModule<B> + Display,
        RlAlgo: RlTrainAlgorithm<B, AM, BM>,
    >(
        &mut self,
        mut actor_net: AM,
        mut baseline_net: BM,
        mut train_algo: RlAlgo,
    ) {
        let mut actor_optimizer = AdamWConfig::new()
            .with_grad_clipping(self.config.grad_clip.clone())
            .init::<B, AM>();
        let mut baseline_optimizer = AdamWConfig::new()
            .with_grad_clipping(self.config.grad_clip.clone())
            .init::<B, BM>();
        self.config
            .save(format!("{}/config.json", self.exp_base_path))
            .expect("save config failed.");

        (actor_net, baseline_net, actor_optimizer, baseline_optimizer) =
            self.resume_from_ckpt(actor_net, baseline_net, actor_optimizer, baseline_optimizer);
        let mut update_info: super::rl_utils::UpdateInfo;
        let mut log_info = HashMap::<String, f32>::new();
        for iter in 0..self.config.train_iter {
            log_info.clear();
            let mut start = SystemTime::now();
            let memory = self.sample_to_memory(&actor_net, iter);

            let mean_reward = memory.reward().clone().sum().into_scalar().to_f32()
                / self.config.env_config.n_env as f32;
            log_info.insert("mean_reward".to_string(), mean_reward);
            log_info.insert("step_num".to_string(), memory.len() as f32);
            log_info.insert(
                "collect_time".to_string(),
                start.elapsed().unwrap().as_millis() as f32,
            );

            start = SystemTime::now();
            (actor_net, baseline_net, update_info) = train_algo
                .train(
                    actor_net,
                    baseline_net,
                    &memory,
                    &mut actor_optimizer,
                    &mut baseline_optimizer,
                    &self.config,
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
            "{}/iter{}_mean_reward{}",
            self.exp_base_path, iter, mean_reward
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
        AM: ActorModel<B> + AutodiffModule<B> + Display,
        BM: BaselineModel<B> + AutodiffModule<B> + Display,
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
