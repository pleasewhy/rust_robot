use burn::{
    config::Config,
    module::Module,
    nn::LinearConfig,
    optim::{decay::WeightDecayConfig, AdamConfig, Optimizer},
    prelude::Backend,
    record::{Record, Recorder},
    tensor::{cast::ToElement, Bool, Int, Tensor, TensorData},
    train::checkpoint::{Checkpointer, FileCheckpointer},
};
use ndarray::{Array2, ArrayView1, ArrayView2, MultiSliceArg};
use ndarray_rand::RandomExt;
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
    rl_algorithm::{
        base::rl_utils::{bool_ndarray2tensor1, ndarray2tensor1, tensor2ndarray2},
        ppo::ppo_agent::PPO,
    },
    rl_env::{
        env::{EnvConfig, MujocoEnv},
        env_sampler::{self, create_n_env, BatchTrajInfo, FlattenBatchTrajInfo},
    },
};

use super::{
    config::TrainConfig,
    model::{ActorModel, BaselineModel, ModelBasedNet, RlTrainAlgorithm},
    rl_utils::{self, ndarray2tensor2},
};

use super::memory::Memory;

pub struct ModelBasedRunner<E: MujocoEnv + Send + 'static, B: AutodiffBackend> {
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
        flatten_batch_traj.next_obs_vec,
        flatten_batch_traj.action_vec,
        flatten_batch_traj.reward_vec,
        flatten_batch_traj.done_vec,
        device,
    );
}

impl<E: MujocoEnv + Send + 'static, B: AutodiffBackend> ModelBasedRunner<E, B> {
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

    pub fn random_policy(&mut self, obs: Array2<f64>) -> Array2<f64> {
        let arr = ndarray::Array2::random(
            [obs.shape()[0], self.eval_env.get_action_dim()],
            ndarray_rand::rand_distr::Uniform::new(-1.0, 1.0),
        );
        return arr;
    }

    pub fn sample_to_memory<AM: ActorModel<B>>(&mut self, actor: &AM, iter: usize) -> Memory<B> {
        let policy = |obs: Array2<f64>| {
            let action = actor.forward(ndarray2tensor2(obs, &self.device)).sample();
            let mut action = tensor2ndarray2(&action);
            let action = action.mapv(|x| x as f64);
            return action;
        };
        let trajs = self.env_sampler.sample_n_trajectories(&policy);
        if iter % self.config.video_log_freq == 0 {
            self.eval_env.run_policy(
                &format!("{}_iter{}.mp4", self.env_name, iter),
                self.config.env_config.traj_length,
                &policy,
            );
        }
        return batch_traj_to_memory(trajs, &self.device);
    }

    pub fn update_dynamic<MBN: ModelBasedNet<B> + AutodiffModule<B> + Display>(
        &self,
        mut dynamic_model: MBN,
        memory: &Memory<B>,
        dynamic_model_optimizer: &mut (impl Optimizer<MBN, B> + Sized),
    ) -> (MBN, f32) {
        let mut loss_f32 = 0.0;
        let update_num = 10;
        for _ in 0..update_num {
            let loss = dynamic_model.loss(
                memory.obs().clone(),
                memory.action().clone(),
                memory.next_obs().clone(),
                memory.reward().clone(),
            );
            dynamic_model = rl_utils::update_parameters(
                loss.clone(),
                dynamic_model,
                dynamic_model_optimizer,
                self.config.ppo_train_config.learning_rate.into(),
            );
            loss_f32 += loss.clone().into_scalar().to_f32();
        }
        return (dynamic_model, loss_f32 / update_num as f32);
    }

    fn collect_memory_from_dynamic_model<
        AM: ActorModel<B> + AutodiffModule<B> + Display,
        MBN: ModelBasedNet<B> + AutodiffModule<B> + Display,
    >(
        &mut self,
        batch_size: usize,
        traj_length: usize,
        actor: AM,
        dynamic_model: MBN,
    ) -> Memory<B> {
        let mut obs = Vec::<f64>::new();
        for i in 0..batch_size {
            self.eval_env.reset();
            let obs_tmp = self.eval_env.get_obs();
            obs.extend_from_slice(obs_tmp.as_slice());
        }
        let mut obs = ndarray2tensor2(
            Array2::from_shape_vec([batch_size, self.eval_env.get_obs_dim()], obs).unwrap(),
            &self.device,
        );
        let mut obs_vec: Vec<Tensor<B, 2>> = Vec::with_capacity(traj_length);
        let mut next_obs_vec: Vec<Tensor<B, 2>> = Vec::with_capacity(traj_length);
        let mut action_vec: Vec<Tensor<B, 2>> = Vec::with_capacity(traj_length);
        let mut reward_vec: Vec<Tensor<B, 1>> = Vec::with_capacity(traj_length);
        let mut done_vec: Vec<Tensor<B, 1, Bool>> = Vec::with_capacity(traj_length);
        for i in 0..traj_length {
            let action: Tensor<B, 2> = actor.forward(obs.clone()).sample();
            let next_obs = dynamic_model.forward(obs.clone(), action.clone());
            obs_vec.push(obs.clone());
            next_obs_vec.push(next_obs.clone());
            action_vec.push(action.clone());

            let obs_tmp = tensor2ndarray2(&obs).mapv(|x| x as f64);
            let next_obs_tmp = tensor2ndarray2(&next_obs).mapv(|x| x as f64);
            let action_tmp = tensor2ndarray2(&action).mapv(|x| x as f64);
            let (reward, dones) =
                self.eval_env
                    .get_reward(obs_tmp.view(), next_obs_tmp.view(), action_tmp.view());
            let reward: Tensor<B, 1> = ndarray2tensor1(reward, &self.device);
            let dones: Tensor<B, 1, Bool> = bool_ndarray2tensor1(dones, &self.device);
            reward_vec.push(reward.clone());
            done_vec.push(dones);
            obs = next_obs;
        }
        let obs = Tensor::cat(obs_vec, 0);
        let next_obs = Tensor::cat(next_obs_vec, 0);
        let action = Tensor::cat(action_vec, 0);
        let reward = Tensor::cat(reward_vec, 0);
        let done = Tensor::cat(done_vec, 0);
        return Memory::new_by_tensor(obs, next_obs, action, reward, done);
    }
    pub fn train_update<
        AM: ActorModel<B> + AutodiffModule<B> + Display,
        BM: BaselineModel<B> + AutodiffModule<B> + Display,
        MBN: ModelBasedNet<B> + AutodiffModule<B> + Display,
        RlAlgo: RlTrainAlgorithm<B, AM, BM>,
    >(
        &mut self,
        mut actor_net: AM,
        mut baseline_net: BM,
        mut dynamic_model: MBN,
        mut train_algo: RlAlgo,
    ) {
        let mut actor_optimizer = AdamWConfig::new()
            .with_grad_clipping(self.config.grad_clip.clone())
            .init::<B, AM>();
        let mut baseline_optimizer = AdamWConfig::new()
            .with_grad_clipping(self.config.grad_clip.clone())
            .init::<B, BM>();
        let mut dynamic_model_optimizer = AdamWConfig::new()
            .with_grad_clipping(self.config.grad_clip.clone())
            .init::<B, MBN>();
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
            let mut memory = self.sample_to_memory(&actor_net, iter);

            let mean_reward = (memory.reward().clone() / self.config.env_config.n_env as f32)
                .sum()
                .into_scalar()
                .to_f32();
            log_info.insert("mean_reward".to_string(), mean_reward);
            log_info.insert("step_num".to_string(), memory.len() as f32);
            log_info.insert(
                "collect_time".to_string(),
                start.elapsed().unwrap().as_millis() as f32,
            );
            let dynamic_loss: f32;
            (dynamic_model, dynamic_loss) =
                self.update_dynamic(dynamic_model, &memory, &mut dynamic_model_optimizer);

            if iter >= 10000 {
                start = SystemTime::now();

                let dynamic_model_memory = self.collect_memory_from_dynamic_model(
                    self.config.env_config.n_env * 50,
                    10,
                    actor_net.clone(),
                    dynamic_model.clone(),
                );
                memory.merge(&dynamic_model_memory);
                log_info.insert(
                    "dynamic_model_collect_time".to_string(),
                    start.elapsed().unwrap().as_millis() as f32,
                );
            }
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
            log_info.insert("dynamic_net_loss".to_string(), dynamic_loss);
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
