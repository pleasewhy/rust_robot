use burn::{
    config::Config,
    optim::Optimizer,
    prelude::Backend,
    record::{Record, Recorder},
    tensor::{cast::ToElement, Float, Tensor},
    train::checkpoint::FileCheckpointer,
};
use burn::{
    module::AutodiffModule,
    optim::AdamWConfig,
    record::{DefaultFileRecorder, FullPrecisionSettings},
    tensor::backend::AutodiffBackend,
};
use chrono::Utc;
use chrono_tz::Asia::Shanghai;
use ndarray::{Array2, Array3};
use std::{collections::HashMap, fmt::Display, marker::PhantomData, time::SystemTime};
use tensorboard_rs::summary_writer::SummaryWriter;

use crate::rl_env::{
    env::MujocoEnv,
    env_sampler::{self, create_n_env, BatchTrajInfo},
};

use crate::rl_algorithm::base::rl_utils::{ndarray2tensor2, tensor2ndarray2};

use super::{
    config::TrainConfig,
    model::{ActorModel, BaselineModel, RlTrainAlgorithm},
    EpochLogger,
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
    env_name: String,
    algo_name: String,
    eval_env: E,
    action_dim: usize,
    env_sampler: env_sampler::BatchEnvSample<E>,
    config: TrainConfig,
    exp_name: String,
    exp_base_path: String,
}

fn batch_traj_to_memory<B: Backend>(batch_traj: BatchTrajInfo, device: &B::Device) -> Memory<B> {
    return Memory::new(
        batch_traj.obs.mapv(|x| crate::f32_to_ftype(x)),
        Array3::zeros((1, 1, 1)),
        batch_traj.action.mapv(|x| crate::f32_to_ftype(x)),
        batch_traj.reward.mapv(|x| crate::f32_to_ftype(x)),
        batch_traj.terminal.map(|x| *x > 0u8),
        batch_traj.traj_length,
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
        EpochLogger::init_writer(format!("./logdir/{}", &exp_name));
        let start = SystemTime::now();
        let env_sampler: env_sampler::BatchEnvSample<E> = env_sampler::BatchEnvSample::new(
            config.env_config.clone(),
            create_n_env::<E>(config.env_config.clone()),
        );
        println!("env_sampler create time={:?}", start.elapsed().unwrap());
        let exp_base_path = format!("{}/{}", config.ckpt_save_path, exp_name);
        std::fs::create_dir_all(&exp_base_path);

        let eval_env = E::new(true, config.env_config.clone());
        let action_dim = eval_env.get_action_dim();
        Self {
            device: device,
            algo_name: algo_name.to_string(),
            backend: PhantomData,
            action_dim,
            env_name: env_name.to_string(),
            env_sampler,
            config,
            eval_env,
            exp_name,
            exp_base_path,
        }
    }

    pub fn get_action<AM: ActorModel<B> + Display>(
        obs: Array2<f64>,
        actor: AM,
        device: B::Device,
        is_eval: bool,
    ) -> Array2<f64> {
        let input = ndarray2tensor2(obs, &device);
        let input = input.unsqueeze_dim::<3>(1); // (batch_size, seq_length, obs_dim), seq_length=1
        let batch_size = input.shape().dims[0];
        let obs_dim = input.shape().dims[2];
        let traj_length = Tensor::ones([batch_size], &device);
        let seq_mask = Tensor::<B::InnerBackend, 2>::ones([batch_size, 1], &device);
        // let mask = mask
        //     .repeat_dim(1, self.action_dim)
        //     .reshape([batch_size, self.action_dim, 1])
        //     .swap_dims(1, 2);

        let action_dis = actor.clone().eval_forward(input, traj_length, seq_mask);
        let action = if is_eval {
            action_dis.mean().squeeze::<2>(1)
        } else {
            action_dis.sample().squeeze::<2>(1)
        };

        let action = tensor2ndarray2::<B::InnerBackend, crate::FType, Float>(&action);
        return action.map(|x| x.to_f64());
    }
    pub fn sample_to_memory<AM: ActorModel<B> + Display>(
        &mut self,
        actor: AM,
        iter: usize,
    ) -> Memory<B> {
        let policy =
            |obs: Array2<f64>| Self::get_action(obs, actor.clone(), self.device.clone(), false);
        let trajs = self.env_sampler.sample_n_trajectories(&policy);
        if iter > 0 && iter % self.config.video_log_freq == 0 {
            println!("iter={} video log", iter);
            let start = SystemTime::now();
            self.eval_env.run_policy(
                &format!("{}_{}_iter{}", self.algo_name, self.env_name, iter),
                self.config.env_config.max_traj_length,
                &|obs: Array2<f64>| Self::get_action(obs, actor.clone(), self.device.clone(), true),
            );
            println!("eval_env run_policy time={:?}", start.elapsed().unwrap());
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
        // let mut log_info = HashMap::<String, f32>::new();
        for iter in 0..self.config.train_iter {
            println!("actor_net.std_mean()={}", actor_net.std_mean());
            let mut start = SystemTime::now();
            let memory = self.sample_to_memory(actor_net.clone(), iter);

            let mean_reward = memory
                .reward()
                .clone()
                .div_scalar(self.config.env_config.n_env as f32)
                .sum()
                .into_scalar()
                .to_f32();
            EpochLogger::add_scalar(("train", "mean_reward"), mean_reward);
            let traj_length = memory.traj_length();
            EpochLogger::add_scalar(
                ("train", "step_num"),
                traj_length.clone().sum().into_scalar().to_f32(),
            );
            EpochLogger::add_scalar(
                ("train", "collect_time"),
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

            EpochLogger::add_scalar(
                ("train", "ppo_update_time"),
                start.elapsed().unwrap().as_millis() as f32,
            );
            EpochLogger::add_scalar(("train", "actor_loss"), update_info.actor_loss.to_f32());
            EpochLogger::add_scalar(("train", "critic_loss"), update_info.critic_loss.to_f32());
            EpochLogger::add_scalar(("train", "mean_q_val"), update_info.mean_q_val.to_f32());
            EpochLogger::add_scalar(
                ("train", "actor_std"),
                actor_net.std_mean().into_scalar().to_f32(),
            );
            EpochLogger::log(iter);
        }
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

        let file_recorder = DefaultFileRecorder::<crate::MyPrecisionSettings>::new();

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
        let file_recorder = DefaultFileRecorder::<crate::MyPrecisionSettings>::new();
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
        println!("actor_net={}", actor_net);
        println!("baseline_net={}", baseline_net);
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
