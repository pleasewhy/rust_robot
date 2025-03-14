// #![allow(warnings)]

mod burn_utils;
mod mujoco;
mod rl_algorithm;
mod rl_env;

use burn::backend::libtorch::LibTorchDevice;
use burn::backend::{Autodiff, LibTorch};
use burn::tensor::{Distribution, Shape, Tensor};
use chrono::{DateTime, Local};
use ndarray::Array1;
use rl_env::env::MujocoEnv;
use rl_env::env_sampler;
use rl_env::gym_humanoid_v4::HumanoidV4;
use rl_env::inverted_pendulum_v4::InvertedPendulumV4;
use rl_env::nd_vec::NdVec2;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;
use tensorboard_rs::summary_writer::SummaryWriter;

fn create_n_env<ENV: MujocoEnv>(n_env: usize) -> Vec<Arc<Mutex<ENV>>> {
    let mut envs = vec![];
    for _ in 0..n_env {
        envs.push(Arc::new(Mutex::new(ENV::new(false))));
    }
    return envs;
}

fn run<ENV: MujocoEnv + Send + 'static>() {
    let mut eval_env = ENV::new(true);
    let ob_dim = eval_env.get_obs_dim();
    let action_dim = eval_env.get_action_dim();
    let learn_rate = 1e-3;
    let reward_discout = 0.99;
    let batch_size = 100;
    let traj_length = 1000;
    let layer_dim = 256;
    let n_layers = 3;
    let n_iter = 10000;
    let baseline_gradient_steps = 5;
    let video_log_freq = 100;

    let pg_agent_conf =
        rl_algorithm::policy_gradient::policy_gradient_agent::PolicyGradientAgentConfig::new(
            ob_dim,
            action_dim,
            false,
            n_layers,
            layer_dim,
            learn_rate,
            reward_discout,
            baseline_gradient_steps,
            0.97f32,
        );
    println!("conf={}", pg_agent_conf);
    let pg_agent = Rc::new(RefCell::new(
        pg_agent_conf.init::<Autodiff<LibTorch>>(&LibTorchDevice::Cpu),
    ));
    let policy = |obs: NdVec2<f64>| pg_agent.borrow_mut().get_action(obs);
    let start = SystemTime::now();

    let n_env = batch_size;
    let envs = create_n_env::<ENV>(n_env);

    let mut env_sampler = env_sampler::BatchEnvSample::new(traj_length, 4, envs);
    println!("init BatchEnvSample={:?}", start.elapsed());
    let env_name = std::any::type_name::<ENV>().rsplit("::").next().unwrap();
    let exp_name = format!(
        "./logdir/{}_{}",
        env_name,
        Local::now().format("%d/%m/%Y %H:%M")
    );
    let mut writer = SummaryWriter::new(&exp_name);
    for iter in 0..n_iter {
        let mut actor_loss = HashMap::<String, f32>::new();
        let mut critic_loss = HashMap::<String, f32>::new();
        let mut q_values = HashMap::<String, f32>::new();
        let mut batch_size = HashMap::<String, f32>::new();
        let mut mean_reward = HashMap::<String, f32>::new();
        let start = SystemTime::now();
        let trajs = env_sampler.sample_n_trajectories(&policy);
        let update_info = pg_agent.borrow_mut().update(
            trajs.observation,
            trajs.action,
            trajs.reward,
            trajs.terminal,
        );
        if iter % video_log_freq == 0 {
            eval_env.run_policy(
                &format!("{}_iter{}.mp4", env_name, iter),
                traj_length,
                &policy,
            );
        }
        actor_loss.insert("actor_loss".to_string(), update_info.actor_loss);
        critic_loss.insert("critic_loss".to_string(), update_info.critic_loss);
        mean_reward.insert("mean_reward".to_string(), update_info.mean_reward);
        batch_size.insert("batch_size".to_string(), update_info.batch_size as f32);
        q_values.insert("mean_q_val".to_string(), update_info.mean_q_val as f32);
        println!(
            "iter={} update cost={:?} update_info={:?}",
            iter,
            start.elapsed(),
            update_info
        );
        writer.add_scalars("humanoid/actor_loss", &actor_loss, iter);
        writer.add_scalars("humanoid/critic_loss", &critic_loss, iter);
        writer.add_scalars("humanoid/mean_reward", &mean_reward, iter);
        writer.add_scalars("humanoid/batch_size", &batch_size, iter);
        writer.add_scalars("humanoid/q_values", &q_values, iter);
        writer.flush();
    }
}
fn main() {
    run::<HumanoidV4>();
}
