// #![allow(warnings)]

mod mujoco;
mod rl_algorithm;
mod rl_env;

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::wgpu::WgpuDevice;
// use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, NdArray, Wgpu};
use burn::tensor::{Shape, Tensor, TensorData};
use mujoco::ffi::mj_step;
use ndarray::{stack, Array, Array1, Array2, ArrayView1, ArrayView2, Axis};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rl_algorithm::policy_gradient::policy;
use rl_env::env_sampler::Trajectory;
use rl_env::{env, env_sampler};
use std::path::Path;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use ndarray as nd;
use ndarray::Array3;

use video_rs::encode::{Encoder, Settings};
use video_rs::time::Time;

fn to_ndarray(vec: &Vec<Trajectory>) -> (Array3<f64>, Array2<f64>, Array3<f64>, Array2<u8>) {
    let observation = vec
        .iter()
        .map(|traj| traj.observation.view())
        .collect::<Vec<ArrayView2<f64>>>();
    let reward = vec
        .iter()
        .map(|traj| traj.reward.view())
        .collect::<Vec<ArrayView1<f64>>>();
    let action = vec
        .iter()
        .map(|traj| traj.action.view())
        .collect::<Vec<ArrayView2<f64>>>();
    // let next_observation = vec
    //     .iter()
    //     .map(|traj| traj.next_observation.view())
    //     .collect::<Vec<ArrayView2<f64>>>();
    let terminal = vec
        .iter()
        .map(|traj| traj.terminal.view())
        .collect::<Vec<ArrayView1<u8>>>();
    let res = (
        nd::stack(Axis(0), &observation).unwrap(),
        nd::stack(Axis(0), &reward).unwrap(),
        nd::stack(Axis(0), &action).unwrap(),
        // nd::stack(Axis(0), &next_observation).unwrap(),
        nd::stack(Axis(0), &terminal).unwrap(),
    );
    return res;
}
fn main() {
    let mut env = env::Env::new("humanoid.xml", false);
    let action_dim = env.get_action_len();
    let ob_dim = env.get_obs_len();
    println!("ob_dim={}", ob_dim);
    println!("action_dim={}", action_dim);
    // let random_policy = move |obs: &Array1<f64>| {
    //     return Array::random(action_dim, Uniform::new(-1.0, 1.0));
    // };
    let learn_rate = 1e-3;
    let reward_discout = 0.99;
    let batch_size = 5000;
    let traj_length = 1000;
    let layer_dim = 128;
    let n_layers = 3;
    let n_iter = 100;
    let baseline_gradient_steps = 5;
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
        );
    println!("conf={}", pg_agent_conf);
    let pg_agent = Arc::new(Mutex::new(
        pg_agent_conf.init::<Autodiff<NdArray>>(&NdArrayDevice::default()),
    ));
    let policy = |obs: &Array2<f64>| pg_agent.lock().unwrap().get_action(obs);
    let start = SystemTime::now();
    let mut env_sampler = env_sampler::BatchEnvSample::new("humanoid.xml", traj_length, batch_size);
    println!("init BatchEnvSample={:?}", start.elapsed());
    for iter in 0..n_iter {
        let start = SystemTime::now();
        let trajs = env_sampler.sample_n_trajectories(&policy);
        // break;
        let (obs, reward, action, terminal) = to_ndarray(&trajs);
        println!("sample traj cost={:?}", start.elapsed());
        for _ in 0..1 {
            let update_info = pg_agent
                .lock()
                .unwrap()
                .update(&obs, &action, &reward, &terminal);
            println!(
                "iter={} update cost={:?} update_info={:?}",
                iter,
                start.elapsed(),
                update_info
            );
        }
    }

    // println!("{:?}", trajs);
}
