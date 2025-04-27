// #![allow(warnings)]

use crate::rl_algorithm::base::config::TrainConfig;
use crate::rl_algorithm::base::model::RlTrainAlgorithm;
use crate::rl_algorithm::base::on_policy_runner;
use crate::rl_algorithm::ppo::config::PPOTrainingConfig;
use crate::rl_algorithm::ppo::ppo_agent::PPO;
use crate::rl_algorithm::preload_net::mlp_critic::MLPCriticConfig;
use crate::rl_algorithm::preload_net::mlp_policy::MLPPolicyConfig;
use crate::rl_algorithm::preload_net::rnn_policy::LstmPolicyConfig;
use crate::rl_env::config::{EnvConfig, TruncateStrategy};
use crate::rl_env::env::MujocoEnv;
use burn::backend::libtorch::LibTorchDevice;
use burn::backend::ndarray::NdArrayDevice;
use burn::backend::wgpu::WgpuDevice;
use burn::backend::Autodiff;
use burn::backend::LibTorch;
use burn::backend::NdArray;
use burn::backend::Wgpu;
use burn::config::Config;
use burn::grad_clipping::GradientClippingConfig;

pub fn train_network<ENV: MujocoEnv + Send + 'static>() {
    let ppo_train_config = PPOTrainingConfig {
        gae_gamma: 0.97,
        reward_lambda: 0.99,
        learning_rate: 1e-4,
        entropy_coef: 0.0,
        epsilon_clip: 0.2,
        update_freq: 20,
        mini_batch_size: 400,
    };

    let config = TrainConfig {
        ppo_train_config,
        video_log_freq: 100,
        train_iter: 10000,
        ckpt_save_path: "./ckpt".to_string(),
        resume_from_ckpt_path: None,
        // resume_from_ckpt_path: Some(
        //     "ckpt/ppo_MobileArm_04-23 18:53/iter600_mean_reward17.587719".to_string(),
        // ),
        save_model_freq: 100,
        grad_clip: Some(GradientClippingConfig::Norm(1.0)),
        env_config: EnvConfig {
            n_env: 600,
            max_traj_length: 1000,
            truncate_strategy: TruncateStrategy::DonedEnvRatio(1.0),
            sample_thread_num: 6,
            ..Default::default()
        },
        ..Default::default()
    };
    let test_env = ENV::new(true, config.env_config.clone());
    let ob_dim = test_env.get_obs_dim();
    let action_dim = test_env.get_action_dim();
    println!("ob_dim={}, action_dim={}", ob_dim, action_dim);
    type MyBackend = LibTorch;
    type MyDevice = LibTorchDevice;
    let device = MyDevice::Cuda(0);
    let mut actor_net =
        MLPPolicyConfig::new(action_dim, ob_dim, 3, 256).init::<Autodiff<MyBackend>>(&device);
    let baseline_net = MLPCriticConfig::new(ob_dim, 3, 256).init::<Autodiff<MyBackend>>(&device);
    println!("actor_net={}", actor_net);
    println!("baseline_net={}", baseline_net);
    let mut runner =
        on_policy_runner::OnPolicyRunner::<ENV, Autodiff<MyBackend>>::new(device, config, "ppo");
    let ppo_algo = PPO::new();
    runner.train_update(actor_net, baseline_net, ppo_algo);
    loop {}
}
