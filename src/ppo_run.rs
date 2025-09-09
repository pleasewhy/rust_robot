// #![allow(warnings)]

use crate::rl_algorithm::base::config::TrainConfig;
use crate::rl_algorithm::base::model::RlTrainAlgorithm;
use crate::rl_algorithm::base::on_policy_runner;
use crate::rl_algorithm::ppo::config::PPOTrainingConfig;
use crate::rl_algorithm::ppo::ppo_agent::PPO;
use crate::rl_algorithm::preload_net::mlp_critic::MLPCriticConfig;
use crate::rl_algorithm::preload_net::normal_mlp_policy::{NormalMLPPolicy, NormalMLPPolicyConfig};
use crate::rl_env::config::{EnvConfig, TruncateStrategy};
use crate::rl_env::env::MujocoEnv;
use burn::backend::wgpu;
use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::config::Config;
use burn::grad_clipping::GradientClippingConfig;
use burn::module::AutodiffModule;
use burn::tensor::Tensor;

pub fn train_network<ENV: MujocoEnv + Send + 'static>() {
    let ppo_train_config = PPOTrainingConfig {
        gae_gamma: 0.97,
        reward_lambda: 0.99,
        learning_rate: 1e-4,
        entropy_coef: 0.01,
        epsilon_clip: 0.15,
        update_freq: 10,
        mini_batch_size: 500,
    };

    let config = TrainConfig {
        ppo_train_config,
        video_log_freq: 100,
        train_iter: 10000,
        ckpt_save_path: "./ckpt".to_string(),
        // resume_from_ckpt_path: None,
        // resume_from_ckpt_path: Some(
        //     "ckpt/ppo_HumanoidV4_05-12 23:45/iter1300_mean_reward4084".to_string(),
        // ),
        // resume_from_ckpt_path: Some(
        //     "ckpt/ppo_HumanoidV4_05-13 15:35/iter600_mean_reward819".to_string(),
        // ),
        save_model_freq: 100,
        grad_clip: Some(GradientClippingConfig::Value(100.0)),
        env_config: EnvConfig {
            n_env: 500,
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

    let device = crate::MyDevice::default();
    let mut actor_net = NormalMLPPolicyConfig::new(action_dim, ob_dim, 3, 256)
        .init::<Autodiff<crate::MyBackend>>(&device);
    let baseline_net =
        MLPCriticConfig::new(ob_dim, 3, 256).init::<Autodiff<crate::MyBackend>>(&device);
    println!("actor_net={}", actor_net);
    println!("baseline_net={}", baseline_net);
    let mut runner = on_policy_runner::OnPolicyRunner::<ENV, Autodiff<crate::MyBackend>>::new(
        device, config, "ppo",
    );
    let ppo_algo = PPO::new();
    runner.train_update(actor_net, baseline_net, ppo_algo);
    loop {}
}
