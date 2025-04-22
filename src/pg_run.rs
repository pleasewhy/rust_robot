// #![allow(warnings)]

use crate::rl_algorithm::base::config::TrainConfig;
use crate::rl_algorithm::base::model::RlTrainAlgorithm;
use crate::rl_algorithm::base::on_policy_runner;
use crate::rl_algorithm::policy_gradient::config::PgTrainingConfig;
use crate::rl_algorithm::policy_gradient::pg_agent::PolicyGradient;
use crate::rl_algorithm::ppo::config::PPOTrainingConfig;
use crate::rl_algorithm::preload_net::mlp_critic::{MLPCritic, MLPCriticConfig};
use crate::rl_algorithm::preload_net::mlp_policy::{MLPPolicy, MLPPolicyConfig};
use crate::rl_env::env::{EnvConfig, MujocoEnv};
use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
// use burn::backend::libtorch::LibTorchDevice;
// use burn::backend::{Autodiff, LibTorch};
use burn::grad_clipping::GradientClippingConfig;
use burn::module::Module;
use burn::nn::{self, LinearConfig};
use burn::record::{DefaultFileRecorder, FullPrecisionSettings};

pub fn train_network<ENV: MujocoEnv + Send + 'static>() {
    let pg_train_config = PgTrainingConfig {
        gae_gamma: 0.97,
        reward_lambda: 0.99,
        learning_rate: 1e-3,
        entropy_coef: 0.0,
        baseline_update_freq: 10,
    };

    let config = TrainConfig {
        pg_train_config,
        video_log_freq: 100,
        train_iter: 10000,
        ckpt_save_path: "./ckpt".to_string(),
        // resume_from_ckpt_path: None,
        resume_from_ckpt_path: Some(
            "ckpt/ppo_HumanoidV4_04-11 12:04/iter600_mean_reward1240.635".to_string(),
        ),
        save_model_freq: 100,
        mujoco_simulate_thread_num: 6,
        grad_clip: Some(GradientClippingConfig::Norm(1.0)),
        env_config: EnvConfig {
            n_env: 500,
            traj_length: 1000,
            reset_state_use_n_step_before_last_failed: 30,
            use_init_state_ratio: 0.3,
        },
        ..Default::default()
    };

    let test_env = ENV::new(false, config.env_config.clone());

    let ob_dim = test_env.get_obs_dim();
    let action_dim = test_env.get_action_dim();
    type MyBackend = NdArray;
    type MyDevice = NdArrayDevice;
    let device = MyDevice::Cpu;
    let mut actor_net =
        MLPPolicyConfig::new(action_dim, ob_dim, 3, 256).init::<Autodiff<MyBackend>>(&device);
    let baseline_net = MLPCriticConfig::new(ob_dim, 3, 256).init::<Autodiff<MyBackend>>(&device);
    println!("actor_net={}", actor_net);
    println!("baseline_net={}", baseline_net);
    actor_net.logstd = LinearConfig::new(1, action_dim)
        .with_bias(false)
        .with_initializer(nn::Initializer::Zeros)
        .init::<Autodiff<MyBackend>>(&device);
    let mut runner = on_policy_runner::OnPolicyRunner::<ENV, Autodiff<MyBackend>>::new(
        device,
        config,
        "policy_gradient",
    );
    let pg_algo = PolicyGradient::new();
    runner.train_update(actor_net, baseline_net, pg_algo);
}
