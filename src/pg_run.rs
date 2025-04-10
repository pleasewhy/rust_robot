// #![allow(warnings)]

use crate::rl_algorithm::base::config::TrainConfig;
use crate::rl_algorithm::base::model::RlTrainAlgorithm;
use crate::rl_algorithm::base::on_policy_runner;
use crate::rl_algorithm::policy_gradient::config::PgTrainingConfig;
use crate::rl_algorithm::policy_gradient::pg_agent::PolicyGradient;
use crate::rl_algorithm::ppo::config::PPOTrainingConfig;
use crate::rl_algorithm::preload_net::mlp_critic::{MLPCritic, MLPCriticConfig};
use crate::rl_algorithm::preload_net::mlp_policy::{MLPPolicy, MLPPolicyConfig};
use crate::rl_env::env::MujocoEnv;
use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use burn::grad_clipping::GradientClippingConfig;
use burn::module::Module;
use burn::record::{DefaultFileRecorder, FullPrecisionSettings};

type MyBackend = Wgpu;
type MyDevice = WgpuDevice;

pub fn train_network<ENV: MujocoEnv + Send + 'static>() {
    let test_env = ENV::new(true);

    let ob_dim = test_env.get_obs_dim();
    let action_dim = test_env.get_action_dim();

    let pg_train_config = PgTrainingConfig {
        gae_gamma: 0.97,
        reward_lambda: 0.99,
        learning_rate: 1e-3,
        entropy_coef: 0.0,
        baseline_update_freq: 10,
    };

    let config = TrainConfig {
        pg_train_config,
        n_env: 1000,
        traj_length: 1000,
        video_log_freq: 100,
        train_iter: 10000,
        ckpt_save_path: "./ckpt".to_string(),
        resume_from_ckpt_path: None,
        // resume_from_ckpt_path: Some("./ckpt/iter200_mean_reward461.66412".to_string()),
        save_model_freq: 100,
        grad_clip: Some(GradientClippingConfig::Norm(1.0)),
        ..Default::default()
    };

    let device = MyDevice::DefaultDevice;
    let actor_net =
        MLPPolicyConfig::new(action_dim, ob_dim, 3, 256).init::<Autodiff<MyBackend>>(&device);
    let baseline_net = MLPCriticConfig::new(ob_dim, 3, 256).init::<Autodiff<MyBackend>>(&device);
    println!("actor_net={}", actor_net);
    println!("baseline_net={}", baseline_net);

    let mut runner = on_policy_runner::OnPolicyRunner::<ENV, Autodiff<MyBackend>>::new(
        device,
        config,
        "policy_gradient",
    );
    let pg_algo = PolicyGradient::new();
    runner.train_update(actor_net, baseline_net, pg_algo);
}
