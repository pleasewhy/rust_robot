use burn::{config::Config, grad_clipping::GradientClippingConfig};

use crate::{
    rl_algorithm::{policy_gradient::config::PgTrainingConfig, ppo::config::PPOTrainingConfig},
    rl_env::config::EnvConfig,
};

#[derive(Config)]
pub struct TrainConfig {
    pub ppo_train_config: PPOTrainingConfig,
    pub pg_train_config: PgTrainingConfig,
    pub video_log_freq: usize,
    pub train_iter: usize,
    pub ckpt_save_path: String,
    pub resume_from_ckpt_path: Option<String>,
    pub save_model_freq: usize,
    pub grad_clip: Option<GradientClippingConfig>,
    pub env_config: EnvConfig,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            ppo_train_config: PPOTrainingConfig::default(),
            pg_train_config: PgTrainingConfig::default(),
            video_log_freq: 100,
            train_iter: 10000,
            ckpt_save_path: "./ckpt".to_string(),
            resume_from_ckpt_path: None,
            save_model_freq: 100,
            grad_clip: Some(GradientClippingConfig::Norm(1.0)),
            env_config: EnvConfig::default(),
        }
    }
}
