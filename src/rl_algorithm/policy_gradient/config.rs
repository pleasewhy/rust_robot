// use crate::base::ElemType;
use burn::grad_clipping::GradientClippingConfig;

pub struct PgTrainingConfig {
    pub gae_gamma: f32,
    pub reward_lambda: f32,
    pub learning_rate: f32,
    pub baseline_update_freq: usize,
    pub entropy_coef: f32,
}

impl Default for PgTrainingConfig {
    fn default() -> Self {
        Self {
            gae_gamma: 0.97,
            reward_lambda: 0.99,
            learning_rate: 1e-3,
            baseline_update_freq: 5,
            entropy_coef: 0.0,
        }
    }
}
