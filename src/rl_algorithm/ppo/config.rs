
pub struct PPOTrainingConfig {
    pub gae_gamma: f32,
    pub reward_lambda: f32,
    pub epsilon_clip: f32,
    pub learning_rate: f32,
    pub update_freq: usize,
    pub mini_batch_size: usize,
    pub entropy_coef: f32,
}

impl Default for PPOTrainingConfig {
    fn default() -> Self {
        Self {
            gae_gamma: 0.97,
            reward_lambda: 0.99,
            epsilon_clip: 0.2,
            learning_rate: 1e-3,
            update_freq: 5,
            mini_batch_size: 1000,
            entropy_coef: 0.2,
        }
    }
}
