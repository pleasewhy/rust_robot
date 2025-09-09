use burn::config::Config;

#[derive(Config, Debug)]

pub enum TruncateStrategy {
    None,
    DonedEnvRatio(f32),
}

impl TruncateStrategy {
    pub fn truncate(&self, done_ratio: f32) -> bool {
        match self {
            TruncateStrategy::DonedEnvRatio(ratio) => {
                return done_ratio >= *ratio;
            }
            TruncateStrategy::None => return false,
        }
    }
}
#[derive(Config, Debug)]
pub struct EnvConfig {
    pub n_env: usize,
    pub max_traj_length: usize,
    pub reset_state_use_n_step_before_last_failed: usize,
    pub use_init_state_ratio: f64,
    pub truncate_strategy: TruncateStrategy,
    pub sample_thread_num: usize,
    pub skip_steps: usize,
}

impl Default for EnvConfig {
    fn default() -> Self {
        Self {
            n_env: 100,
            max_traj_length: 1000,
            reset_state_use_n_step_before_last_failed: 50,
            use_init_state_ratio: 0.3,
            truncate_strategy: TruncateStrategy::DonedEnvRatio(1.0),
            sample_thread_num: 4,
            skip_steps: 1,
        }
    }
}
