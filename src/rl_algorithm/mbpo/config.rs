// use crate::base::ElemType;
use burn::config::Config;

#[derive(Config, Debug)]
pub struct MbTrainingConfig {
    pub learning_rate: f32,
}

impl Default for MbTrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
        }
    }
}
