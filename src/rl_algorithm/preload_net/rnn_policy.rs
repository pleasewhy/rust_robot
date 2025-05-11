use crate::rl_algorithm::base::rl_utils::{ndarray2tensor1, tensor2ndarray1, tensor2ndarray2};
use burn::{
    config::Config,
    module::Module,
    nn::{self, LinearConfig, LstmConfig, Tanh},
    prelude::Backend,
    tensor::{Int, Tensor},
};
use video_rs::ffmpeg::format::Output;

use crate::{burn_utils::distribution::Normal, rl_algorithm::base::model::ActorModel};

#[derive(Module, Debug)]
pub struct LstmPolicy<B: Backend> {
    lstm: burn::nn::Lstm<B>,
    output: burn::nn::Linear<B>,
    logstd: burn::nn::Linear<B>,
    tanh: Tanh,
}

impl<B: Backend> ActorModel<B> for LstmPolicy<B> {
    fn forward(
        &self,
        input: Tensor<B, 3>,
        flatten_indices: Tensor<B, 1, Int>,
        mask: Tensor<B, 3>,
    ) -> Normal<B> {
        let mean = self.lstm.forward(input, None).0;
        let mean = self.output.forward(mean);
        let mean = self.tanh.forward(mean);
        let logstd = self.logstd.weight.val().flatten(0, 1);

        return Normal::new(mean, logstd.exp().powi_scalar(2), Some(mask));
    }

    fn std_mean(&self) -> Tensor<B, 1> {
        return self.logstd.weight.val().mean();
    }
}

#[derive(Config, Debug)]
pub struct LstmPolicyConfig {
    action_dim: usize,
    observation_dim: usize,
    layer_size: usize,
}

impl LstmPolicyConfig {
    /// Returns the initialized Mlp.
    pub fn init<B: Backend>(&self, device: &B::Device) -> LstmPolicy<B> {
        let lstm = LstmConfig::new(self.observation_dim, self.layer_size, true).init(device);
        let output = LinearConfig::new(self.layer_size, self.action_dim).init::<B>(&device);
        let logstd = LinearConfig::new(1, self.action_dim)
            .with_bias(false)
            .with_initializer(nn::Initializer::Zeros)
            .init::<B>(&device);
        return LstmPolicy {
            lstm,
            output,
            logstd,
            tanh: Tanh::new(),
        };
    }
}
