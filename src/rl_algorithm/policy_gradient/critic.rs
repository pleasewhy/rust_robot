use crate::rl_algorithm::normal;
use crate::rl_algorithm::utils;
use burn::nn::loss::Reduction;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::Adam;
use burn::optim::AdamConfig;
use burn::optim::GradientsParams;
use burn::optim::Optimizer;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::cast::ToElement;
use burn::{nn::Linear, nn::LinearConfig, prelude::*};

#[derive(Module, Debug)]
pub struct MLPCritic<B: Backend> {
    net: utils::Mlp<B>,
}

impl<B: Backend> MLPCritic<B> {
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        return self.net.forward::<D>(input);
    }
}

#[derive(Config, Debug)]
pub struct MLPCriticConfig {
    observation_dim: usize,
    n_layers: usize,
    layer_size: usize,
}

impl MLPCriticConfig {
    /// Returns the initialized Mlp.
    pub fn init<B: Backend>(&self, device: &B::Device) -> MLPCritic<B> {
        let net = utils::MlpConfig::new(self.observation_dim, 1, self.n_layers, self.layer_size)
            .init::<B>(&device);
        return MLPCritic { net };
    }
}

pub struct MLPCriticTrainer<B: AutodiffBackend> {
    pub mlp_critic: MLPCritic<B>,
    pub adam: OptimizerAdaptor<Adam, MLPCritic<B>, B>,
    pub lr: f64,
}

impl<B: AutodiffBackend> MLPCriticTrainer<B> {
    pub fn new(
        device: &B::Device,
        mlp_critic_config: MLPCriticConfig,
        adam_config: AdamConfig,
        learn_rate: f64,
    ) -> Self {
        return Self {
            mlp_critic: mlp_critic_config.init(device),
            adam: adam_config.init(),
            lr: learn_rate,
        };
    }
    pub fn train_update(&mut self, obs: Tensor<B, 2>, q_values: Tensor<B, 1>) -> f32 {
        let pred_q_values = self.mlp_critic.forward(obs).flatten::<1>(0, 1);
        let loss = burn::nn::loss::MseLoss::new().forward(pred_q_values, q_values, Reduction::Mean);
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.mlp_critic);

        self.mlp_critic = self.adam.step(self.lr, self.mlp_critic.clone(), grads);
        return loss.into_scalar().to_f32();
    }
}
