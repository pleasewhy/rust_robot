use super::super::ppo::model::BaselineModel;
use crate::burn_utils::{build_mlp, Sequence};
use burn::prelude::*;

#[derive(Module, Debug)]
pub struct MLPCritic<B: Backend> {
    net: Sequence<B>,
}

impl<B: Backend> BaselineModel<B> for MLPCritic<B> {
    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 1> {
        return self.net.forward::<2>(input).flatten(0, 1);
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
        let net = build_mlp(
            self.observation_dim,
            1,
            self.n_layers,
            self.layer_size,
            device,
        );
        return MLPCritic { net };
    }
}
