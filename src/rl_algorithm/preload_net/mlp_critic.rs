use crate::burn_utils::{build_mlp, Sequence};
use crate::rl_algorithm::base::model::BaselineModel;
use crate::rl_algorithm::base::rl_utils::tensor2ndarray1;
use burn::module::AutodiffModule;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use ndarray::Array1;

#[derive(Module, Debug)]
pub struct MLPCritic<B: Backend> {
    net: Sequence<B>,
}

impl<B: Backend> MLPCritic<B> {
    fn forward(
        &self,
        input: Tensor<B, 3>,
        traj_length: Tensor<B, 1, Int>,
        seq_mask: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let batch_size = input.shape().dims[0];
        let seq_length = input.shape().dims[1];
        let device = &self.net.devices()[0];

        let output = self.net.forward(input);
        let output = output.flatten::<2>(1, 2).mul(seq_mask);
        return output;
    }
}

impl<B: AutodiffBackend> BaselineModel<B> for MLPCritic<B> {
    fn autodiff_forward(
        &self,
        input: Tensor<B, 3>,
        traj_length: Tensor<B, 1, Int>,
        seq_mask: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        return self.forward(input, traj_length, seq_mask);
    }
    fn eval_forward(
        &self,
        input: Tensor<B::InnerBackend, 3>,
        traj_length: Tensor<B::InnerBackend, 1, Int>,
        seq_mask: Tensor<B::InnerBackend, 2>,
    ) -> Tensor<B::InnerBackend, 2> {
        return self.valid().forward(input, traj_length, seq_mask);
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
