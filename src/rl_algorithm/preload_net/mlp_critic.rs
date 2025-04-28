use crate::burn_utils::{build_mlp, Sequence};
use crate::rl_algorithm::base::model::BaselineModel;
use crate::rl_algorithm::base::rl_utils::tensor2ndarray1;
use burn::prelude::*;
use ndarray::Array1;

#[derive(Module, Debug)]
pub struct MLPCritic<B: Backend> {
    net: Sequence<B>,
}

impl<B: Backend> BaselineModel<B> for MLPCritic<B> {
    fn forward(
        &self,
        input: Tensor<B, 3>,
        traj_length: Tensor<B, 1, Int>,
        _: Tensor<B, 3>,
    ) -> Tensor<B, 2> {
        let batch_size = input.shape().dims[0];
        let seq_length = input.shape().dims[1];
        let device = &self.net.devices()[0];

        let flatten_indices =
            super::utils::generate_flatten_idx(tensor2ndarray1(&traj_length), seq_length, device);

        let input = input.flatten::<2>(0, 1);
        let input = input.select(0, flatten_indices.clone());

        let output = self.net.forward::<2>(input).flatten::<1>(0, 1);
        let zero_output = Tensor::zeros([batch_size * seq_length], device);

        let output = zero_output
            .select_assign(0, flatten_indices, output.clone())
            .reshape([batch_size, seq_length]);
        return output;
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
