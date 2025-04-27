use crate::burn_utils::distribution::Normal;
use crate::burn_utils::{build_mlp, Sequence};
use crate::rl_algorithm::base::model::ActorModel;
use crate::rl_algorithm::base::rl_utils::{ndarray2tensor1, tensor2ndarray1, tensor2ndarray2};
use burn::nn::Tanh;
use burn::{nn::Linear, nn::LinearConfig, prelude::*};
use ndarray::{Array1, AssignElem};

#[derive(Module, Debug)]
pub struct MLPPolicy<B: Backend> {
    mean_net: Sequence<B>,
    pub logstd: Linear<B>,
    tanh: Tanh,
}

impl<B: Backend> ActorModel<B> for MLPPolicy<B> {
    fn forward(
        &self,
        input: Tensor<B, 3>,
        traj_length: Tensor<B, 1, Int>,
        mask: Tensor<B, 3>,
    ) -> Normal<B> {
        let batch_size = input.shape().dims[0];
        let seq_length = input.shape().dims[1];

        let device = &self.mean_net.devices()[0];
        let flatten_indices =
            super::utils::generate_flatten_idx(tensor2ndarray1(&traj_length), seq_length, device);
        let input = input.flatten::<2>(0, 1);

        let input = input.select(0, flatten_indices.clone());
        let mean = self.mean_net.forward::<2>(input);
        let mean = self.tanh.forward(mean);
        let ac_dim = mean.shape().dims[1];
        let zero_mean = Tensor::zeros([batch_size * seq_length, ac_dim], device);
        // println!("mean.shape={:?}", mean.shape());
        // println!("zero_mean.shape={:?}", zero_mean.shape());
        // println!("flatten_indices.shape={:?}", flatten_indices.shape());

        let mean = zero_mean
            .select_assign(0, flatten_indices, mean.clone())
            .reshape([batch_size, seq_length, ac_dim]);

        let logstd = self.logstd.weight.val().flatten(0, 1);
        return Normal::new(mean, logstd.exp().powi_scalar(2), None);
    }

    fn std_mean(&self) -> Tensor<B, 1> {
        return self.logstd.weight.val().mean();
    }
}

// impl<B: Backend> MLPPolicy<B> {
//     pub fn forward(&self, input: Tensor<B, 2>) -> Normal<B> {
//         let mean: Tensor<B, 2> = self.mean_net.forward::<2>(input);
//         let mean = self.tanh.forward(mean);
//         let logstd = self.logstd.weight.val().flatten(0, 1);

//         return Normal::new(mean, logstd.exp().powi_scalar(2));
//     }
// }

#[derive(Config, Debug)]
pub struct MLPPolicyConfig {
    action_dim: usize,
    observation_dim: usize,
    n_layers: usize,
    layer_size: usize,
}

impl MLPPolicyConfig {
    /// Returns the initialized Mlp.
    pub fn init<B: Backend>(&self, device: &B::Device) -> MLPPolicy<B> {
        let mean_net: Sequence<B> = build_mlp(
            self.observation_dim,
            self.action_dim,
            self.n_layers,
            self.layer_size,
            device,
        );
        let logstd = LinearConfig::new(1, self.action_dim)
            .with_bias(false)
            .with_initializer(nn::Initializer::Zeros)
            .init::<B>(&device);
        return MLPPolicy {
            mean_net,
            logstd,
            tanh: Tanh::new(),
        };
    }
}
