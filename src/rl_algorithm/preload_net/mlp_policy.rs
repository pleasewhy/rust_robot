use crate::burn_utils::distribution::Normal;
use crate::burn_utils::{build_mlp, Sequence};
use crate::rl_algorithm::base::model::ActorModel;
use burn::nn::Tanh;
use burn::{nn::Linear, nn::LinearConfig, prelude::*};
use ndarray::AssignElem;

#[derive(Module, Debug)]
pub struct MLPPolicy<B: Backend> {
    mean_net: Sequence<B>,
    pub logstd: Linear<B>,
    tanh: Tanh,
}

impl<B: Backend> ActorModel<B> for MLPPolicy<B> {
    fn forward(&self, input: Tensor<B, 2>) -> Normal<B> {
        let mean: Tensor<B, 2> = self.mean_net.forward::<2>(input);
        let mean = self.tanh.forward(mean);
        let logstd = self.logstd.weight.val().flatten(0, 1);

        return Normal::new(mean, logstd.exp().powi_scalar(2));
    }

    fn std_mean(&self) -> Tensor<B, 1> {
        return self.logstd.weight.val().mean();
    }

    fn reset_logstd(&mut self, logstd: f32) {
        let logstd = self.logstd.clone();
        self.logstd.weight = logstd.weight.map(|x| x.mul_scalar(0.95f32).require_grad());
        println!(
            "self.logstd.weight={:?}",
            self.logstd.weight.val().to_data().to_vec::<f32>()
        );
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
