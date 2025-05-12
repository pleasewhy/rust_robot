use crate::burn_utils::distribution::normal::Normal;
use crate::burn_utils::distribution::Distribution;
use crate::burn_utils::{build_mlp, Sequence};
use crate::rl_algorithm::base::model::ActorModel;
use crate::rl_algorithm::base::rl_utils::{ndarray2tensor1, tensor2ndarray1, tensor2ndarray2};
use burn::module::AutodiffModule;
use burn::nn::Tanh;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::cast::ToElement;
use burn::{nn::Linear, nn::LinearConfig, prelude::*};
use ndarray::{Array1, AssignElem};

#[derive(Module, Debug)]
pub struct NormalMLPPolicy<B: Backend> {
    mean_net: Sequence<B>,
    logstd_linear: Linear<B>,
    one: Tensor<B, 1>,
    tanh: Tanh,
}

impl<B: Backend> NormalMLPPolicy<B> {
    fn forward(
        &self,
        input: Tensor<B, 3>,
        traj_length: Tensor<B, 1, Int>,
        seq_mask: Tensor<B, 2>,
    ) -> Distribution<B> {
        let batch_size = input.shape().dims[0];
        let seq_length = input.shape().dims[1];

        let device = &self.mean_net.devices()[0];

        let mean = self.mean_net.forward(input);
        let mean = self.tanh.forward(mean);

        let max_seq_len = mean.shape().dims[1]; // (B, T, ac_dim)
        let ac_dim = mean.shape().dims[2]; // (B, T, ac_dim)

        let logstd = self.logstd_linear.forward(self.one.clone());
        let action_mask = seq_mask
            .clone()
            .repeat_dim(1, ac_dim)
            .reshape([batch_size, ac_dim, max_seq_len])
            .swap_dims(1, 2);
        if mean.is_nan().any().into_scalar() {
            println!("mean has nan");
            std::process::exit(-1);
        }
        return Distribution::Normal(Normal::new(
            mean,
            logstd.exp().powi_scalar(2),
            Some(action_mask),
        ));
    }
}
impl<B: AutodiffBackend> ActorModel<B> for NormalMLPPolicy<B> {
    fn autodiff_forward(
        &self,
        input: Tensor<B, 3>,
        traj_length: Tensor<B, 1, Int>,
        seq_mask: Tensor<B, 2>,
    ) -> Distribution<B> {
        return self.forward(input, traj_length, seq_mask);
    }
    fn std_mean(&self) -> Tensor<B, 1> {
        let logstd = self.logstd_linear.forward(self.one.clone());
        return logstd.mean();
    }

    fn eval_forward(
        &self,
        input: Tensor<B::InnerBackend, 3>,
        traj_length: Tensor<B::InnerBackend, 1, Int>,
        seq_mask: Tensor<B::InnerBackend, 2>,
    ) -> Distribution<B::InnerBackend> {
        return self.valid().forward(input, traj_length, seq_mask);
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
pub struct NormalMLPPolicyConfig {
    action_dim: usize,
    observation_dim: usize,
    n_layers: usize,
    layer_size: usize,
}

impl NormalMLPPolicyConfig {
    /// Returns the initialized Mlp.
    pub fn init<B: Backend>(&self, device: &B::Device) -> NormalMLPPolicy<B> {
        let mean_net: Sequence<B> = build_mlp(
            self.observation_dim,
            self.action_dim,
            self.n_layers,
            self.layer_size,
            device,
        );
        let logstd_linear = LinearConfig::new(1, self.action_dim)
            .with_bias(false)
            .with_initializer(nn::Initializer::Zeros)
            .init::<B>(&device);
        return NormalMLPPolicy {
            mean_net,
            logstd_linear,
            one: Tensor::ones([1], device),
            tanh: Tanh::new(),
        };
    }
}
