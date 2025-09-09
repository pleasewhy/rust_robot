use crate::burn_utils::distribution::categorical::Categorical;
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
pub struct CategoricalMLPPolicy<B: Backend> {
    mean_net: Sequence<B>,
    action_net: Linear<B>,
    logstd_linear: Linear<B>,
    tanh: Tanh,
    action_dim: usize,
    interval: f32,
    action_logit_num: usize,
}

impl<B: Backend> CategoricalMLPPolicy<B> {
    fn forward(
        &self,
        input: Tensor<B, 3>,
        traj_length: Tensor<B, 1, Int>,
        seq_mask: Tensor<B, 2>,
    ) -> Distribution<B> {
        let batch_size = input.shape().dims[0];
        let seq_length = input.shape().dims[1];

        let device = &self.mean_net.devices()[0];

        let mean = self.mean_net.forward(input); // (B, T, hiddim)

        let acion_prob = self.action_net.forward(mean).reshape([
            batch_size,
            seq_length,
            self.action_dim,
            self.action_logit_num,
        ]); // (B, T, action_dim * action_logit_num)
        let acion_prob = burn::tensor::activation::sigmoid(acion_prob);
        let max_seq_len = acion_prob.shape().dims[1]; // (B, T, ac_dim)
        let logstd = self.logstd_linear.weight.val().flatten(0, 1);
        let action_mask = seq_mask
            .clone()
            .repeat_dim(1, self.action_dim)
            .reshape([batch_size, self.action_dim, max_seq_len])
            .swap_dims(1, 2);

        if acion_prob.clone().is_nan().any().into_scalar().to_bool() {
            println!("has nan");
            std::process::exit(-1);
        }
        // println!("mean={}", mean);
        // println!("logstd={}", logstd);
        // println!("action_mask={}", action_mask);
        return Distribution::Categorical(Categorical::new(
            acion_prob,
            logstd.exp().powi_scalar(2),
            Some(action_mask),
            -1.0,
            self.interval,
        ));
    }
}

impl<B: AutodiffBackend> ActorModel<B> for CategoricalMLPPolicy<B> {
    fn autodiff_forward(
        &self,
        input: Tensor<B, 3>,
        traj_length: Tensor<B, 1, Int>,
        seq_mask: Tensor<B, 2>,
    ) -> Distribution<B> {
        return self.forward(input, traj_length, seq_mask);
    }
    fn std_mean(&self) -> Tensor<B, 1> {
        let logstd = self.logstd_linear.weight.val().flatten::<1>(0, 1);
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

#[derive(Config, Debug)]
pub struct CategoricalMLPPolicyConfig {
    action_dim: usize,
    observation_dim: usize,
    n_layers: usize,
    layer_size: usize,
    #[config(default = 100)]
    action_logit_num: usize,
}

impl CategoricalMLPPolicyConfig {
    /// Returns the initialized Mlp.
    pub fn init<B: Backend>(&self, device: &B::Device) -> CategoricalMLPPolicy<B> {
        let mean_net: Sequence<B> = build_mlp(
            self.observation_dim,
            self.layer_size,
            self.n_layers,
            self.layer_size,
            device,
        );
        let action_net =
            LinearConfig::new(self.layer_size, self.action_dim * self.action_logit_num)
                .init::<B>(&device);
        let logstd_linear = LinearConfig::new(1, self.action_dim)
            .with_bias(false)
            .with_initializer(nn::Initializer::Zeros)
            .init::<B>(&device);
        return CategoricalMLPPolicy {
            mean_net,
            action_net,
            logstd_linear,
            tanh: Tanh::new(),
            action_dim: self.action_dim,
            interval: 2.0 / self.action_logit_num as f32,
            action_logit_num: self.action_logit_num,
        };
    }
}
