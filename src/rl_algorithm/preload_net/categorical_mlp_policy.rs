use crate::burn_utils::distribution::categorical::Categorical;
use crate::burn_utils::distribution::Distribution;
use crate::burn_utils::{build_mlp, Sequence};
use crate::rl_algorithm::base::model::ActorModel;
use crate::rl_algorithm::base::rl_utils::{ndarray2tensor1, tensor2ndarray1, tensor2ndarray2};
use burn::module::AutodiffModule;
use burn::nn::Tanh;
use burn::tensor::backend::AutodiffBackend;
use burn::{nn::Linear, nn::LinearConfig, prelude::*};
use ndarray::{Array1, AssignElem};

#[derive(Module, Debug)]
pub struct CategoricalMLPPolicy<B: Backend> {
    mean_net: Sequence<B>,
    action_net: Linear<B>,
    logstd: Tensor<B, 1>,
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
        mask: Tensor<B, 3>,
    ) -> Distribution<B> {
        let batch_size = input.shape().dims[0];
        let seq_length = input.shape().dims[1];

        let device = &self.mean_net.devices()[0];
        let flatten_indices =
            super::utils::generate_flatten_idx(tensor2ndarray1(&traj_length), seq_length, device);

        // (batch_size * seq_len, ob_dim)
        let input = input.flatten::<2>(0, 1);

        let input = input.select(0, flatten_indices.clone());

        let mean = self.mean_net.forward::<2>(input); // (batch_size * seq_len, hidden_size)
        let mean = self.action_net.forward::<2>(mean); // (, self.action_dim * self.action_logit_num)

        // (batch_size * seq_len, self.action_dim, self.action_logit_num)
        let mean =
            mean.clone()
                .reshape([mean.shape().dims[0], self.action_dim, self.action_logit_num]);
        let mean = burn::tensor::activation::softmax(mean, 2);

        // let mean = self.tanh.forward(mean);
        let ac_dim = mean.shape().dims[1];
        let zero_mean = Tensor::zeros(
            [batch_size * seq_length, ac_dim, self.action_logit_num],
            device,
        );
        // println!("mean.shape={:?}", mean.shape());
        // println!("zero_mean.shape={:?}", zero_mean.shape());
        // println!("flatten_indices.shape={:?}", flatten_indices.shape());

        let mean = zero_mean
            .select_assign(0, flatten_indices, mean.clone())
            .reshape([batch_size, seq_length, ac_dim, self.action_logit_num]);

        return Distribution::Categorical(Categorical::new(
            mean,
            self.logstd.clone(),
            Some(mask),
            -1.0,
            self.interval,
        ));
    }
}

impl<B: AutodiffBackend> ActorModel<B> for CategoricalMLPPolicy<B> {
    fn std_mean(&self) -> Tensor<B, 1> {
        return self.logstd.clone().mean();
    }

    fn autodiff_forward(
        &self,
        input: Tensor<B, 3>,
        traj_length: Tensor<B, 1, Int>,
        mask: Tensor<B, 3>,
    ) -> Distribution<B> {
        return self.forward(input, traj_length, mask);
    }

    fn eval_forward(
        &self,
        input: Tensor<<B as AutodiffBackend>::InnerBackend, 3>,
        traj_length: Tensor<<B as AutodiffBackend>::InnerBackend, 1, Int>,
        mask: Tensor<<B as AutodiffBackend>::InnerBackend, 3>,
    ) -> Distribution<<B as AutodiffBackend>::InnerBackend> {
        return self.valid().forward(input, traj_length, mask);
    }
}

#[derive(Config, Debug)]
pub struct CategoricalMLPPolicyConfig {
    action_dim: usize,
    observation_dim: usize,
    n_layers: usize,
    layer_size: usize,
    #[config(default = 10)]
    action_logit_num: usize,
}

impl CategoricalMLPPolicyConfig {
    /// Returns the initialized Mlp.
    pub fn init<B: Backend>(&self, device: &B::Device) -> CategoricalMLPPolicy<B> {
        let mean_net: Sequence<B> = build_mlp(
            self.observation_dim,
            self.observation_dim,
            self.n_layers,
            self.layer_size,
            device,
        );
        let action_net = LinearConfig::new(
            self.observation_dim,
            self.action_dim * self.action_logit_num,
        )
        .init::<B>(&device);
        let logstd = Tensor::<B, 1>::ones([self.action_dim], device).require_grad();
        return CategoricalMLPPolicy {
            mean_net,
            action_net,
            logstd,
            tanh: Tanh::new(),
            action_dim: self.action_dim,
            interval: 2.0 / self.action_logit_num as f32,
            action_logit_num: self.action_logit_num,
        };
    }
}
