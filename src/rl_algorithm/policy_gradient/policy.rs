
use crate::burn_utils::distribution::Normal;
use crate::burn_utils::{build_mlp, Sequence};
use burn::nn::Tanh;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::Adam;
use burn::optim::AdamConfig;
use burn::optim::GradientsParams;
use burn::optim::Optimizer;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::cast::ToElement;
use burn::{nn::Linear, nn::LinearConfig, prelude::*};

#[derive(Module, Debug)]
pub struct MLPPolicy<B: Backend> {
    mean_net: Sequence<B>,
    logstd: Linear<B>,
    tanh: Tanh,
}

impl<B: Backend> MLPPolicy<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Normal<B> {
        let mean: Tensor<B, 2> = self.mean_net.forward::<2>(input);
        // let mean = self.tanh.forward(mean);
        let logstd = self.logstd.weight.val().flatten(0, 1);

        return Normal::new(mean, logstd.exp().powi_scalar(2));
    }
}

#[derive(Config, Debug)]
pub struct MLPPolicyConfig {
    action_dim: usize,
    observation_dim: usize,
    discrete: bool,
    n_layers: usize,
    layer_size: usize,
}

impl MLPPolicyConfig {
    /// Returns the initialized Mlp.
    pub fn init<B: Backend>(&self, device: &B::Device) -> MLPPolicy<B> {
        let mean_net = build_mlp(
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

pub struct MLPPolicyTrainer<B: AutodiffBackend> {
    pub mlp_policy: MLPPolicy<B>,
    pub adam: OptimizerAdaptor<Adam, MLPPolicy<B>, B>,
    pub lr: f64,
}

impl<B: AutodiffBackend> MLPPolicyTrainer<B> {
    pub fn new(
        device: &B::Device,
        mlp_policy_config: MLPPolicyConfig,
        adam_config: AdamConfig,
        learn_rate: f64,
    ) -> Self {
        return Self {
            mlp_policy: mlp_policy_config.init(device),
            adam: adam_config.init(),
            lr: learn_rate,
        };
    }
    pub fn train_update(
        &mut self,
        obs: Tensor<B, 2>,
        actions: Tensor<B, 2>,
        advantages: Tensor<B, 1>,
    ) -> f32 {
        let log_prob = self.mlp_policy.forward(obs).independent_log_prob(actions);
        let loss = -(log_prob.clone() * advantages.clone()).mean();
        let grads = loss.backward();

        let grads = GradientsParams::from_grads(grads, &self.mlp_policy);

        self.mlp_policy = self.adam.step(self.lr, self.mlp_policy.clone(), grads);
        return loss.into_scalar().to_f32();
    }
}
