use crate::rl_algorithm::normal;
use crate::rl_algorithm::utils;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::Adam;
use burn::optim::AdamConfig;
use burn::optim::GradientsParams;
use burn::optim::Optimizer;
use burn::tensor::backend::AutodiffBackend;
use burn::{nn::Linear, nn::LinearConfig, prelude::*};

#[derive(Module, Debug)]
pub struct MLPPolicy<B: Backend> {
    mean_net: utils::Mlp<B>,
    logstd: Linear<B>,
    place_holder: Tensor<B, 2>,
}

impl<B: Backend> MLPPolicy<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> normal::Normal<B> {
        let mean = self.mean_net.forward::<2>(input);
        let logstd = self.logstd.forward(self.place_holder.clone().flatten(0, 1));

        return normal::Normal::new(mean, logstd.exp());
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
        let mean_net = utils::MlpConfig::new(
            self.observation_dim,
            self.action_dim,
            self.n_layers,
            self.layer_size,
        )
        .init::<B>(&device);
        let logstd = LinearConfig::new(1, self.action_dim)
            .with_bias(false)
            .init::<B>(&device);
        let place_holder = Tensor::<B, 2>::from_floats([[1.0]], &device);
        return MLPPolicy {
            mean_net,
            logstd,
            place_holder,
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
    ) {
        let log_prob = self.mlp_policy.forward(obs).independent_log_prob(actions);
        let loss = -(log_prob * advantages).mean();
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.mlp_policy);

        self.mlp_policy = self.adam.step(self.lr, self.mlp_policy.clone(), grads);
    }
}
