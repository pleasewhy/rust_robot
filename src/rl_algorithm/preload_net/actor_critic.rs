use super::super::ppo::model::BaselineModel;
use crate::burn_utils::{build_mlp, build_mlp_by_dims, distribution::Normal, Sequence};
use burn::{
    nn::{Linear, LinearConfig},
    prelude::*,
};

#[derive(Module, Debug)]
pub struct MlpActorCritic<B: Backend> {
    critic: Sequence<B>,
    mean_net: Sequence<B>,
    logstd: Linear<B>,
}

impl<B: Backend> MlpActorCritic<B> {
    fn actor_forward(&self, input: Tensor<B, 2>) -> Normal<B> {
        let mean: Tensor<B, 2> = self.mean_net.forward::<2>(input);
        let logstd = self.logstd.weight.val().flatten(0, 1);
        return Normal::new(mean, logstd.exp().powi_scalar(2));
    }
    fn critic_forward(&self, input: Tensor<B, 2>) -> Tensor<B, 1> {
        self.critic.forward::<2>(input).flatten(0, 1)
    }
}

#[derive(Config, Debug)]
pub struct MlpActorCriticConfig {
    obs_dim: usize,
    action_dim: usize,
    actor_layer_dims: Vec<usize>,
    critic_layer_dims: Vec<usize>,
}

impl MlpActorCriticConfig {
    /// Returns the initialized Mlp.
    pub fn init<B: Backend>(&self, device: &B::Device) -> MlpActorCritic<B> {
        let critic = build_mlp_by_dims(self.obs_dim, 1, &self.critic_layer_dims, device);
        let mean_net = build_mlp_by_dims(
            self.obs_dim,
            self.action_dim,
            &self.actor_layer_dims,
            device,
        );
        let logstd = LinearConfig::new(1, self.action_dim)
            .with_bias(false)
            .with_initializer(nn::Initializer::Zeros)
            .init::<B>(&device);
        return MlpActorCritic {
            critic,
            mean_net,
            logstd,
        };
    }
}
