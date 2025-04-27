use std::fmt::Display;

use crate::burn_utils::distribution::normal::Normal;
use burn::module::{AutodiffModule, Module};
use burn::optim::Optimizer;
use burn::prelude::*;
use burn::tensor::backend::{AutodiffBackend, Backend};

use super::config::TrainConfig;
use super::memory::Memory;
use super::rl_utils::UpdateInfo;

// for predicting next state
pub trait ModelBasedNet<B: Backend>: Module<B> {
    fn forward(&self, obs: Tensor<B, 3>, action: Tensor<B, 3>) -> Tensor<B, 3>;
    fn loss(
        &mut self,
        obs: Tensor<B, 3>,
        action: Tensor<B, 3>,
        next_obs: Tensor<B, 3>,
        reward: Tensor<B, 2>,
    ) -> Tensor<B, 2>;
}

// for predicting action normal distribution
pub trait ActorModel<B: Backend>: Module<B> {
    fn forward(
        &self,
        input: Tensor<B, 3>,
        traj_length: Tensor<B, 1, Int>,
        mask: Tensor<B, 3>,
    ) -> Normal<B>;
    fn std_mean(&self) -> Tensor<B, 1>;
}

// for predicting values of states
pub trait BaselineModel<B: Backend>: Module<B> {
    fn forward(
        &self,
        input: Tensor<B, 3>,
        traj_length: Tensor<B, 1, Int>,
        mask: Tensor<B, 3>,
    ) -> Tensor<B, 2>;
}

pub trait RlTrainAlgorithm<
    B: AutodiffBackend,
    AM: ActorModel<B> + AutodiffModule<B> + Display,
    BM: BaselineModel<B> + AutodiffModule<B> + Display,
>
{
    fn new() -> Self;
    fn train(
        &mut self,
        actor_net: AM,
        baseline_net: BM,
        memory: &Memory<B>,
        actor_optimizer: &mut (impl Optimizer<AM, B> + Sized),
        baseline_optimizer: &mut (impl Optimizer<BM, B> + Sized),
        config: &TrainConfig,
        device: &B::Device,
    ) -> Option<(AM, BM, UpdateInfo)>;
}
