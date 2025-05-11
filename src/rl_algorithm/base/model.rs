use std::collections::HashMap;
use std::fmt::Display;

use crate::burn_utils::distribution::Distribution;
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
pub trait ActorModel<B: AutodiffBackend>: Module<B> {
    fn autodiff_forward(
        &self,
        input: Tensor<B, 3>,
        traj_length: Tensor<B, 1, Int>,
        seq_mask: Tensor<B, 2>,
    ) -> Distribution<B>;

    fn eval_forward(
        &self,
        input: Tensor<B::InnerBackend, 3>,
        traj_length: Tensor<B::InnerBackend, 1, Int>,
        seq_mask: Tensor<B::InnerBackend, 2>,
    ) -> Distribution<B::InnerBackend>;

    fn std_mean(&self) -> Tensor<B, 1>;
}

// for predicting values of states
pub trait BaselineModel<B: AutodiffBackend>: Module<B> {
    fn autodiff_forward(
        &self,
        input: Tensor<B, 3>,
        traj_length: Tensor<B, 1, Int>,
        seq_mask: Tensor<B, 2>,
    ) -> Tensor<B, 2>;

    fn eval_forward(
        &self,
        input: Tensor<B::InnerBackend, 3>,
        traj_length: Tensor<B::InnerBackend, 1, Int>,
        seq_mask: Tensor<B::InnerBackend, 2>,
    ) -> Tensor<B::InnerBackend, 2>;
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
        logger: &mut HashMap<String, f32>,
        memory: &Memory<B>,
        actor_optimizer: &mut (impl Optimizer<AM, B> + Sized),
        baseline_optimizer: &mut (impl Optimizer<BM, B> + Sized),
        config: &TrainConfig,
        device: &B::Device,
    ) -> Option<(AM, BM, UpdateInfo)>;
}
