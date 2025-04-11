use std::fmt::Display;

use crate::burn_utils::distribution::normal::Normal;
use burn::module::{AutodiffModule, Module};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::Tensor;

use super::config::TrainConfig;
use super::memory::Memory;
use super::rl_utils::UpdateInfo;
use burn::optim::Optimizer;

pub trait ActorModel<B: Backend>: Module<B> {
    fn forward(&self, input: Tensor<B, 2>) -> Normal<B>;
    fn std_mean(&self) -> Tensor<B, 1>;
    fn reset_logstd(&mut self, logstd: f32);
}

pub trait BaselineModel<B: Backend>: Module<B> {
    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 1>;
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
