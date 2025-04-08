use crate::burn_utils::distribution::normal::Normal;
use crate::rl_algorithm::model::Model;
use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

pub struct ActorOutput<B: Backend> {
    pub actions_normal: Normal<B>,
}

impl<B: Backend> ActorOutput<B> {
    pub fn new(actions_normal: Normal<B>) -> Self {
        Self { actions_normal }
    }
}

pub trait ActorModel<B: Backend>: Module<B> {
    fn forward(&self, input: Tensor<B, 2>) -> Normal<B>;
    fn std_mean(&self) -> Tensor<B, 1>;
}

pub trait BaselineModel<B: Backend>: Module<B> {
    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 1>;
}

pub trait ActorCriticModel<B: Backend>: Module<B> {
    fn actor_forward(&self, input: Tensor<B, 2>) -> Normal<B>;
    fn critic_forward(&self, input: Tensor<B, 2>) -> Tensor<B, 1>;
}
