use crate::burn_utils::distribution::normal::Normal;
use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

pub trait ActorModel<B: Backend>: Module<B> {
    fn forward(&self, input: Tensor<B, 2>) -> Normal<B>;
    fn std_mean(&self) -> Tensor<B, 1>;
}

pub trait BaselineModel<B: Backend>: Module<B> {
    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 1>;
}

