use burn::prelude::*;

pub mod categorical;
pub mod normal;
// pub use normal::*;

#[derive(Debug)]
pub enum Distribution<B: Backend> {
    Normal(normal::Normal<B>),
    Categorical(categorical::Categorical<B>),
}

impl<B: Backend> Distribution<B> {
    pub fn sample(&self) -> Tensor<B, 3> {
        match self {
            Distribution::Normal(normal) => normal.sample(),
            Distribution::Categorical(categorical) => categorical.sample(),
        }
    }

    pub fn log_prob(&self, value: Tensor<B, 3>) -> Tensor<B, 3> {
        match self {
            Distribution::Normal(normal) => normal.log_prob(value),
            Distribution::Categorical(categorical) => categorical.log_prob(value),
        }
    }
    pub fn independent_log_prob(&self, value: Tensor<B, 3>) -> Tensor<B, 2> {
        match self {
            Distribution::Normal(normal) => normal.independent_log_prob(value),
            Distribution::Categorical(categorical) => categorical.independent_log_prob(value),
        }
    }
    pub fn entropy(&self) -> Tensor<B, 1> {
        match self {
            Distribution::Normal(normal) => normal.entropy(),
            Distribution::Categorical(categorical) => categorical.entropy(),
        }
    }
    pub fn mean(&self) -> Tensor<B, 3> {
        match self {
            Distribution::Normal(normal) => normal.loc.clone(),
            Distribution::Categorical(categorical) => categorical.mask.as_ref().unwrap().clone(),
        }
    }
}
