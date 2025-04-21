use std::marker::PhantomData;

use burn::prelude::*;
use burn::{
    module::AutodiffModule,
    nn::loss::{MseLoss, Reduction},
    optim::Optimizer,
    prelude::Backend,
    tensor::{backend::AutodiffBackend, cast::ToElement, Tensor},
};

use crate::rl_algorithm::base::{config::TrainConfig, model::ModelBasedNet, rl_utils};

struct ModelBasedAgent<B: Backend, NET: ModelBasedNet<B>> {
    b: PhantomData<B>,
    am: PhantomData<NET>,
}

impl<B: AutodiffBackend, NET: ModelBasedNet<B> + AutodiffModule<B>> ModelBasedAgent<B, NET> {
    pub fn new() -> Self {
        ModelBasedAgent {
            b: PhantomData,
            am: PhantomData,
        }
    }
    pub fn train(
        &self,
        model: NET,
        obs: Tensor<B, 2>,
        action: Tensor<B, 2>,
        next_obs: Tensor<B, 2>,
        model_optimizer: &mut (impl Optimizer<NET, B> + Sized),
        config: &TrainConfig,
    ) -> (NET, f32) {
        let mb_config = &config.mb_train_config;

        let next_action_pred = model.forward(obs, action);

        let loss = MseLoss.forward(next_action_pred.clone(), next_obs, Reduction::Mean);

        return (
            rl_utils::update_parameters(
                loss.clone(),
                model,
                model_optimizer,
                mb_config.learning_rate.into(),
            ),
            loss.into_scalar().to_f32(),
        );
    }
}
