use crate::rl_algorithm::base::config::TrainConfig;
use crate::rl_algorithm::base::memory::Memory;
use crate::rl_algorithm::base::rl_utils;
use crate::rl_algorithm::base::rl_utils::UpdateInfo;
use crate::rl_env::nd_vec::{booltensor2vec1, tensor2vec1, tensor2vec2, vec2tensor2, NdVec2};

use crate::rl_algorithm::base::model::{ActorModel, BaselineModel};
use burn::module::AutodiffModule;
use burn::nn::loss::{MseLoss, Reduction};
use burn::optim::Optimizer;
use burn::prelude::Backend;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::cast::ToElement;
use burn::tensor::Tensor;
use std::fmt::Display;
use std::marker::PhantomData;

pub struct PolicyGradient<B: Backend, AM: ActorModel<B>, BM: BaselineModel<B>> {
    backend: PhantomData<B>,
    actor: PhantomData<AM>,
    baseline_net: PhantomData<BM>,
}

impl<B: Backend, AM: ActorModel<B>, BM: BaselineModel<B>> PolicyGradient<B, AM, BM> {
    pub fn new() -> Self {
        Self {
            backend: PhantomData,
            actor: PhantomData,
            baseline_net: PhantomData,
        }
    }

    pub fn get_action(actor: AM, obs: NdVec2<f64>) -> NdVec2<f64> {
        let obs = vec2tensor2(obs, &actor.devices()[0]);
        let action = actor.forward(obs).sample();
        return tensor2vec2(&action).to_f64();
    }
}

impl<
        B: AutodiffBackend,
        AM: ActorModel<B> + AutodiffModule<B> + Display,
        BM: BaselineModel<B> + AutodiffModule<B> + Display,
    > PolicyGradient<B, AM, BM>
{
    pub fn update_actor(
        actor_net: AM,
        obs: Tensor<B, 2>,
        action: Tensor<B, 2>,
        advantages: Tensor<B, 1>,
        actor_optimizer: &mut (impl Optimizer<AM, B> + Sized),
        config: &TrainConfig,
    ) -> (AM, f32) {
        let pg_config = &config.pg_train_config;
        let log_prob = actor_net.forward(obs).independent_log_prob(action);
        let actor_loss = -(log_prob.clone() * advantages.clone()).mean();

        return (
            rl_utils::update_parameters(
                actor_loss.clone(),
                actor_net,
                actor_optimizer,
                pg_config.learning_rate.into(),
            ),
            actor_loss.into_data().as_slice().unwrap()[0],
        );
    }

    pub fn update_baseline(
        baseline_net: BM,
        obs: Tensor<B, 2>,
        returns: Tensor<B, 1>,
        baseline_optimizer: &mut (impl Optimizer<BM, B> + Sized),
        config: &TrainConfig,
    ) -> (BM, f32) {
        let pg_config = &config.pg_train_config;

        let pred = baseline_net.forward(obs);
        let baseline_loss = MseLoss.forward(returns.clone(), pred, Reduction::Mean);
        return (
            rl_utils::update_parameters(
                baseline_loss.clone(),
                baseline_net,
                baseline_optimizer,
                pg_config.learning_rate.into(),
            ),
            baseline_loss.into_data().as_slice().unwrap()[0],
        );
    }

    pub fn train(
        mut actor_net: AM,
        mut baseline_net: BM,
        memory: &Memory<B>,
        actor_optimizer: &mut (impl Optimizer<AM, B> + Sized),
        baseline_optimizer: &mut (impl Optimizer<BM, B> + Sized),
        config: &TrainConfig,
        device: &B::Device,
    ) -> Option<(AM, BM, UpdateInfo)> {
        let pg_config = &config.pg_train_config;

        let obs = memory.obs();
        let action = memory.action();
        let rewards = memory.reward();
        let not_dones = memory.done().clone().bool_not();

        let values = baseline_net.forward(obs.clone());
        let gae_output = rl_utils::get_gae::<B>(
            &tensor2vec1(&values),
            &tensor2vec1(rewards),
            &booltensor2vec1(&not_dones),
            pg_config.gae_gamma,
            pg_config.reward_lambda,
            &device,
        )?;

        let advantages = gae_output.advantages;
        let expected_values = gae_output.expected_returns;
        let mut update_info = UpdateInfo::new();

        update_info.mean_q_val = expected_values.clone().mean().into_scalar().to_f32();

        let actor_loss: f32;
        let baseline_loss: f32;

        (actor_net, actor_loss) = Self::update_actor(
            actor_net,
            obs.clone(),
            action.clone(),
            advantages.clone(),
            actor_optimizer,
            config,
        );
        (baseline_net, baseline_loss) = Self::update_baseline(
            baseline_net,
            obs.clone(),
            expected_values,
            baseline_optimizer,
            config,
        );
        update_info.actor_loss += actor_loss;
        update_info.critic_loss += baseline_loss;

        return Some((actor_net, baseline_net, update_info));
    }
}
