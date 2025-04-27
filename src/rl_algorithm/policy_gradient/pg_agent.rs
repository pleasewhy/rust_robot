use crate::rl_algorithm::base::config::TrainConfig;
use crate::rl_algorithm::base::memory::Memory;
use crate::rl_algorithm::base::rl_utils::UpdateInfo;
use crate::rl_algorithm::base::rl_utils::{self, tensor2ndarray2};

use crate::rl_algorithm::base::model::{ActorModel, BaselineModel, RlTrainAlgorithm};
use burn::module::AutodiffModule;
use burn::nn::loss::{MseLoss, Reduction};
use burn::optim::Optimizer;
use burn::prelude::Backend;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::cast::ToElement;
use burn::tensor::{Bool, Tensor};
use std::fmt::Display;
use std::marker::PhantomData;

pub struct PolicyGradient<B: Backend, AM: ActorModel<B>, BM: BaselineModel<B>> {
    backend: PhantomData<B>,
    actor: PhantomData<AM>,
    baseline_net: PhantomData<BM>,
}

impl<
        B: AutodiffBackend,
        AM: ActorModel<B> + AutodiffModule<B> + Display,
        BM: BaselineModel<B> + AutodiffModule<B> + Display,
    > PolicyGradient<B, AM, BM>
{
    pub fn update_actor(
        actor_net: AM,
        obs: Tensor<B, 3>,
        traj_length: Tensor<B, 1, Int>,
        valid_timestep: Tensor<B, 1>,
        actor_net_mask: Tensor<B, 3>,
        action: Tensor<B, 3>,
        advantages: Tensor<B, 2>,
        actor_optimizer: &mut (impl Optimizer<AM, B> + Sized),
        config: &TrainConfig,
    ) -> (AM, f32) {
        let pg_config = &config.pg_train_config;
        let log_prob = actor_net
            .forward(obs, traj_length, actor_net_mask)
            .independent_log_prob(action);
        let actor_loss = -(log_prob.clone() * advantages.clone()).sum() / valid_timestep;

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
        obs: Tensor<B, 3>,
        traj_length: Tensor<B, 1, Int>,
        baseline_net_mask: Tensor<B, 3>,
        returns: Tensor<B, 2>,
        baseline_optimizer: &mut (impl Optimizer<BM, B> + Sized),
        config: &TrainConfig,
    ) -> (BM, f32) {
        let pg_config = &config.pg_train_config;

        let pred = baseline_net.forward(obs, traj_length, baseline_net_mask);
        let baseline_loss = MseLoss.forward(returns.clone(), pred, Reduction::Sum);
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
}

impl<
        B: AutodiffBackend,
        AM: ActorModel<B> + AutodiffModule<B> + Display,
        BM: BaselineModel<B> + AutodiffModule<B> + Display,
    > RlTrainAlgorithm<B, AM, BM> for PolicyGradient<B, AM, BM>
{
    fn new() -> Self {
        Self {
            backend: PhantomData,
            actor: PhantomData,
            baseline_net: PhantomData,
        }
    }
    fn train(
        &mut self,
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

        let traj_length = memory.traj_length();
        let valid_timestep = traj_length.clone().sum().float();
        let actor_net_mask = memory.actor_net_mask();
        let baseline_net_mask = memory.baseline_net_mask();
        let seq_mask = memory.seq_mask();

        let not_dones = memory.done().clone().bool_not();

        let values =
            baseline_net.forward(obs.clone(), traj_length.clone(), baseline_net_mask.clone());
        let gae_output = rl_utils::get_gae::<B>(
            tensor2ndarray2(&values).view(),
            tensor2ndarray2(rewards).view(),
            tensor2ndarray2::<B, bool, Bool>(&not_dones).view(),
            seq_mask.clone(),
            pg_config.gae_gamma,
            pg_config.reward_lambda,
            &device,
        )?;

        let advantages = gae_output.advantages;
        let expected_values = gae_output.expected_returns;
        let mut update_info = UpdateInfo::new();

        update_info.mean_q_val = expected_values.clone().mean().into_scalar().to_f32();

        let actor_loss: f32;
        let mut baseline_loss: f32;

        (actor_net, actor_loss) = Self::update_actor(
            actor_net,
            obs.clone(),
            traj_length.clone(),
            valid_timestep.clone(),
            actor_net_mask.clone(),
            action.clone(),
            advantages.clone(),
            actor_optimizer,
            config,
        );
        for _ in 0..pg_config.baseline_update_freq {
            (baseline_net, baseline_loss) = Self::update_baseline(
                baseline_net,
                obs.clone(),
                traj_length.clone(),
                baseline_net_mask.clone(),
                expected_values.clone(),
                baseline_optimizer,
                config,
            );
            update_info.critic_loss += baseline_loss;
        }

        update_info.actor_loss += actor_loss;

        return Some((actor_net, baseline_net, update_info));
    }
}
