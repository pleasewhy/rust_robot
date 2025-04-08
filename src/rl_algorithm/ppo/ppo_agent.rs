use crate::burn_utils;
use crate::rl_algorithm::memory::Memory;
use crate::rl_algorithm::utils::UpdateInfo;
use crate::rl_env::nd_vec::{
    booltensor2vec1, tensor2vec1, tensor2vec2, vec2tensor1, vec2tensor2, NdVec2,
};

use super::super::utils;
use super::config::PPOTrainingConfig;
use super::model::{ActorModel, BaselineModel};
use burn::module::AutodiffModule;
use burn::nn::loss::{MseLoss, Reduction};
use burn::optim::Optimizer;
use burn::prelude::Backend;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Distribution, Float, Tensor};
use std::fmt::Display;
use std::marker::PhantomData;

pub struct PPO<B: Backend, AM: ActorModel<B>, BM: BaselineModel<B>> {
    backend: PhantomData<B>,
    actor: PhantomData<AM>,
    baseline_net: PhantomData<BM>,
    // pub ppo_train_config: PPOTrainingConfig,
    // pub ppo_train_config: PPOTrainingConfig,
}

impl<B: Backend, AM: ActorModel<B>, BM: BaselineModel<B>> PPO<B, AM, BM> {
    pub fn new() -> Self {
        Self {
            // state: PhantomData,
            // action: PhantomData,
            backend: PhantomData,
            actor: PhantomData,
            baseline_net: PhantomData,
            // ppo_train_config,
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
    > PPO<B, AM, BM>
{
    pub fn update_actor(
        actor_net: AM,
        obs: Tensor<B, 2>,
        action: Tensor<B, 2>,
        advantages: Tensor<B, 1>,
        old_logprobs: Tensor<B, 1>,
        actor_optimizer: &mut (impl Optimizer<AM, B> + Sized),
        config: &PPOTrainingConfig,
    ) -> (AM, f32) {
        let normal = actor_net.forward(obs);
        let logprobs = normal.independent_log_prob(action);
        let ratio = (logprobs - old_logprobs).exp();
        let clipped_ratio = ratio
            .clone()
            .clamp(1.0 - config.epsilon_clip, 1.0 + config.epsilon_clip);
        let now_advantage = ratio * advantages.clone();
        let clip_advantage = clipped_ratio * advantages.clone();
        let actor_loss = -now_advantage.min_pair(clip_advantage).mean()
            - normal.entropy().mean() * config.entropy_coef;
        return (
            utils::update_parameters(
                actor_loss.clone(),
                actor_net,
                actor_optimizer,
                config.learning_rate.into(),
            ),
            actor_loss.into_data().as_slice().unwrap()[0],
        );
    }

    pub fn update_baseline(
        baseline_net: BM,
        obs: Tensor<B, 2>,
        returns: Tensor<B, 1>,
        baseline_optimizer: &mut (impl Optimizer<BM, B> + Sized),
        config: &PPOTrainingConfig,
    ) -> (BM, f32) {
        let pred = baseline_net.forward(obs);
        let baseline_loss = MseLoss.forward(returns.clone(), pred, Reduction::Mean);
        return (
            utils::update_parameters(
                baseline_loss.clone(),
                baseline_net,
                baseline_optimizer,
                config.learning_rate.into(),
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
        config: &PPOTrainingConfig,
        device: &B::Device,
    ) -> Option<(AM, BM, UpdateInfo)> {
        let obs = memory.obs();
        let action = memory.action();
        let rewards = memory.reward();
        let not_dones = memory.done().clone().bool_not();

        let old_values = baseline_net.forward(obs.clone());
        let old_gae_output = get_gae::<B>(
            &tensor2vec1(&old_values),
            &tensor2vec1(rewards),
            &booltensor2vec1(&not_dones),
            config.gae_gamma,
            config.reward_lambda,
            &device,
        )?;

        let old_logprobs = actor_net
            .forward(obs.clone())
            .independent_log_prob(action.clone());
        let advantages = old_gae_output.advantages;
        let expected_values = old_gae_output.expected_returns;
        let mut update_info = UpdateInfo::new();

        update_info.mean_q_val = expected_values
            .clone()
            .mean()
            .into_data()
            .as_slice()
            .unwrap()[0];
        let mut actor_loss = 0.0;
        let mut baseline_loss = 0.0;
        let mini_batch_size = config.mini_batch_size.min(memory.len());
        let mini_batch_iter = memory.mini_batch_iter(
            config.update_freq,
            memory.len() / config.mini_batch_size + 1,
            old_logprobs,
            expected_values,
            advantages,
        );
        for (mini_obs, mini_action, mini_old_logprobs, mini_expected_values, mini_advantages) in
            mini_batch_iter
        {
            (actor_net, actor_loss) = Self::update_actor(
                actor_net,
                mini_obs.clone(),
                mini_action.clone(),
                mini_advantages.clone(),
                mini_old_logprobs,
                actor_optimizer,
                config,
            );
            (baseline_net, baseline_loss) = Self::update_baseline(
                baseline_net,
                mini_obs.clone(),
                mini_expected_values,
                baseline_optimizer,
                config,
            );
            update_info.actor_loss += actor_loss;
            update_info.critic_loss += baseline_loss;
        }
        return Some((actor_net, baseline_net, update_info));
    }
}

pub(crate) struct GAEOutput<B: Backend> {
    expected_returns: Tensor<B, 1>,
    advantages: Tensor<B, 1>,
}

pub(crate) fn get_gae<B: Backend>(
    values: &Vec<f32>,
    rewards: &Vec<f32>,
    not_dones: &Vec<bool>,
    reward_gamma: f32,
    gae_lambda: f32,
    device: &B::Device,
) -> Option<GAEOutput<B>> {
    let mut returns = vec![0.0 as f32; rewards.len()];
    let mut advantages = returns.clone();

    let mut running_return: f32 = 0.0;
    let mut running_advantage: f32 = 0.0;

    for i in (0..rewards.len()).rev() {
        let reward = rewards.get(i)?;
        let not_done = *not_dones.get(i)? as i8 as f32;

        running_return = reward + reward_gamma * running_return * not_done;
        running_advantage = reward - values.get(i)?
            + reward_gamma
                * not_done
                * (values.get(i + 1).unwrap_or(&0.0) + gae_lambda * running_advantage);

        returns[i] = running_return;
        advantages[i] = running_advantage;
    }

    return Some(GAEOutput {
        expected_returns: vec2tensor1(returns, device),
        advantages: utils::normalize(vec2tensor1(advantages, device)),
    });
}
