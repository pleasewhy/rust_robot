use crate::rl_algorithm::base::config::TrainConfig;
use crate::rl_algorithm::base::memory::Memory;
use crate::rl_algorithm::base::rl_utils::{self, booltensor2vec1, tensor2vec1};
use crate::rl_algorithm::base::rl_utils::UpdateInfo;

use crate::rl_algorithm::base::model::{ActorModel, BaselineModel, RlTrainAlgorithm};
use burn::module::AutodiffModule;
use burn::nn::loss::{MseLoss, Reduction};
use burn::optim::Optimizer;
use burn::prelude::Backend;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Tensor;
use std::collections::HashMap;
use std::fmt::Display;
use std::marker::PhantomData;

pub struct PPO<B: Backend, AM: ActorModel<B>, BM: BaselineModel<B>> {
    backend: PhantomData<B>,
    actor: PhantomData<AM>,
    baseline_net: PhantomData<BM>,
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
        config: &TrainConfig,
    ) -> (AM, f32) {
        let ppo_config = &config.ppo_train_config;
        let normal = actor_net.forward(obs);
        let logprobs = normal.independent_log_prob(action);
        let ratio = (logprobs - old_logprobs).exp();
        let clipped_ratio = ratio
            .clone()
            .clamp(1.0 - ppo_config.epsilon_clip, 1.0 + ppo_config.epsilon_clip);
        let now_advantage = ratio * advantages.clone();
        let clip_advantage = clipped_ratio * advantages.clone();
        let actor_loss = -now_advantage.min_pair(clip_advantage).mean()
            - normal.entropy().mean() * ppo_config.entropy_coef;
        return (
            rl_utils::update_parameters(
                actor_loss.clone(),
                actor_net,
                actor_optimizer,
                ppo_config.learning_rate.into(),
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
        let ppo_config = &config.ppo_train_config;

        let pred = baseline_net.forward(obs);
        let baseline_loss = MseLoss.forward(returns.clone(), pred, Reduction::Mean);
        return (
            rl_utils::update_parameters(
                baseline_loss.clone(),
                baseline_net,
                baseline_optimizer,
                ppo_config.learning_rate.into(),
            ),
            baseline_loss.into_data().as_slice().unwrap()[0],
        );
    }
}

impl<
        B: AutodiffBackend,
        AM: ActorModel<B> + AutodiffModule<B> + Display,
        BM: BaselineModel<B> + AutodiffModule<B> + Display,
    > RlTrainAlgorithm<B, AM, BM> for PPO<B, AM, BM>
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
        let ppo_config = &config.ppo_train_config;
        let obs = memory.obs();
        let action = memory.action();
        let rewards = memory.reward();
        let not_dones = memory.done().clone().bool_not();

        let old_values = baseline_net.forward(obs.clone());
        let old_gae_output = rl_utils::get_gae::<B>(
            &tensor2vec1(&old_values),
            &tensor2vec1(rewards),
            &booltensor2vec1(&not_dones),
            ppo_config.gae_gamma,
            ppo_config.reward_lambda,
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
        let mut actor_loss: f32;
        let mut baseline_loss: f32;
        let mini_batch_size = ppo_config.mini_batch_size.min(memory.len());

        let extra_mini_batch_map = HashMap::from([
            ("old_logprobs", old_logprobs),
            ("expected_values", expected_values),
            ("advantages", advantages),
        ]);

        let mini_batch_iter = memory.mini_batch_iter(
            ppo_config.update_freq,
            memory.len() / mini_batch_size,
            Some(extra_mini_batch_map),
        );
        for (mini_obs, mini_action, mini_batch_map) in mini_batch_iter {
            let mini_old_logprobs = mini_batch_map["old_logprobs"].clone();
            let mini_expected_values = mini_batch_map["expected_values"].clone();
            let mini_advantages = mini_batch_map["advantages"].clone();

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
