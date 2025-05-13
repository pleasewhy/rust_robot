use crate::burn_utils;
use crate::rl_algorithm::base::config::TrainConfig;
use crate::rl_algorithm::base::memory::Memory;
use crate::rl_algorithm::base::rl_utils::UpdateInfo;
use crate::rl_algorithm::base::rl_utils::{self, tensor2ndarray2};

use crate::rl_algorithm::base::model::{ActorModel, BaselineModel, RlTrainAlgorithm};
use burn::module::AutodiffModule;
use burn::nn::loss::{MseLoss, Reduction};
use burn::optim::Optimizer;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{cast::ToElement, Bool, Int, Tensor, TensorData};
use log::{info, trace};
use ndarray_rand::rand::seq;
use std::collections::HashMap;
use std::fmt::Display;
use std::marker::PhantomData;
use std::process;

pub struct PPO<B: AutodiffBackend, AM: ActorModel<B>, BM: BaselineModel<B>> {
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
        obs: Tensor<B, 3>,
        traj_length: Tensor<B, 1, Int>,
        num_timestep: Tensor<B, 1>,
        seq_mask: Tensor<B, 2>,
        actor_net_mask: Tensor<B, 3>,
        action: Tensor<B, 3>,
        advantages: Tensor<B, 2>,
        old_logprobs: Tensor<B, 2>,
        actor_optimizer: &mut (impl Optimizer<AM, B> + Sized),
        config: &TrainConfig,
    ) -> (AM, crate::FType) {
        let ppo_config = &config.ppo_train_config;
        let normal = actor_net.autodiff_forward(obs, traj_length.clone(), seq_mask.clone());
        let logprobs: Tensor<B, 2> = normal.independent_log_prob(action);
        trace!("logprobs={}", logprobs);
        trace!("old_logprobs={}", old_logprobs);
        let ratio = (logprobs - old_logprobs).exp();
        trace!("ratio={}", ratio);
        let clipped_ratio = ratio
            .clone()
            .clamp(1.0 - ppo_config.epsilon_clip, 1.0 + ppo_config.epsilon_clip);
        trace!("clipped_ratio={}", clipped_ratio);

        let now_advantage = ratio * advantages.clone();
        let clip_advantage = clipped_ratio * advantages.clone();

        trace!("advantages={}", advantages);
        trace!("now_advantage={}", now_advantage);
        trace!("clip_advantage={}", clip_advantage);

        let advantage_loss = -now_advantage.min_pair(clip_advantage).mul(seq_mask.clone());

        let actor_loss = burn_utils::mean_with_mask(advantage_loss, seq_mask.bool())
            - normal.entropy().mean() * ppo_config.entropy_coef;

        return (
            rl_utils::update_parameters(
                actor_loss.clone(),
                actor_net,
                actor_optimizer,
                ppo_config.learning_rate.into(),
                Some("actor"),
            ),
            actor_loss.into_data().as_slice().unwrap()[0],
        );
    }

    pub fn update_baseline(
        baseline_net: BM,
        obs: Tensor<B, 3>,
        traj_length: Tensor<B, 1, Int>,
        num_timestep: Tensor<B, 1>,
        mask: Tensor<B, 3>,
        seq_mask: Tensor<B, 2>,
        returns: Tensor<B, 2>,
        baseline_optimizer: &mut (impl Optimizer<BM, B> + Sized),
        config: &TrainConfig,
    ) -> (BM, crate::FType) {
        let ppo_config = &config.ppo_train_config;

        let pred: Tensor<B, 2> =
            baseline_net.autodiff_forward(obs.clone(), traj_length, seq_mask.clone());
        // println!("pred={}", pred);
        let baseline_loss =
            burn_utils::avoid_overflow::mse_loss_with_mask(pred, returns, seq_mask.bool());
        return (
            rl_utils::update_parameters(
                baseline_loss.clone(),
                baseline_net,
                baseline_optimizer,
                ppo_config.learning_rate.into(),
                Some("baseline"),
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

        let traj_length = memory.traj_length();
        let valid_timestep = traj_length.clone().sum();
        let valid_timestep = valid_timestep.float();

        let actor_net_mask = memory.actor_net_mask();
        let baseline_net_mask = memory.baseline_net_mask();
        let seq_mask = memory.seq_mask();

        let old_values = baseline_net.clone().eval_forward(
            obs.clone().inner(),
            traj_length.clone().inner(),
            seq_mask.clone().inner(),
        );
        trace!("old_values={}", old_values);
        let not_done_arr = tensor2ndarray2::<B, bool, Bool>(&not_dones);
        let old_gae_output = rl_utils::get_gae::<B>(
            tensor2ndarray2(&old_values).view(),
            tensor2ndarray2(rewards).view(),
            not_done_arr.view(),
            seq_mask.clone(),
            crate::f32_to_ftype(ppo_config.gae_gamma),
            crate::f32_to_ftype(ppo_config.reward_lambda),
            &device,
        )?;

        let old_logprobs = actor_net
            .clone()
            .eval_forward(
                obs.clone().inner(),
                traj_length.clone().inner(),
                seq_mask.clone().inner(),
            )
            .independent_log_prob(action.clone().inner());
        let old_logprobs = Tensor::<B, 2>::from_inner(old_logprobs);
        let advantages = old_gae_output.advantages;
        let expected_values = old_gae_output.expected_returns;
        let mut update_info = UpdateInfo::new();
        println!("expected_values={}", expected_values);
        println!(
            "expected_values={:?}",
            burn_utils::mean_with_mask(expected_values.clone(), seq_mask.clone().bool())
                .into_data()
                .as_slice::<crate::FType>()
                .unwrap()[0]
        );
        update_info.mean_q_val =
            burn_utils::mean_with_mask(expected_values.clone(), seq_mask.clone().bool())
                .into_data()
                .as_slice()
                .unwrap()[0];
        let mut actor_loss: crate::FType;
        let mut baseline_loss: crate::FType;
        let mini_batch_size = ppo_config.mini_batch_size.min(memory.len());
        let extra_mini_batch_map_2d = HashMap::from([
            ("old_logprobs", old_logprobs),
            ("expected_values", expected_values),
            ("advantages", advantages),
            ("seq_mask", seq_mask.clone()),
        ]);
        let extra_mini_batch_map_3d = HashMap::from([
            ("actor_net_mask", actor_net_mask.clone()),
            ("baseline_net_mask", baseline_net_mask.clone()),
        ]);

        let mini_batch_iter = memory.mini_batch_iter(
            ppo_config.update_freq,
            memory.len() / mini_batch_size,
            Some(extra_mini_batch_map_2d),
            Some(extra_mini_batch_map_3d),
        );
        for (mini_obs, mini_action, mini_traj_length, mini_batch_map_2d, mini_batch_map_3d) in
            mini_batch_iter
        {
            let mini_old_logprobs = mini_batch_map_2d["old_logprobs"].clone();
            let mini_expected_values = mini_batch_map_2d["expected_values"].clone();
            let mini_advantages = mini_batch_map_2d["advantages"].clone();
            let mini_seq_mask = mini_batch_map_2d["seq_mask"].clone();

            let mini_actor_net_mask = mini_batch_map_3d["actor_net_mask"].clone();
            let mini_baseline_net_mask = mini_batch_map_3d["baseline_net_mask"].clone();
            let valid_timestep = mini_seq_mask.clone().sum();
            (actor_net, actor_loss) = Self::update_actor(
                actor_net,
                mini_obs.clone(),
                mini_traj_length.clone(),
                valid_timestep.clone(),
                mini_seq_mask.clone(),
                mini_actor_net_mask.clone(),
                mini_action.clone(),
                mini_advantages.clone(),
                mini_old_logprobs,
                actor_optimizer,
                config,
            );
            (baseline_net, baseline_loss) = Self::update_baseline(
                baseline_net,
                mini_obs.clone(),
                mini_traj_length,
                valid_timestep.clone(),
                mini_baseline_net_mask.clone(),
                mini_seq_mask.clone(),
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
