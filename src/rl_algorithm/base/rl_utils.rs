use burn::module::AutodiffModule;
use burn::optim::{GradientsParams, Optimizer};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::cast::ToElement;
use burn::tensor::{Int, Tensor};
use burn::LearningRate;

use crate::rl_env::nd_vec::vec2tensor1;

pub(crate) fn update_parameters<B: AutodiffBackend, M: AutodiffModule<B>>(
    loss: Tensor<B, 1>,
    module: M,
    optimizer: &mut impl Optimizer<M, B>,
    learning_rate: LearningRate,
) -> M {
    let gradients = loss.backward();
    let gradient_params = GradientsParams::from_grads(gradients, &module);
    optimizer.step(learning_rate, module, gradient_params)
}

pub fn normalize<B: Backend>(tensor: Tensor<B, 1>) -> Tensor<B, 1> {
    return (tensor.clone() - tensor.clone().mean().into_scalar())
        / (tensor.var(0).sqrt().into_scalar().to_f64() + 1e-6);
}

#[derive(Debug)]
pub struct UpdateInfo {
    pub actor_loss: f32,
    pub critic_loss: f32,
    pub mean_q_val: f32,
}

impl UpdateInfo {
    pub fn new() -> Self {
        return Self {
            actor_loss: 0.0,
            critic_loss: 0.0,
            mean_q_val: 0.0,
        };
    }
}

pub(crate) struct GAEOutput<B: Backend> {
    pub expected_returns: Tensor<B, 1>,
    pub advantages: Tensor<B, 1>,
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
        advantages: normalize(vec2tensor1(advantages, device)),
    });
}
