use crate::burn_utils::distribution::Normal;
use crate::burn_utils::{build_mlp, Sequence};
use crate::rl_algorithm::base::model::{ActorModel, ModelBasedNet};
use burn::nn::loss::{MseLoss, Reduction};
use burn::nn::Tanh;
use burn::tensor::cast::ToElement;
use burn::{nn::Linear, nn::LinearConfig, prelude::*};
use ndarray::AssignElem;

#[derive(Module, Debug)]
pub struct MlpDynamicNet<B: Backend> {
    obs_net: Sequence<B>,
    ac_dim: usize,
    ob_dim: usize,
    ob_ac_mean: Tensor<B, 1>,
    ob_ac_std: Tensor<B, 1>,
    obs_delta_mean: Tensor<B, 1>,
    obs_delta_std: Tensor<B, 1>,
    reward_mean: Tensor<B, 1>,
    reward_std: Tensor<B, 1>,
}

impl<B: Backend> MlpDynamicNet<B> {
    fn update_stat(&mut self, obs: Tensor<B, 2>, action: Tensor<B, 2>, next_obs: Tensor<B, 2>) {
        let ob_ac = Tensor::cat(vec![obs.clone(), action], 1);
        let obs_delta = next_obs - obs;

        let ob_ac_mean = ob_ac.clone().mean_dim(0).flatten::<1>(0, 1);
        let ob_ac_std = ob_ac.var(0).sqrt().flatten::<1>(0, 1);
        let obs_delta_mean = obs_delta.clone().mean_dim(0).flatten::<1>(0, 1);
        let obs_delta_std = obs_delta.var(0).sqrt().flatten::<1>(0, 1);
        let ob_ac_mean = self.ob_ac_mean.clone() * 0.6 + ob_ac_mean.clone() * 0.4;
        let ob_ac_std = self.ob_ac_std.clone() * 0.6 + ob_ac_std.clone() * 0.4;
        let obs_delta_mean = self.obs_delta_mean.clone() * 0.6 + obs_delta_mean.clone() * 0.4;
        let obs_delta_std = self.obs_delta_std.clone() * 0.6 + obs_delta_std.clone() * 0.4;

        self.ob_ac_mean.assign_elem(ob_ac_mean.clone());
        self.ob_ac_std.assign_elem(ob_ac_std.clone());
        self.obs_delta_mean.assign_elem(obs_delta_mean.clone());
        self.obs_delta_std.assign_elem(obs_delta_std.clone());
    }
}
impl<B: Backend> ModelBasedNet<B> for MlpDynamicNet<B> {
    fn forward(&self, obs: Tensor<B, 2>, action: Tensor<B, 2>) -> Tensor<B, 2> {
        let ob_ac = Tensor::cat(vec![obs.clone(), action], 1);
        // let ob_ac = (ob_ac - ob_ac_mean) / (ob_ac_std + 1e-6);

        let obs_delta = self.obs_net.forward::<2>(ob_ac.clone());
        // let obs_delta = obs_delta * (obs_delta_std + 1e-6) + obs_delta_mean;
        let next_obs = obs + obs_delta;
        return next_obs;
    }

    fn loss(
        &mut self,
        obs: Tensor<B, 2>,
        action: Tensor<B, 2>,
        next_obs: Tensor<B, 2>,
        reward: Tensor<B, 1>,
    ) -> Tensor<B, 1> {
        self.update_stat(obs.clone(), action.clone(), next_obs.clone());
        let pred_next_obs = self.forward(obs, action);
        let next_obs_loss = MseLoss.forward(pred_next_obs, next_obs, Reduction::Mean);
        return next_obs_loss;
    }
}

#[derive(Config, Debug)]
pub struct MlpDynamicNetConfig {
    ob_dim: usize,
    ac_dim: usize,
    n_layers: usize,
    layer_size: usize,
}

impl MlpDynamicNetConfig {
    /// Returns the initialized Mlp.
    pub fn init<B: Backend>(&self, device: &B::Device) -> MlpDynamicNet<B> {
        let obs_net: Sequence<B> = build_mlp(
            self.ob_dim + self.ac_dim,
            self.ob_dim,
            self.n_layers,
            self.layer_size,
            device,
        );

        let ob_ac_mean = Tensor::<B, 1>::zeros([self.ob_dim + self.ac_dim], device);
        let ob_ac_std = Tensor::<B, 1>::ones([self.ob_dim + self.ac_dim], device);
        let obs_delta_mean = Tensor::<B, 1>::zeros([self.ob_dim], device);
        let obs_delta_std = Tensor::<B, 1>::ones([self.ob_dim], device);
        let reward_mean = Tensor::<B, 1>::zeros([1], device);
        let reward_std = Tensor::<B, 1>::ones([1], device);
        return MlpDynamicNet {
            obs_net,
            ac_dim: self.ac_dim,
            ob_dim: self.ob_dim,
            ob_ac_mean,
            ob_ac_std,
            obs_delta_mean,
            obs_delta_std,
            reward_mean,
            reward_std,
        };
    }
}
