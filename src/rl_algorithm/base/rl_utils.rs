use burn::module::AutodiffModule;
use burn::optim::{GradientsParams, Optimizer};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::cast::ToElement;
use burn::tensor::{Bool, Element, Int, Tensor, TensorData};
use burn::LearningRate;

use ndarray::{Array1, Array2};
use num_traits::{ToPrimitive, Zero};

use crate::rl_env::env::MujocoEnv;

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

pub fn vec2tensor1<B: Backend, T: Element + Zero + ToPrimitive>(
    arr: Vec<T>,
    device: &B::Device,
) -> Tensor<B, 1> {
    let shape = [arr.len()];
    let tensor_data = TensorData::new(arr, shape);
    return Tensor::<B, 1>::from_data(tensor_data, device);
}

pub fn ndarray2tensor1<B: Backend, T: Element + Zero + ToPrimitive>(
    arr: Array1<T>,
    device: &B::Device,
) -> Tensor<B, 1> {
    let shape = arr.shape().to_vec();
    let vec = arr.into_raw_vec_and_offset().0;
    // let vec = vec.iter().map(|x| *x as f32).collect::<Vec<f32>>();
    let tensor_data = TensorData::new(vec, shape);
    return Tensor::<B, 1>::from_data(tensor_data, device);
}

pub fn ndarray2tensor2<B: Backend, T: Element + Zero + ToPrimitive>(
    arr: Array2<T>,
    device: &B::Device,
) -> Tensor<B, 2> {
    let shape = arr.shape().to_vec();
    let vec = arr.into_raw_vec_and_offset().0;
    let tensor_data = TensorData::new(vec, shape);
    return Tensor::<B, 2>::from_data(tensor_data, device);
}

pub fn bool_ndarray2tensor1<B: Backend>(
    arr: Array1<bool>,
    device: &B::Device,
) -> Tensor<B, 1, Bool> {
    let shape = arr.shape().to_vec();
    let vec = arr.into_raw_vec_and_offset().0;
    // let vec = vec.iter().map(|x| *x as f32).collect::<Vec<f32>>();
    let tensor_data = TensorData::new(vec, shape);
    return Tensor::<B, 1, Bool>::from_data(tensor_data, device);
}

pub fn tensor2ndarray2<B: Backend>(tensor: &Tensor<B, 2>) -> Array2<f32> {
    let vec = tensor.to_data().into_vec::<f32>().unwrap();
    let shape: [usize; 2] = tensor.shape().dims();
    return Array2::from_shape_vec(shape, vec).unwrap();
}

pub fn tensor2vec1<B: Backend>(tensor: &Tensor<B, 1>) -> Vec<f32> {
    let vec = tensor.to_data().into_vec::<f32>().unwrap();
    return vec;
}
pub fn booltensor2vec1<B: Backend>(tensor: &Tensor<B, 1, Bool>) -> Vec<bool> {
    let vec = tensor.to_data().into_vec::<bool>().unwrap();
    return vec;
}
