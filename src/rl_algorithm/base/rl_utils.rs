use std::marker::PhantomData;

use burn::module::{list_param_ids, AutodiffModule, ModuleVisitor, ParamId};
use burn::optim::{GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::cast::ToElement;
use burn::tensor::{self, BasicOps, Bool, Element, Int, Tensor, TensorData, TensorKind};
use burn::LearningRate;

use log::{trace, warn};
use ndarray::{Array1, Array2, Array3, ArrayView2, AssignElem};

struct GradMeanVisitor<'a, B: AutodiffBackend> {
    gradient_params: &'a GradientsParams,
    phantom: PhantomData<B>,
}

impl<B: AutodiffBackend> ModuleVisitor<B> for GradMeanVisitor<'_, B>
where
    B: Backend,
{
    fn visit_float<const D: usize>(&mut self, id: ParamId, _tensor: &Tensor<B, D>) {
        if let Some(grad) = self.gradient_params.get::<B::InnerBackend, D>(id) {
            trace!("grad={}", grad);
            if grad.is_nan().any().into_scalar() {
                println!("grad has nan");
                std::process::exit(-1);
            }
        } else {
            println!("unknown grad");
        }
    }
}

pub(crate) fn update_parameters<B: AutodiffBackend, M: AutodiffModule<B>>(
    loss: Tensor<B, 1>,
    module: M,
    optimizer: &mut impl Optimizer<M, B>,
    learning_rate: LearningRate,
) -> M {
    trace!("loss={}", loss);
    let gradients = loss.backward();
    let gradient_params: GradientsParams = GradientsParams::from_grads(gradients, &module);
    let mut visitor: GradMeanVisitor<'_, B> = GradMeanVisitor {
        gradient_params: &gradient_params,
        phantom: PhantomData,
    };
    module.visit(&mut visitor);

    optimizer.step(learning_rate, module, gradient_params)
}

fn mean_with_mask<B: Backend>(tensor: Tensor<B, 2>, mask: Tensor<B, 2>) -> f32 {
    let num_elements = mask.clone().sum().into_scalar().to_f32();
    let mean = (tensor.clone() * mask.clone())
        .div_scalar(num_elements)
        .sum();
    return mean.into_scalar().to_f32();
}

fn std_with_mask<B: Backend>(tensor: Tensor<B, 2>, mean: f32, mask: Tensor<B, 2>) -> Tensor<B, 1> {
    let num_elements = mask.clone().sum().into_scalar().to_f32();
    let std = ((tensor.clone() - mean).powi_scalar(2) * mask.clone())
        .div_scalar(num_elements - 1.0)
        .sum()
        .sqrt();

    return std;
}
pub fn normalize_with_mask<B: Backend>(tensor: Tensor<B, 2>, mask: Tensor<B, 2>) -> Tensor<B, 2> {
    let mean = mean_with_mask(tensor.clone(), mask.clone());
    let std = std_with_mask(tensor.clone(), mean, mask.clone());

    return (tensor.clone() - mean).mul(mask) / (std + 1e-6).into_scalar();
}

#[derive(Debug)]
pub struct UpdateInfo {
    pub actor_loss: crate::FType,
    pub critic_loss: crate::FType,
    pub mean_q_val: crate::FType,
}

impl UpdateInfo {
    pub fn new() -> Self {
        return Self {
            actor_loss: crate::FType::ZERO,
            critic_loss: crate::FType::ZERO,
            mean_q_val: crate::FType::ZERO,
        };
    }
}

pub(crate) struct GAEOutput<B: Backend> {
    pub expected_returns: Tensor<B, 2>,
    pub advantages: Tensor<B, 2>,
}
pub(crate) fn get_gae<B: Backend>(
    values: ArrayView2<crate::FType>,
    rewards: ArrayView2<crate::FType>,
    not_dones: ArrayView2<bool>,
    seq_mask: Tensor<B, 2>,
    reward_gamma: crate::FType,
    gae_lambda: crate::FType,
    device: &B::Device,
) -> Option<GAEOutput<B>> {
    let batch_size = values.shape()[0];
    let traj_length = values.shape()[1];
    let shape = (batch_size, traj_length);
    let mut returns = Array2::zeros(shape);
    let mut advantages = Array2::zeros(shape);
    let start = std::time::SystemTime::now();
    for b in 0..batch_size {
        let mut running_return = crate::FType::ZERO;
        let mut running_advantage = crate::FType::ZERO;
        for i in (0..traj_length).rev() {
            let now_idx = (b, i);
            let reward = rewards.get(now_idx)?;
            let not_done = crate::FType::from_f32(*not_dones.get(now_idx)? as i8 as f32);
            running_return = reward + reward_gamma * running_return * not_done;
            running_advantage = reward - values.get(now_idx)?
                + reward_gamma
                    * not_done
                    * (values.get((b, i + 1)).unwrap_or(&crate::FType::ZERO)
                        + gae_lambda * running_advantage);

            *returns.get_mut(now_idx).unwrap() = running_return;
            *advantages.get_mut(now_idx).unwrap() = running_advantage;
        }
    }
    // println!("rewards: {:?}", rewards);
    // println!("expected_returns: {:?}", returns);
    // println!("advantages: {:?}", advantages);
    let res = Some(GAEOutput {
        expected_returns: ndarray2tensor2(returns, device),
        // advantages: normalize_with_mask(ndarray2tensor2(advantages, device), seq_mask),
        advantages: ndarray2tensor2(advantages, device),
    });

    return res;
}

pub fn ndarray2tensor1<B: Backend, ArrType: Element, TensorType>(
    arr: Array1<ArrType>,
    device: &B::Device,
) -> Tensor<B, 1, TensorType>
where
    TensorType: TensorKind<B> + BasicOps<B>,
{
    let shape = arr.shape().to_vec();
    let vec = arr.into_raw_vec_and_offset().0;
    // let vec = vec.iter().map(|x| *x as f32).collect::<Vec<f32>>();
    let tensor_data = TensorData::new(vec, shape);
    return Tensor::<B, 1, TensorType>::from_data(tensor_data, device);
}

pub fn ndarray2tensor2<B: Backend, ArrType: Element, TensorType>(
    arr: Array2<ArrType>,
    device: &B::Device,
) -> Tensor<B, 2, TensorType>
where
    TensorType: TensorKind<B> + BasicOps<B>,
{
    let shape = arr.shape().to_vec();
    let vec = arr.into_raw_vec_and_offset().0;
    let tensor_data = TensorData::new(vec, shape);
    return Tensor::<B, 2, TensorType>::from_data(tensor_data, device);
}

pub fn ndarray2tensor3<B: Backend, ArrType: Element, TensorType>(
    arr: Array3<ArrType>,
    device: &B::Device,
) -> Tensor<B, 3, TensorType>
where
    TensorType: TensorKind<B> + BasicOps<B>,
{
    let shape = arr.shape().to_vec();
    let vec = arr.into_raw_vec_and_offset().0;
    let tensor_data = TensorData::new(vec, shape);
    return Tensor::<B, 3, TensorType>::from_data(tensor_data, device);
}

pub fn tensor2ndarray1<B: Backend, ArrType: Element, TensorType>(
    tensor: &Tensor<B, 1, TensorType>,
) -> Array1<ArrType>
where
    TensorType: TensorKind<B> + BasicOps<B>,
{
    let vec = tensor.to_data().into_vec::<ArrType>().unwrap();
    let shape: [usize; 1] = tensor.shape().dims();
    return Array1::from_shape_vec(shape, vec).unwrap();
}

pub fn tensor2ndarray2<B: Backend, ArrType: Element, TensorType>(
    tensor: &Tensor<B, 2, TensorType>,
) -> Array2<ArrType>
where
    TensorType: TensorKind<B> + BasicOps<B>,
{
    let vec = tensor.to_data().into_vec::<ArrType>().unwrap();
    let shape: [usize; 2] = tensor.shape().dims();
    return Array2::from_shape_vec(shape, vec).unwrap();
}

pub fn tensor2ndarray3<B: Backend, ArrType: Element, TensorType>(
    tensor: &Tensor<B, 3, TensorType>,
) -> Array3<ArrType>
where
    TensorType: TensorKind<B> + BasicOps<B>,
{
    let vec = tensor.to_data().into_vec::<ArrType>().unwrap();
    let shape: [usize; 3] = tensor.shape().dims();
    return Array3::from_shape_vec(shape, vec).unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    use burn::prelude::*;
    use burn::tensor::TensorData;
    #[test]
    fn test_std_with_mask() {
        let device = NdArrayDevice::Cpu;
        let tensor: Tensor<NdArray, 2> =
            Tensor::from_data(TensorData::from([[1.0, 2.0], [3.0, 4.0]]), &device);
        let mask: Tensor<NdArray, 2> =
            Tensor::from_data(TensorData::from([[1.0, 0.0], [1.0, 1.0]]), &device);
        let num_elements: f32 = mask.clone().sum().into_scalar().to_f32();

        let mean = mean_with_mask(tensor.clone(), mask.clone());
        let std: Tensor<NdArray, 1> = std_with_mask::<NdArray>(tensor.clone(), mean, mask.clone());
        // println!("num_elements: {:?}", num_elements);
        // println!("mean: {:?}", mean);
        // println!("std: {:?}", std);
        assert!((std.into_scalar().to_f32() - 1.5275252) < 1e-6);
    }
}
