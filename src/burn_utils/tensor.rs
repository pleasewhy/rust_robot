use burn::{
    prelude::Backend,
    tensor::{cast::ToElement, Bool, Float, Tensor},
};

use crate::rl_algorithm::base::rl_utils::tensor2ndarray2;

/// Calculate masked mean of tensor elements while preventing overflow
///
/// # Arguments
/// * `tensor` - Input tensor
/// * `mask` - Boolean mask tensor with the same shape as input tensor
///
/// # Returns
/// * Tensor containing the masked mean
pub fn mean_with_mask<B: Backend>(tensor: Tensor<B, 2>, mask: Tensor<B, 2, Bool>) -> Tensor<B, 1>
where
    B: Backend,
{
    let count = mask.clone().int().sum().into_scalar().to_f32();

    let arr = tensor2ndarray2::<B, crate::FType, Float>(&tensor).mapv(|x| x.to_f32());
    let mean = arr.sum() / count;

    let tensor = tensor.mask_fill(mask.bool_not(), mean).sub_scalar(mean);

    tensor.mean() + mean
}
