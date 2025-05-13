use burn::{
    prelude::Backend,
    tensor::{backend::AutodiffBackend, cast::ToElement, Bool, Float, Tensor},
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
    let is_nan = arr.iter().any(|x| x.is_nan());
    let is_inf = arr.iter().any(|x| x.is_infinite());
    let mean = arr.sum() / count;

    let tensor = tensor
        .mask_fill(mask.bool_not(), mean)
        .sub_scalar(mean)
        .div_scalar(10.0);
    let res = tensor.mean().mul_scalar(10.0) + mean;
    // println!("res={}", res);
    return res;
}

pub fn mse_loss_with_mask<B: AutodiffBackend>(
    logits: Tensor<B, 2>,
    targets: Tensor<B, 2>,
    mask: Tensor<B, 2, Bool>,
) -> Tensor<B, 1> {
    let out = logits.sub(targets);
    let out = out.clamp(-100, 100);
    let out = out.powf_scalar(2.0);
    return mean_with_mask(out, mask);
}
