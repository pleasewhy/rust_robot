use burn::{
    prelude::Backend,
    tensor::{
        backend::AutodiffBackend, cast::ToElement, Bool, DType, ElementLimits, Float, FloatDType,
        Tensor,
    },
};

use crate::{
    rl_algorithm::base::rl_utils::{ndarray2tensor2, tensor2ndarray2},
    MyBackend,
};

fn check_nan_and_inf<B: Backend, const D: usize>(tensor: Tensor<B, D>) {
    let is_pos_inf = tensor
        .clone()
        .equal_elem(crate::FType::INFINITY)
        .any()
        .into_scalar()
        .to_bool();
    let is_neg_inf = tensor
        .clone()
        .equal_elem(crate::FType::NEG_INFINITY)
        .any()
        .into_scalar()
        .to_bool();
    let is_nan = tensor.is_nan().any().into_scalar().to_bool();
    if is_pos_inf || is_neg_inf || is_nan {
        unreachable!("has nan or inf")
    }
}

/// Calculate masked mean of tensor elements while preventing overflow
///
/// # Arguments
/// * `tensor` - Input tensor
/// * `mask` - Boolean mask tensor with the same shape as input tensor
///
/// # Returns
/// * Tensor containing the masked mean
pub fn mean_with_mask<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    mask: Tensor<B, D, Bool>,
    debug_name: &str,
) -> Tensor<B, 1>
where
    B: Backend,
{
    // println!("mean_with_mask tensor={} debug_name={}", tensor, debug_name);
    check_nan_and_inf(tensor.clone());
    let num_element = mask.clone().int().sum().into_scalar().to_f32();

    let mean = tensor
        .clone()
        .cast(FloatDType::F32)
        .sum()
        .div_scalar(num_element)
        .into_data()
        .to_vec::<f32>()
        .unwrap()[0];
    let tensor = tensor.mask_fill(mask.bool_not(), mean).sub_scalar(mean);
    let res = tensor.clone().mean() + mean;
    // println!("mean={}", mean);
    // println!("tensor={}", tensor);
    // println!("res={}", res);

    check_nan_and_inf(res.clone());
    return res;
}

pub fn mse_loss_with_mask<B: AutodiffBackend>(
    logits: Tensor<B, 2>,
    targets: Tensor<B, 2>,
    mask: Tensor<B, 2, Bool>,
) -> Tensor<B, 1> {
    let out = logits.sub(targets);
    let out = out.clamp(-200, 200);
    let out = out.powf_scalar(2.0);
    return mean_with_mask(out, mask, "mse_loss");
}

fn std_with_mask<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    mean: f32,
    mask: Tensor<B, D, Bool>,
) -> f32 {
    let tensor = tensor
        .clone()
        .mul(mask.clone().float())
        .cast(FloatDType::F32);
    let num_elements = mask.clone().int().sum().into_scalar().to_f32();

    let std = ((tensor.clone() - mean).powi_scalar(2))
        .sum()
        .div_scalar(num_elements - 1.0)
        .sqrt();
    let std = std.into_data().to_vec::<f32>().unwrap()[0];

    return std.min(crate::ftype_to_f32(crate::FType::MAX));
}

pub fn normalize_with_mask<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    mask: Tensor<B, D, Bool>,
) -> Tensor<B, D> {
    // println!("normalize_with_mask={}");
    check_nan_and_inf(tensor.clone());
    let mean = mean_with_mask(tensor.clone(), mask.clone(), "normalize")
        .into_scalar()
        .to_f32();

    let std = std_with_mask(tensor.clone(), mean, mask.clone());

    let mut out = (tensor.clone() - mean)
        .mul(mask.float())
        .div_scalar(std + 1e-6);

    check_nan_and_inf(out.clone());
    return out;
}
