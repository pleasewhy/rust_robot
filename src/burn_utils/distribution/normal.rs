use burn::tensor::Distribution as td;
use burn::tensor::{backend::Backend, ReshapeArgs, Tensor, TensorData};
use core::f32;
use rand::prelude::*;
use rand_distr::{Distribution, StandardNormal};
use std::{marker::PhantomData, time::SystemTime};

#[derive(Debug, Clone)]
pub struct Normal<B: Backend, const D: usize = 3> {
    loc: Tensor<B, D>,          // 均值
    scale: Tensor<B, 1>,        // 标准差
    mask: Option<Tensor<B, D>>, // mask
}

impl<B: Backend, const D: usize> Normal<B, D> {
    pub fn new(loc: Tensor<B, D>, scale: Tensor<B, 1>, mask: Option<Tensor<B, D>>) -> Self {
        Self { loc, scale, mask }
    }

    pub fn sample(&self) -> Tensor<B, D> {
        let rng = rand::rng();
        let iter = rng.sample_iter::<f64, StandardNormal>(StandardNormal);
        let vec = iter
            .take(self.loc.shape().num_elements())
            .collect::<Vec<f64>>();
        let shape = [vec.len(); 1];
        let standard_normal_rand =
            Tensor::<B, 1>::from_data(TensorData::new(vec, shape), &self.loc.device());
        let standard_normal_rand = standard_normal_rand.reshape(self.loc.shape());
        let res = self.loc.clone()
            + standard_normal_rand.clone()
                * self.scale.clone().expand(standard_normal_rand.shape());
        if let Some(mask) = &self.mask {
            return res.clone().mul(mask.clone());
        }
        return res;
    }

    pub fn log_prob(&self, value: Tensor<B, D>) -> Tensor<B, D> {
        let pi = std::f32::consts::PI;
        let x = -((value.clone() - self.loc.clone()).powf_scalar(2.0)
            / (self.scale.clone().powf_scalar(2.0).mul_scalar(2.0)).expand(value.shape()))
            - self.scale.clone().log().expand(value.shape())
            - (2f32 * pi).sqrt().ln();
        if let Some(mask) = &self.mask {
            return x.clone().mul(mask.clone());
        }
        return x;
    }
    pub fn independent_log_prob(&self, value: Tensor<B, D>) -> Tensor<B, 2> {
        let x = self.log_prob(value).sum_dim(2).flatten::<2>(1, 2);
        return x;
    }
    pub fn entropy(&self) -> Tensor<B, 1> {
        let pi = f32::consts::PI;
        return self.scale.clone().log() + 0.5 + 0.5 * (2.0 * pi).ln();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    use burn::tensor::TensorData;

    #[test]
    fn test_log_prob() {
        let loc = Tensor::from_data(
            TensorData::from([[[-1.0, -2.0], [1.0, 2.0], [3.0, 4.0]]]),
            &NdArrayDevice::default(),
        );
        let scale = Tensor::from_data(TensorData::from([5.0, 6.0]), &NdArrayDevice::default());
        let x = Tensor::<NdArray, 3>::from_floats(
            [[[-100.0, 100.0], [7.0, 8.0], [9.0, 10.0]]],
            &NdArrayDevice::default(),
        );
        let normal = super::Normal {
            loc,
            scale,
            mask: None,
        };
        let log_prob = normal.log_prob(x.clone());
        let independent_log_prob = normal.independent_log_prob(x);
        println!("log_prob={}", log_prob);
        println!("independent_log_prob={}", independent_log_prob);
        let expected = Tensor::<NdArray, 3>::from_floats(
            [[
                [-198.54839, -147.21071],
                [-3.2483764, -3.2106981],
                [-3.2483764, -3.2106981],
            ]],
            &NdArrayDevice::default(),
        );

        assert!((log_prob - expected).abs().sum().into_scalar() < 1e-6);
    }

    // #[test]
    // fn test_log_prob_off_mean() {
    //     let normal = create_normal(2.0, 0.5);
    //     let value = Tensor::from_data(TensorData::from([[3.0]]), &NdArray::default());

    //     let log_prob = normal.log_prob(value).into_data().value[0];
    //     let manual_calc = (-((3.0 - 2.0).powi(2))/(2.0 * 0.5f32.powi(2)).ln()
    //                       - 0.5f32.ln()
    //                       - (2.0 * std::f32::consts::PI).ln()/2.0) as f64;

    //     assert!((log_prob - manual_calc).abs() < 1e-6);
    // }

    // #[test]
    // fn test_batch_input() {
    //     let normal = create_normal(1.0, 2.0);
    //     let values = Tensor::from_data(TensorData::from([[1.0], [2.0], [3.0]]), &NdArray::default());

    //     let log_probs = normal.log_prob(values).into_data().value;
    //     assert_eq!(log_probs.len(), 3);

    //     // 验证第一个元素（值=均值）是否最大
    //     assert!(log_probs[0] > log_probs[1]);
    //     assert!(log_probs[0] > log_probs[2]);
    // }
}
