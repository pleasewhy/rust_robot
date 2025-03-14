use burn::tensor::Distribution as td;
use burn::tensor::{backend::Backend, ReshapeArgs, Tensor, TensorData};
use rand::prelude::*;
use rand_distr::{Distribution, StandardNormal};
use std::{marker::PhantomData, time::SystemTime};

#[derive(Debug, Clone)]
pub struct Normal<B: Backend> {
    loc: Tensor<B, 2>,   // 均值
    scale: Tensor<B, 1>, // 标准差
}

impl<B: Backend> Normal<B> {
    pub fn new(loc: Tensor<B, 2>, scale: Tensor<B, 1>) -> Self {
        Self { loc, scale }
    }

    pub fn sample(&self) -> Tensor<B, 2> {
        let rng = rand::rng();
        let iter = rng.sample_iter::<f64, StandardNormal>(StandardNormal);
        let vec = iter
            .take(self.loc.shape().num_elements())
            .collect::<Vec<f64>>();
        let shape = [vec.len(); 1];
        let standard_normal_rand =
            Tensor::<B, 1>::from_data(TensorData::new(vec, shape), &self.loc.device());
        let standard_normal_rand = standard_normal_rand.reshape(self.loc.shape());
        return self.loc.clone()
            + standard_normal_rand.clone()
                * self.scale.clone().expand(standard_normal_rand.shape());
    }

    pub fn log_prob(&self, value: Tensor<B, 2>) -> Tensor<B, 2> {
        let pi = std::f32::consts::PI;
        -((value.clone() - self.loc.clone()).powf_scalar(2.0)
            / (self.scale.clone().powf_scalar(2.0).mul_scalar(2.0)).expand(value.shape()))
            - self.scale.clone().log().expand(value.shape())
            - (2f32 * pi).sqrt().ln()
    }
    pub fn independent_log_prob(&self, value: Tensor<B, 2>) -> Tensor<B, 1> {
        let x = self.log_prob(value).sum_dim(1).flatten::<1>(0, 1);
        return x;
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
            TensorData::from([[-1.0, -2.0], [1.0, 2.0], [3.0, 4.0]]),
            &NdArrayDevice::default(),
        );
        let scale = Tensor::from_data(TensorData::from([5.0, 6.0]), &NdArrayDevice::default());
        let x = Tensor::<NdArray, 2>::from_floats(
            [[-100.0, 100.0], [7.0, 8.0], [9.0, 10.0]],
            &NdArrayDevice::default(),
        );
        let normal = super::Normal { loc, scale };
        let log_prob = normal.log_prob(x.clone());
        let independent_log_prob = normal.independent_log_prob(x);
        println!("log_prob={}", log_prob);
        println!("independent_log_prob={}", independent_log_prob);
        let expected = Tensor::<NdArray, 2>::from_floats(
            [
                [-198.54839, -147.21071],
                [-3.2483764, -3.2106981],
                [-3.2483764, -3.2106981],
            ],
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
