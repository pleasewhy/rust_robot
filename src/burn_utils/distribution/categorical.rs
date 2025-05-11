use burn::prelude::*;
use burn::tensor::Distribution as td;
use burn::tensor::{backend::Backend, ReshapeArgs, Tensor, TensorData};
use core::f32;
use rand::prelude::*;
use rand_distr::{Distribution, Uniform};
use std::{marker::PhantomData, time::SystemTime};
use video_rs::ffmpeg::decoder::new;

#[derive(Debug, Clone)]
pub struct Categorical<B: Backend> {
    prob: Tensor<B, 4>,         // (B, seq_length, ac_dim, categorical_num)
    scale: Tensor<B, 1>,        // 标准差
    mask: Option<Tensor<B, 3>>, // mask (B, seq_length, action_dim)
    range_start: f32,
    interval: f32,
}

impl<B: Backend> Categorical<B> {
    pub fn new(
        prob: Tensor<B, 4>,
        scale: Tensor<B, 1>,
        mask: Option<Tensor<B, 3>>,
        range_start: f32,
        interval: f32,
    ) -> Self {
        // println!("prob.clone().sum_dim(3)={}", prob.clone().sum_dim(3));
        Self {
            prob: prob.clone(),
            scale,
            mask,
            range_start,
            interval,
        }
    }

    pub fn sample(&self) -> Tensor<B, 3> {
        let device = &self.prob.device();
        let x = Tensor::<B, 4>::random(
            self.prob.shape(),
            burn::tensor::Distribution::Uniform(0.0, 1.0),
            device,
        )
        .mul(self.prob.clone());
        let idx = x.argmax(3).flatten(2, 3).float();
        let mean = idx * self.interval + self.range_start;
        let sample = mean.clone();
        let standard_normal = Tensor::<B, 3>::random(
            mean.shape(),
            burn::tensor::Distribution::Normal(0.0, 1.0),
            device,
        );

        // println!("self.scale={}", self.scale);
        let sample = mean * self.scale.clone().expand(standard_normal.shape());
        if let Some(mask) = &self.mask {
            return sample.clone().mul(mask.clone());
        }
        return sample;
    }

    // value: (B, seq_len, ac_dim)
    pub fn log_prob(&self, value: Tensor<B, 3>) -> Tensor<B, 3> {
        let idx = (value.unsqueeze_dim::<4>(3) - self.range_start) / self.interval;
        // println!("idx={}", idx.clone().int());
        // println!("prob={}", self.prob.clone().int());
        let prob = self.prob.clone().gather(3, idx.int());
        let logprob = prob.add_scalar(1.0).log().flatten(2, 3);
        if let Some(mask) = &self.mask {
            return logprob * mask.clone();
        }
        return logprob;
    }
    pub fn independent_log_prob(&self, value: Tensor<B, 3>) -> Tensor<B, 2> {
        let x = self.log_prob(value).sum_dim(2).flatten::<2>(1, 2);
        return x;
    }
    pub fn entropy(&self) -> Tensor<B, 1> {
        let pi = f32::consts::PI;
        return Tensor::<B, 1>::zeros([1], &self.prob.device());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    use burn::tensor::TensorData;

    #[test]
    fn test_log_prob() {
        //(1,3,1,2) = (B, seq_len, ac_dim, num)
        let prob = Tensor::from_data(
            TensorData::from([[[[0.1, 0.3]], [[0.1, 0.4]], [[0.5, 0.5]]]]),
            &NdArrayDevice::default(),
        );

        let scale = Tensor::zeros([1], &NdArrayDevice::default());

        // (1,3,1)=(B, seq_len, ac_dim)
        let x =
            Tensor::<NdArray, 3>::from_floats([[[0.0], [1.0], [0.0]]], &NdArrayDevice::default());
        let normal = super::Categorical::new(prob, scale, None, 0.0, 1.0);
        let log_prob = normal.log_prob(x.clone());
        // let independent_log_prob = normal.independent_log_prob(x);

        let expected =
            Tensor::<NdArray, 3>::from_floats([[[0.25], [0.8], [0.5]]], &NdArrayDevice::default())
                .log();
        println!("log_prob={}", log_prob);
        // println!("independent_log_prob={}", independent_log_prob);
        println!("expected={}", expected);
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
