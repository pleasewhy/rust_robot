use burn::module::AutodiffModule;
use burn::optim::{GradientsParams, Optimizer};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Int, Tensor};
use burn::tensor::cast::ToElement;
use burn::LearningRate;

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
        / (tensor
            .var(0)
            .sqrt()
            .into_scalar()
            .to_f64()
            + 1e-6);
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
