use burn::{
    nn::{Linear, LinearConfig, Relu, Tanh},
    prelude::*,
};
use tch;

#[derive(Module, Debug)]
pub struct MlpLayer<B: Backend> {
    linear: Linear<B>,
    active_func: Relu,
}

impl<B: Backend> MlpLayer<B> {
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        return self.active_func.forward(self.linear.forward(input));
    }
}

#[derive(Config, Debug)]
pub struct MlpLayerConfig {
    d_input: usize,
    d_output: usize,
}

impl MlpLayerConfig {
    /// Returns the initialized Mlp.
    pub fn init<B: Backend>(&self, device: &B::Device) -> MlpLayer<B> {
        let init = nn::Initializer::KaimingUniform {
            gain: 1.0 / num_traits::Float::sqrt(9.0),
            fan_out_only: true,
        };
        return MlpLayer {
            linear: LinearConfig::new(self.d_input, self.d_output)
                .with_initializer(init)
                .init(device),
            active_func: Relu::new(),
        };
    }
}

#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    layers: Vec<MlpLayer<B>>,
    out_linear: Linear<B>,
}

impl<B: Backend> Mlp<B> {
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let mut out = self.layers[0].forward(input);
        for i in 1..self.layers.len() {
            out = self.layers[i].forward(out);
        }
        return self.out_linear.forward(out);
    }
}

#[derive(Config, Debug)]
pub struct MlpConfig {
    d_input: usize,
    d_output: usize,
    n_hidden_layers: usize,
    hidden_layer_dim: usize,
    #[config(default = "0.5")]
    dropout: f64,
}

impl MlpConfig {
    /// Returns the initialized Mlp.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mlp<B> {
        let mut layers = Vec::<MlpLayer<B>>::new();
        let mut d_input = self.d_input;
        for i in 0..self.n_hidden_layers {
            let linear = MlpLayerConfig::new(d_input, self.hidden_layer_dim).init(device);
            layers.push(linear);
            d_input = self.hidden_layer_dim;
        }
        return Mlp {
            layers,
            out_linear: LinearConfig::new(d_input, self.d_output).init(device),
        };
    }
}

pub fn build_mlp(
    input_size: i64,
    output_size: i64,
    n_layers: i64,
    hidden_dim: i64,
) -> impl tch::nn::Module {
    let mut layers = tch::nn::seq();
    let mut in_size = input_size;
    let vs = tch::nn::VarStore::new(tch::Device::Cpu);
    for i in 0..n_layers {
        layers = layers
            .add(tch::nn::linear(
                vs.root() / format!("layer{}", i),
                in_size,
                hidden_dim,
                Default::default(),
            ))
            .add_fn(|data| data.tanh());
        in_size = hidden_dim;
    }

    return layers.add(tch::nn::linear(
        vs.root() / "outputlayer",
        in_size,
        output_size,
        Default::default(),
    ));
}
