use burn::{
    nn::{Linear, LinearConfig, Tanh},
    prelude::*,
};

#[derive(Module, Debug)]
pub struct MlpLayer<B: Backend> {
    linear: Linear<B>,
    active_func: Tanh,
}

impl<B: Backend> MlpLayer<B> {
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        return self.linear.forward(input);
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
        return MlpLayer {
            linear: LinearConfig::new(self.d_input, self.d_output).init(device),
            active_func: Tanh::new(),
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
