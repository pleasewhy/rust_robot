use burn::{
    nn::{Linear, LinearConfig, Lstm, Relu, Tanh},
    prelude::*,
};

#[derive(Module, Debug)]
pub enum BurnForwarder<B: Backend> {
    Linear(Linear<B>),
    Lstm(Lstm<B>),
    Relu(Relu),
    Tanh(Tanh),
}

#[derive(Module, Debug)]
pub struct Sequence<B: Backend> {
    forwarder_vec: Vec<BurnForwarder<B>>,
}

impl<B: Backend> Sequence<B> {
    pub fn push(&mut self, forwarder: BurnForwarder<B>) {
        self.forwarder_vec.push(forwarder);
    }

    pub fn append(&mut self, vec: &mut Vec<BurnForwarder<B>>) {
        self.forwarder_vec.append(vec);
    }

    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let mut out = input.clone();
        for forwarder in &self.forwarder_vec {
            out = match forwarder {
                BurnForwarder::Linear(linear) => linear.forward(out),
                BurnForwarder::Lstm(lstm) => {
                    assert_eq!(D, 3);
                    let shape = out.shape();
                    let (pred, state) = lstm.forward(
                        out.reshape([shape.dims[0], shape.dims[1], shape.dims[2]]),
                        None,
                    );
                    pred.expand(input.shape())
                }
                BurnForwarder::Relu(relu) => relu.forward(out),
                BurnForwarder::Tanh(tanh) => tanh.forward(out),
            }
        }
        return out;
    }
}

pub fn build_mlp<B: Backend>(
    input_size: usize,
    output_size: usize,
    n_layers: usize,
    hidden_dim: usize,
    device: &B::Device,
) -> Sequence<B> {
    let mut seq: Sequence<B> = Sequence {
        forwarder_vec: vec![],
    };
    let mut in_size = input_size;
    for i in 0..n_layers {
        seq.push(BurnForwarder::Linear(
            LinearConfig::new(in_size, hidden_dim).init(device),
        ));
        seq.push(BurnForwarder::Relu(Relu::new()));
        in_size = hidden_dim;
    }
    seq.push(BurnForwarder::Linear(
        LinearConfig::new(hidden_dim, output_size).init(device),
    ));
    return seq;
}

pub fn build_mlp_by_dims<B: Backend>(
    input_size: usize,
    output_size: usize,
    layer_dims: &Vec<usize>,
    device: &B::Device,
) -> Sequence<B> {
    let mut seq: Sequence<B> = Sequence {
        forwarder_vec: vec![],
    };
    let mut in_size = input_size;
    for hidden_dim in layer_dims {
        seq.push(BurnForwarder::Linear(
            LinearConfig::new(in_size, *hidden_dim).init(device),
        ));
        seq.push(BurnForwarder::Relu(Relu::new()));
        in_size = *hidden_dim;
    }
    seq.push(BurnForwarder::Linear(
        LinearConfig::new(in_size, output_size).init(device),
    ));
    return seq;
}
