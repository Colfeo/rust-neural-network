// src/neural_network.rs

use ndarray::{Array1, Array2};
use rand::Rng;

pub struct NeuralNetwork {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    weights_input_hidden: Array2<f64>,
    weights_hidden_output: Array2<f64>,
}

impl NeuralNetwork {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let mut rng = rand::thread_rng();

        let weights_input_hidden = Array2::from_shape_fn((input_size, hidden_size), |_| rng.gen_range(-1.0..1.0));
        let weights_hidden_output = Array2::from_shape_fn((hidden_size, output_size), |_| rng.gen_range(-1.0..1.0));

        NeuralNetwork {
            input_size,
            hidden_size,
            output_size,
            weights_input_hidden,
            weights_hidden_output,
        }
    }

    pub fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        let hidden_input = self.weights_input_hidden.dot(input);
        let hidden_output = hidden_input.mapv(sigmoid);

        let final_input = self.weights_hidden_output.dot(&hidden_output);
        let final_output = final_input.mapv(sigmoid);

        final_output
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
