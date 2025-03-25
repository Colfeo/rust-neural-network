// src/neural_network.rs

use ndarray::{Array1, Array2};
use rand::Rng;
use serde_json;


pub struct NeuralNetwork {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    weights_input_hidden: Array2<f64>,
    weights_hidden_output: Array2<f64>,
}

impl NeuralNetwork {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let mut rng = rand::rng();

        let weights_input_hidden = Array2::from_shape_fn((input_size, hidden_size), |_| rng.random_range(-1.0..1.0));
        let weights_hidden_output = Array2::from_shape_fn((hidden_size, output_size), |_| rng.random_range(-1.0..1.0));

        NeuralNetwork {
            input_size,
            hidden_size,
            output_size,
            weights_input_hidden,
            weights_hidden_output,
        }
    }

    pub fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        let hidden_input = input.dot(&self.weights_input_hidden);
        let hidden_output = hidden_input.mapv(sigmoid);

        let final_input = hidden_output.dot(&self.weights_hidden_output);
        let final_output = final_input.mapv(sigmoid);

        final_output
    }

/*     pub fn backward(&self, input: &Array1<f64>, target: &Array1<f64>) {
        let hidden_input = input.dot(&self.weights_input_hidden);
        let hidden_output = hidden_input.mapv(sigmoid);

        let final_input = hidden_output.dot(&self.weights_hidden_output);
        let final_output = final_input.mapv(sigmoid);

        let output_error = target - final_output;
        let output_delta = output_error * final_output * (1.0 - final_output);

        let hidden_error = output_delta.dot(&self.weights_hidden_output.t());
        let hidden_delta = hidden_error * hidden_output * (1.0 - hidden_output);

        let hidden_output = hidden_output.insert_axis(Axis(0));
        let input = input.insert_axis(Axis(0));

        let hidden_delta = hidden_delta.insert_axis(Axis(0));
        let output_delta = output_delta.insert_axis(Axis(0));

        self.weights_hidden_output += hidden_output.t().dot(&output_delta);
        self.weights_input_hidden += input.t().dot(&hidden_delta);
    }

    pub fn train(&self, input: &Array2<f64>, target: &Array2<f64>, epochs: usize) {
        for _ in 0..epochs {
            for (input, target) in input.outer_iter().zip(target.outer_iter()) {
                self.backward(&input, &target);
            }
        }
    } */
    pub fn save_weights(&self) -> std::io::Result<()> {
        let input_hidden_path = "weights/weights_input_hidden.json";
        let hidden_output_path = "weights/weights_hidden_output.json";

        let input_hidden_file = std::fs::File::create(input_hidden_path)?;
        let hidden_output_file = std::fs::File::create(hidden_output_path)?;

        serde_json::to_writer(input_hidden_file, &self.weights_input_hidden)?;
        serde_json::to_writer(hidden_output_file, &self.weights_hidden_output)?;

        Ok(())
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
