use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use ndarray_nn::relu;

use ndarray::prelude::*;

pub struct ConvolutionalNeuralNetwork {
    pub filters: Array4<f32>, // Convolutional filters
    pub weights: Array2<f32>, // Fully connected layer weights
    pub biases: Array1<f32>,  // Fully connected layer biases
}

impl ConvolutionalNeuralNetwork {
    pub fn new(filter_shape: (usize, usize, usize, usize), fc_shape: (usize, usize)) -> Self {
        let filters = Array::random(filter_shape, Uniform::new(-0.1, 0.1));
        let weights = Array::random(fc_shape, Uniform::new(-0.1, 0.1));
        let biases = Array::zeros(fc_shape.1);

        Self {
            filters,
            weights,
            biases,
        }
    }

    pub fn convolve(&self, input: &Array3<f32>) -> Array3<f32> {
        let (filter_count, filter_depth, filter_height, filter_width) = self.filters.dim();
        let (input_depth, input_height, input_width) = input.dim();

        assert_eq!(filter_depth, input_depth, "Filter depth must match input depth");

        let output_height = input_height - filter_height + 1;
        let output_width = input_width - filter_width + 1;

        let mut output = Array3::<f32>::zeros((filter_count, output_height, output_width));

        for f in 0..filter_count {
            for i in 0..output_height {
                for j in 0..output_width {
                    let region = input.slice(s![
                        ..,
                        i..i + filter_height,
                        j..j + filter_width
                    ]);
                    let filter = self.filters.slice(s![f, .., .., ..]);
                    output[[f, i, j]] = (region * filter).sum();
                }
            }
        }

        relu(output) // Apply ReLU activation
    }

    pub fn forward(&self, input: &Array3<f32>) -> Array1<f32> {
        let conv_output = self.convolve(input);
        let flattened = conv_output.into_shape((conv_output.len(),)).unwrap();
        let fc_output = self.weights.dot(&flattened) + &self.biases;
        relu(fc_output) // Apply ReLU activation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cnn() {
        let cnn = ConvolutionalNeuralNetwork::new((8, 3, 3, 3), (72, 10));
        let input = Array::random((3, 8, 8), Uniform::new(0.0, 1.0));
        let output = cnn.forward(&input);
        assert_eq!(output.len(), 10);
    }
}