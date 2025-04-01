// src/neural_network.rs

use ndarray::{Array1, Array2};
use rand::Rng;
use serde_json;

pub struct NeuralNetwork {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    weight_1: Array2<f64>, // Now shape: (hidden_size, input_size)
    weight_2: Array2<f64>, // Now shape: (output_size, hidden_size)
    bias_1: Array2<f64>,
    bias_2: Array2<f64>,   
}

impl NeuralNetwork {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let mut rng = rand::rng();

        let weight_1 = Array2::from_shape_fn((hidden_size, input_size), |_| rng.random_range(-1.0..1.0));
        let weight_2 = Array2::from_shape_fn((output_size, hidden_size), |_| rng.random_range(-1.0..1.0));
        let bias_1 = Array1::from_shape_fn(hidden_size, |_| rng.random_range(-1.0..1.0));
        let bias_2 = Array1::from_shape_fn(output_size, |_| rng.random_range(-1.0..1.0));
        let bias_1 = bias_1.insert_axis(ndarray::Axis(1));
        let bias_2 = bias_2.insert_axis(ndarray::Axis(1));

        NeuralNetwork {
            input_size,
            hidden_size,
            output_size,
            weight_1,
            weight_2,
            bias_1,
            bias_2,
        }
    }
    
    pub fn single_collumn_forward(&self, input: &Array1<f64>) -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
        let hidden_input = self.weight_1.dot(input) + &self.bias_1;
        let hidden_output = hidden_input.mapv(_relu);

        let final_input = self.weight_2.dot(&hidden_output) + &self.bias_2;
        let final_output = _softmax_batch(&final_input);

        (hidden_input, hidden_output, final_input, final_output)
    }
        
    // Modify bias.clone by batch bias of size [nbr_neurons, 1]
    pub fn forward(&self, input: &Array2<f64>) -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {

        //println!("Input shape: {:?}", input.shape());
        //println!("Weight 1 shape: {:?}", self.weight_1.shape());

        let hidden_input = self.weight_1.dot(input) + &self.bias_1;
        let hidden_output = hidden_input.mapv(_relu);

        //println!("Hidden input shape: {:?}", hidden_input.shape());
        //println!("Hidden output shape: {:?}", hidden_output.shape());
        //println!("Weight 2 shape: {:?}", self.weight_2.shape());
        let final_input = self.weight_2.dot(&hidden_output) + &self.bias_2;
        let final_output = _softmax_batch(&final_input);

        (hidden_input, hidden_output, final_input, final_output)
    }
           
    
    pub fn backward(&self, x: &Array2<f64>, y: &Array2<f64>, t: &Array2<f64>, z1: &Array2<f64>, _z2: &Array2<f64>, 
                            a1: &Array2<f64>) 
                            -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
        // Compute the gradients for the weights and biases
        //println!("shape of y : {:?}", y.shape());
        //println!("shape of t : {:?}", t.shape());
        let m = x.shape()[1];
        let output_error = y - t;
        //println!("Output error shape: {:?}", output_error.shape());
        let weight2_delta = output_error.dot(&a1.t()) / m as f64;
        //println!("Output error shape: {:?}", output_error.shape());
        //let bias2_delta = output_error.sum_axis(ndarray::Axis(0));
        
        //println!("m: {:?}", m);
        let bias2_delta = output_error.sum_axis(ndarray::Axis(1)) / m as f64;
        let hidden_error = self.weight_2.t().dot(&output_error) 
            * z1.mapv(_relu_derivative);

        let weight1_delta = hidden_error.dot(&x.t()) / m as f64;
        //println!("Hidden error shape: {:?}", hidden_error.shape());
        let bias1_delta = hidden_error.sum_axis(ndarray::Axis(1)) / m as f64;

        (weight1_delta, bias1_delta.insert_axis(ndarray::Axis(1)), weight2_delta, bias2_delta.insert_axis(ndarray::Axis(1)))
    }
    
    pub fn update_weights(&mut self, weight1_delta: &Array2<f64>, bias1_delta: &Array2<f64>, 
                        weight2_delta: &Array2<f64>, bias2_delta: &Array2<f64>, learning_rate: f64) {
        // Update weights and biases using the deltas
        //println!("HERE Weight1 delta shape: {:?}", weight1_delta.shape());
        self.weight_1 = &self.weight_1 - weight1_delta * learning_rate;
        //println!("HERE Bias1 delta shape: {:?}", bias1_delta.shape());
        //println!("HERE Bias1 shape: {:?}", self.bias_1.shape());
        //println!("HERE Bias2 delta shape: {:?}", bias2_delta.shape());
        self.bias_1 = &self.bias_1 - bias1_delta * learning_rate;
        //println!("HERE Weight2 delta shape: {:?}", weight2_delta.shape());
        self.weight_2 = &self.weight_2 - weight2_delta * learning_rate; 
        //println!("HERE Bias2 delta shape: {:?}", bias2_delta.shape());
        self.bias_2 = &self.bias_2 - bias2_delta * learning_rate;
    
    }
        

    //Add bias to the weight matrix
    pub fn save_weights(&self) -> std::io::Result<()> {
        // Define the file paths
        let input_hidden_path = "weights/weight_1.json";
        let hidden_output_path = "weights/weight_2.json";
        let bias_1_path = "weights/bias_1.json";
        let bias_2_path = "weights/bias_2.json";
        // Create the directory if it doesn't exist
        let input_hidden_file = std::fs::File::create(input_hidden_path)?;
        let hidden_output_file = std::fs::File::create(hidden_output_path)?;
        let bias_1_file = std::fs::File::create(bias_1_path)?;
        let bias_2_file = std::fs::File::create(bias_2_path)?;
        // Convert the weight matrices to Vec<Vec<f64>> for serialization
        let weight_1_vec: Vec<Vec<f64>> = self.weight_1.outer_iter().map(|row| row.to_vec()).collect();
        let weight_2_vec: Vec<Vec<f64>> = self.weight_2.outer_iter().map(|row| row.to_vec()).collect();
        let bias_1_vec: Vec<f64> = self.bias_1.iter().cloned().collect();
        let bias_2_vec: Vec<f64> = self.bias_2.iter().cloned().collect();
        // Serialize and write to files
        serde_json::to_writer(input_hidden_file, &weight_1_vec)?;
        serde_json::to_writer(hidden_output_file, &weight_2_vec)?;
        serde_json::to_writer(bias_1_file, &bias_1_vec)?;
        serde_json::to_writer(bias_2_file, &bias_2_vec)?;
        // Print confirmation
        println!("Weights and biases saved to files.");



        Ok(())
    }


    pub fn load_neural_network(weights_dir: &str) -> std::io::Result<Self> {
        // Define the file paths
        let input_hidden_path = format!("{}/weight_1.json", weights_dir);
        let hidden_output_path = format!("{}/weight_2.json", weights_dir);
        let bias_1_path = format!("{}/bias_1.json", weights_dir);
        let bias_2_path = format!("{}/bias_2.json", weights_dir);
        // Load the weights and biases from files
        let weight_1_file = std::fs::File::open(&input_hidden_path)?;
        let weight_2_file = std::fs::File::open(&hidden_output_path)?;
        let bias_1_file = std::fs::File::open(&bias_1_path)?;
        let bias_2_file = std::fs::File::open(&bias_2_path)?;
        // Deserialize the data
        let weight_1_vec: Vec<Vec<f64>> = serde_json::from_reader(weight_1_file)?;
        let weight_2_vec: Vec<Vec<f64>> = serde_json::from_reader(weight_2_file)?;
        let bias_1_vec: Vec<f64> = serde_json::from_reader(bias_1_file)?;
        let bias_2_vec: Vec<f64> = serde_json::from_reader(bias_2_file)?;
        // Compute sizes based on the loaded data
        let hidden_size = weight_1_vec.len();
        let input_size = weight_1_vec[0].len();
        let output_size = weight_2_vec.len();
        // Convert the vectors back to ndarray arrays
        let weight_1 = Array2::from_shape_vec((hidden_size, input_size), weight_1_vec.into_iter().flatten().collect()).unwrap();
        let weight_2 = Array2::from_shape_vec((output_size, hidden_size), weight_2_vec.into_iter().flatten().collect()).unwrap();
        let bias_1 = Array2::from_shape_vec((hidden_size, 1), bias_1_vec).unwrap();
        let bias_2 = Array2::from_shape_vec((output_size, 1), bias_2_vec).unwrap();
        
        Ok(NeuralNetwork {
        input_size,
        hidden_size,
        output_size,
        weight_1,
        weight_2,
        bias_1,
        bias_2,
        })
    }

}


fn _relu(x: f64) -> f64 {
    if x > 0.0 { x } else { 0.0 }
}

fn _relu_derivative(x: f64) -> f64 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

fn _softmax(x: &Array1<f64>) -> Array1<f64> {
    let max = x.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let exp_x = x.mapv(|v| (v - max).exp());
    let sum_exp_x = exp_x.sum();
    exp_x / sum_exp_x
}

fn _softmax_batch(x: &Array2<f64>) -> Array2<f64> {
    let max = x.fold_axis(ndarray::Axis(0), f64::NEG_INFINITY, |a, &b| a.max(b));
    let exp_x = x - &max.insert_axis(ndarray::Axis(0));
    let exp_x = exp_x.mapv(|v| v.exp());
    let sum_exp_x = exp_x.sum_axis(ndarray::Axis(0)).insert_axis(ndarray::Axis(0));
    
    exp_x / sum_exp_x
}
