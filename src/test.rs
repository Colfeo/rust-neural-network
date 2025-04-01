use ndarray::{Array1, Array2, Axis};

fn main() {
    // Example input matrix (e.g., batch of input features)
    let input = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    // Example weight matrix
    let weight_1 = Array2::from_shape_vec((2, 3), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap();

    // Example bias vector
    let bias_1 = Array1::from_vec(vec![2., 5.]);

    // Print shapes
    println!("Input shape: {:?}", input.shape());
    println!("Weight 1 shape: {:?}", weight_1.shape());

    // Create a bias matrix by inserting a new axis
    let bias_matrix = bias_1.insert_axis(Axis(1));
    println!("Bias 1 shape: {:?}", bias_matrix.shape());

    // Perform matrix multiplication and add bias
    let hidden_input = weight_1.dot(&input) + &bias_matrix;
    println!("Hidden input shape: {:?}", hidden_input);

    let test_input = hidden_input + bias_matrix;

    // Print the result
    println!(" test_input:\n{:?}", test_input);
}
