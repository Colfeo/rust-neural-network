//train.rs 
// Train the neural network

mod load;
mod neural_network;

use ndarray::s;

fn main() {
        // Load the data
        let (train_data, train_labels, _test_data, _test_labels) = load::mnist_loader(false);
        println!("Data loaded");
        
        // Reshape the labels
        let train_labels = train_labels.to_shape((50000, 1)).expect("Reshape failed");
        let mut one_hot_labels = ndarray::Array2::<f64>::zeros((50000, 10));
        for (i, label) in train_labels.iter().enumerate() {
            let label = *label as usize;
            if label >= 10 {
                panic!("Label out of range: {}", label);
            }
            one_hot_labels[[i, label]] = 1.0;
        }
        let train_labels = one_hot_labels.t();

        // Reshape the data
        let train_data = train_data.to_shape((50000, 28 * 28)).expect("Reshape failed");
        let train_data = train_data.t().to_owned();


        // Create the neural network and run a forward pass
        let mut nn = neural_network::NeuralNetwork::new(784, 128, 10);

        let _ = nn.gradient_descent(
            &train_data.slice(s![.., ..]).mapv(|x| x as f64),
            &train_labels.slice(s![..,..]).mapv(|x| x as f64),
            0.01,
            1_000,
        );
        
        // Save the model
        let _ = nn.save_weights();


}
