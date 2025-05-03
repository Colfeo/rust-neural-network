//test.rs
//test the neural network performance

mod load;
mod neural_network;


fn main() {
    // Load the data
    let (_train_data, _train_labels, test_data, test_labels) = load::mnist_loader(false);
    println!("Data loaded");

    
    // Reshape the labels
    let test_labels = test_labels.to_shape((10000, 1)).expect("Reshape failed");
    let mut one_hot_labels = ndarray::Array2::<f64>::zeros((10000, 10));
    for (i, label) in test_labels.iter().enumerate() {
        let label = *label as usize;
        if label >= 10 {
            panic!("Label out of range: {}", label);
        }
        one_hot_labels[[i, label]] = 1.0;
    }
    let test_labels = one_hot_labels.t();

    // Reshape the data
    let test_data = test_data.to_shape((10000, 28 * 28)).expect("Reshape failed");
    let test_data = test_data.t().to_owned();

    //load the model
    let path = "weights";
    let nn = neural_network::NeuralNetwork::load_neural_network(path).expect("Model load failed");
    println!("Model loaded");

    // Test the model
    let (_accuracy, _precision, _recall) = nn.performance(&test_data.mapv(|x| x as f64),
                                                    &test_labels.mapv(|x| x as f64));
    println!("Neural Network tested !");
    

}