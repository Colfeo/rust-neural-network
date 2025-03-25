mod load;
mod neural_network;

use ndarray::s;

fn main() {
        println!("Hello, world!");

        // Load the data
        let (train_data, train_labels, _test_data, _test_labels) = load::mnist_loader(false);
        println!("Data loaded");
        println!("Train data dimensions: {:?}", train_data.dim());
        println!("Train labels dimensions: {:?}", train_labels.dim());
        
        // Reshape the labels
        let train_labels = train_labels.into_shape((50000, 1)).expect("Reshape failed");
        let mut one_hot_labels = ndarray::Array2::<f64>::zeros((50000, 10));
        for (i, label) in train_labels.iter().enumerate() {
            let label = *label as usize;
            if label >= 10 {
                panic!("Label out of range: {}", label);
            }
            one_hot_labels[[i, label]] = 1.0;
        }
        let train_labels = one_hot_labels;
        println!("Train labels dimensions: {:?}", train_labels.dim());

        // Reshape the data
        let train_data = train_data.into_shape((50000, 28 * 28)).expect("Reshape failed");
        let train_data = train_data.t().to_owned();

        println!("Train data dimensions: {:?}", train_data.dim());

        // Create the neural network and run a forward pass
        let nn = neural_network::NeuralNetwork::new(784, 128, 10);
        let output = nn.forward(&train_data.slice(s![.., 0]).mapv(|x| x as f64));
        println!("{:?}", output);     

}
