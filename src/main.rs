mod load;
mod neural_network;

use ndarray::s;

fn main() {
        // Load the data
        let (train_data, train_labels, _test_data, _test_labels) = load::mnist_loader(false);
        println!("Data loaded");
        //println!("Train data dimensions: {:?}", train_data.dim());
        //println!("Train labels dimensions: {:?}", train_labels.dim());
        
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
        let train_labels = one_hot_labels.t();
        //println!("Train labels dimensions: {:?}", train_labels.dim());

        // Reshape the data
        let train_data = train_data.into_shape((50000, 28 * 28)).expect("Reshape failed");
        let train_data = train_data.t().to_owned();

        //println!("Train data dimensions: {:?}", train_data.dim());

        // Create the neural network and run a forward pass
        let mut nn = neural_network::NeuralNetwork::new(784, 128, 10);
        
        /*                  
        
        let (_z1, _a1, _z2, y) = nn.single_collumn_forward(&train_data.slice(s![0.., 0]).mapv(|x| x as f64));
        let (_z1_batch, _a1_batch, _z2_batch, y_batch) = nn.forward(&train_data.slice(s![.., 0..5]).mapv(|x| x as f64));  
        println!("Neural Network single column out shape : {:?}", y.shape()); 
        println!("Neural Network out shape : {:?}", y_batch.shape());

        let (_weight1_delta_batch, _bias1_delta_batch, _weight2_delta_batch, _bias2_delta_batch) = nn.backward(
            &train_data.slice(s![.., 0..5]).mapv(|x| x as f64),
            &y_batch,
            &train_labels.slice(s![.., 0..5]).mapv(|x| x as f64),
            &_z1_batch,
            &_z2_batch,
            &_a1_batch,
        );

        println!("Weight1_delta batch shape: {:?}", _weight1_delta_batch.shape());

        //update the weights and biases
        nn.update_weights(
            &_weight1_delta_batch,
            &_bias1_delta_batch,
            &_weight2_delta_batch,
            &_bias2_delta_batch,
            0.01,
        );
        println!("Weights and biases updated");
        */

        let _ = nn.gradient_descent(
            &train_data.slice(s![.., 0..5_000]).mapv(|x| x as f64),
            &train_labels.slice(s![.., 0..5_000]).mapv(|x| x as f64),
            0.01,
            50_000,
        );
        
        // Save the model
        let _ = nn.save_weights();

        /*
        //load the model
        let path = "weights";
        let _nn1 = neural_network::NeuralNetwork::load_neural_network(path);
        println!("Model loaded");
        */

}
