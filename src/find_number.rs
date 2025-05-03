//find_number.rs
// Load image and predict the number
use std::env;

use image::ImageReader;
use image::GenericImageView;

use ndarray_stats::QuantileExt;

mod neural_network;


fn main() -> Result<(), Box<dyn std::error::Error>> {

    // Collect the command-line arguments into a vector
    let args: Vec<String> = env::args().collect();

    // Check if a name was provided
    if args.len() > 1 {
        let path = &args[1];
        println!("Path to number to read : {}!", path);
    } else {
        println!("Please provide a name as an argument.");
    }

    // Load the image
    //let path = "/Users/damienvanoldeneel/Downloads/image_test_mnist/6_mnist_type.png";
    let path = "/Users/damienvanoldeneel/Downloads/image_test_mnist/6.1_mnist_type.jpeg";
    //let path = "/Users/damienvanoldeneel/Downloads/image_test_mnist/8_sans_fond.jpeg";
    let img = ImageReader::open(path)?.decode()?;

    // Convert the image to grayscale
    let img = img.grayscale();
    // Resize the image to 28x28
    let img = img.resize_exact(28,28, image::imageops::FilterType::Triangle);
    
    let image = false;
    if image == true{
        img.save("test.png").unwrap();
    }

    if img.dimensions() != (28,28) {
        panic!("Image failed to reshape to 28x28 format !");
    }

    //convert image to array
    let img = img.to_luma32f();

    let img = img.into_raw();
    let vector = ndarray::Array2::from_shape_vec((28, 28), img).expect("Error converting image to Array2 struct");
    let vector = vector.to_shape((28 * 28,1)).expect("Reshape failed");
    

    //load the model
    let path = "weights";
    let nn = neural_network::NeuralNetwork::load_neural_network(path).expect("Model load failed");
    

    let (_z1, _a1, _z2, y) = nn.forward(&vector.mapv(|x| x as f64));
    let number = y.argmax().unwrap();
    // take the first element of the tuple and convert it to a u8
    let number = number.0 as u8;
    println!("Predicted number: {:?}", number);


    Ok(())
}