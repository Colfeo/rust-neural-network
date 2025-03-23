mod load;

fn main() {
        println!("Hello, world!");

        
        let (_train_data, _train_labels, _test_data, _test_labels) = load::mnist_loader(true);

}
