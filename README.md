# RUST Neural Network for MNIST Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📊 Project Overview

This project I've implemented a fully-functional two-layer neural network to classify handwritten digits from the MNIST dataset. The model processes 28×28 grayscale images (784 pixels) and predicts digits 0-9.

Why RUST ? Why not ?

## 🧠 Network Architecture

The neural network features a simple yet effective structure:

- **Input Layer**: 784 neurons (one per pixel)
- **Hidden Layer**: 128 neurons with ReLU activation
- **Output Layer**: 10 neurons with softmax activation

All implemented in pure RUST without external machine learning libraries. Note that the number of neurons for each layer has been chosen arbitrarily and can be easely modified.  

## 📈 Training Performance

 HOLDER 

## 🚀 Getting Started


### Prerequisites

- RUST
- MNIST dataset in .gz format (not included due to size)

### Dataset Setup

**Note:** The MNIST dataset files are not included in this repository due to their large size.
 1. download the file from [kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)
 2. unzip the .gz file in the data directory

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Colfeo/rust-neural-network
   cd rust-neural-network
   ```
2. Update/install dependencies:
   ```bash
   cargo update
   ```
3. Create and train a Neural Network:
   ```bash
   cargo run --bin train
   ```
4. Test perfomance of the Neural Network:
   ```bash
   cargo run --bin performance
   ```
5. TO DO determine a number based on a .png image in input


### Usage

When build the binaries can be run by :
```bash
./target/debug/train
```
```bash
./target/debug/performance
```
```bash
./target/debug/main
```
or if build for for release : 
```bash
./target/release/train
```
```bash
./target/release/performance
```
```bash
./target/release/main
```


## 📁 Project Structure

```bash
.
├── Cargo.lock
├── Cargo.toml  # toml with dependencies
├── README.md   # This README file
├── data        # Dataset directory 
│   ├── t10k-images-idx3-ubyte # Testing dataset (to be downloaded)
│   ├── t10k-labels-idx1-ubyte # Testing dataset (to be downloaded)
│   ├── train-images-idx3-ubyte # Testing dataset (to be downloaded)
│   └── train-labels-idx1-ubyte # Testing dataset (to be downloaded)
├── src
│   ├── load.rs            # Function for loading the data set
│   ├── main.rs            # TO BE 
│   ├── neural_network.rs  # Function direclty related to the Neural Network 
│   ├── performance.rs     # Load the NN from the weigths directory and compute performance
│   └── train.rs           # Create, train and save a NeuralNetwork 
└── weights                # Neural Network weigth directory 
    ├── bias_1.json        # Bias of the first layer
    ├── bias_2.json        # Bias of the second layer
    ├── weight_1.json      # Weigths of the first layer
    └── weight_2.json      # Weigths of the second layer 

```


## 🔧 How It Works


HOLDER

## 💡 Implementation Details

This implementation started with the Rust Programming Language tutorial and Building a neural network FROM SCRATCH video from youtube and expanded to include new features. 

## 🤝 Contributing

Contributions are welcome! Feel free to submit a Pull Request.

## 📚 Resources

- [Building a neural network FROM SCRATCH ](https://www.youtube.com/watch?v=w8yWXqWQYmU&t=450s)
- [RUST : Get started with Rust](https://www.rust-lang.org/learn)

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🧩 Future Improvements

- Add more hidden layers
- Implement mini-batch gradient descent
- Add dropout for regularization
- Add variable step
- Create visualization tools for results
