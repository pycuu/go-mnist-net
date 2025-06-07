# Handwritten Digit Recognition Neural Network (Go)

This project implements a simple fully connected feedforward neural network from scratch in Go, designed to recognize handwritten digits from the MNIST dataset.

## Features

- Custom neural network written in pure Go — no ML libraries
- Trainable using MNIST images
- Configurable architecture and training parameters
- Generates result CSVs for further analysis
- Includes unit tests to verify correctness
- Clean and hackable code structure

---

## Getting Started

### 1. Clone and Build

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
go build
```

## 2. Training the Network

- Open the main.go file and make sure the following line is uncommented:
```bash
trainingLoop(layers, learning_rate, epochs, batch_size)
```
- This will launch the training loop using the parameters defined in main():
```bash
layers := []int{784, 16, 16, 10} // input layer (28x28), two hidden layers, 10 output classes
learning_rate := 0.6
epochs := 12
batch_size := 32
```
- You can adjust the network structure, learning rate, number of epochs, and batch size directly in main().

### 3. Generating Output for Analysis

- If you want to run the network with a range of parameters and save the results for analysis, comment out the training loop and uncomment the following line:
```bash
testParameters()
```
- This will generate a CSV file at results/output.csv, which you can analyze using the  Python script included in the folder "results":
```bash
python3 analysis.py
```
- Make sure Python 3 and any required libraries (e.g. matplotlib, pandas) are installed.
### 4. Running Tests

- Unit tests are available in network_test.go. Run them using:
```bash
go test -v
```
- These tests ensure the core functionality of the network is working as expected.
### 5. Planned Improvements

- This is a work in progress. Planned future improvements include:

    Optimization of training speed and memory usage

    More efficient weight update algorithms

    Support for additional activation functions and optimizers

    Better analysis and visualization tools

License

MIT – free to use, modify, and share.