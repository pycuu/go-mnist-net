package main

import (
	"fmt"
	"handwritten-nn/dataset"
	"handwritten-nn/nn"
	"math"
	"math/rand"
	"time"
)

func main() {
	// load training and test data
	trainData, err := dataset.ReadCsv("data/mnist_train.csv")
	if err != nil {
		panic(fmt.Sprintf("Error loading training data: %v", err))
	}

	testData, err := dataset.ReadCsv("data/mnist_test.csv")
	if err != nil {
		panic(fmt.Sprintf("Error loading test data: %v", err))
	}

	r := rand.New(rand.NewSource(time.Now().UnixNano()))

	// shuffle training data
	r.Shuffle(len(trainData), func(i, j int) {
		trainData[i], trainData[j] = trainData[j], trainData[i]
	})

	// set network parameters
	layers := []int{784, 128, 64, 10} // Input layer (28x28=784), hidden layers, output layer (10 digits)
	learningRate := 0.1
	epochs := 10
	batchSize := 32

	// initialize neural network
	net, err := nn.NewNetwork(layers)
	if err != nil {
		panic(fmt.Sprintf("Error creating network: %v", err))
	}

	// training loop
	fmt.Println("Starting training...")
	for epoch := 0; epoch < epochs; epoch++ {
		startTime := time.Now()
		totalLoss := 0.0
		correct := 0

		// Process in batches
		for i := 0; i < len(trainData); i += batchSize {
			end := i + batchSize
			if end > len(trainData) {
				end = len(trainData)
			}
			batch := trainData[i:end]

			// Accumulate gradients across the batch
			var batchGradW [][][]float64
			var batchGradB [][]float64

			// Initialize batch gradients
			for l := 0; l < len(net.Layers)-1; l++ {
				current := net.Layers[l]
				next := net.Layers[l+1]

				gradW := make([][]float64, next)
				for i := 0; i < next; i++ {
					gradW[i] = make([]float64, current)
				}
				gradB := make([]float64, next)

				batchGradW = append(batchGradW, gradW)
				batchGradB = append(batchGradB, gradB)
			}

			// Process each sample in the batch
			for _, sample := range batch {
				gradW, gradB, err := net.Backward(sample.Input, sample.Target)
				if err != nil {
					panic(fmt.Sprintf("Error during backpropagation: %v", err))
				}

				// Accumulate gradients
				for l := range gradW {
					for i := range gradB[l] {
						batchGradB[l][i] += gradB[l][i]
						for j := range gradW[l][i] {
							batchGradW[l][i][j] += gradW[l][i][j]
						}
					}
				}

				// Calculate loss and accuracy for this sample
				output, _, _, err := net.Forward(sample.Input)
				if err != nil {
					panic(fmt.Sprintf("Error during forward pass: %v", err))
				}

				// Cross-entropy loss
				loss := 0.0
				for k := range output {
					loss += sample.Target[k]*log(output[k]) + (1-sample.Target[k])*log(1-output[k])
				}
				totalLoss += -loss / float64(len(sample.Target))

				// Accuracy
				if argmax(output) == sample.Label {
					correct++
				}
			}

			// Average gradients over the batch and update weights
			for l := range batchGradW {
				for i := range batchGradB[l] {
					batchGradB[l][i] /= float64(len(batch))
					for j := range batchGradW[l][i] {
						batchGradW[l][i][j] /= float64(len(batch))
					}
				}
			}

			net.UpdateWeights(batchGradW, batchGradB, learningRate)
		}

		// Calculate metrics
		avgLoss := totalLoss / float64(len(trainData))
		accuracy := float64(correct) / float64(len(trainData)) * 100
		elapsed := time.Since(startTime)

		fmt.Printf("Epoch %d/%d - Loss: %.4f - Accuracy: %.2f%% - Time: %s\n",
			epoch+1, epochs, avgLoss, accuracy, elapsed)

		// Evaluate on test set after each epoch
		testAccuracy := evaluate(net, testData)
		fmt.Printf("Test Accuracy: %.2f%%\n", testAccuracy)
	}

	fmt.Println("Training completed!")
}

// evaluate calculates the accuracy on the test set
func evaluate(net *nn.Network, testData []dataset.Sample) float64 {
	correct := 0

	for _, sample := range testData {
		output, _, _, err := net.Forward(sample.Input)
		if err != nil {
			panic(fmt.Sprintf("Error during evaluation: %v", err))
		}

		if argmax(output) == sample.Label {
			correct++
		}
	}

	return float64(correct) / float64(len(testData)) * 100
}

func argmax(slice []float64) int {
	maxIndex := 0
	maxValue := slice[0]

	for i, v := range slice {
		if v > maxValue {
			maxValue = v
			maxIndex = i
		}
	}

	return maxIndex
}

func log(x float64) float64 {
	if x < 1e-10 {
		return -23.025850929940457 // log(1e-10)
	}
	return math.Log(x)
}
