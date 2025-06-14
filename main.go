package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"go_mnist_net/dataset"
	"go_mnist_net/nn"
	"log"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"time"
)

func main() {

	// set network parameters for trainingLoop
	layers := []int{784, 32, 16, 10} // input layer (28x28=784), hidden layers, output layer (10 digits)
	learning_rate := 0.6
	epochs := 12
	batch_size := 16

	// trains the model and saves the weights/biases/layers to a file
	trainingLoop(layers, learning_rate, epochs, batch_size)

	// uncomment to predict based on saved network (from file in the same directory as main.go)
	// predicting from a file seems to not work well
	/*
		network, err := nn.LoadNetwork("network_parameters")
		if err != nil {
			fmt.Println(err)
		}
		prediction, err := predictFromFile("7.png", *network)
		if err != nil {
			fmt.Println(err)
		}
		fmt.Println("The image is number", prediction)
	*/

	// uncomment to test different parameters
	//testParameters()

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
	max_index := 0
	max_value := slice[0]

	for i, v := range slice {
		if v > max_value {
			max_value = v
			max_index = i
		}
	}

	return max_index
}
func safeLog(x float64) float64 {
	if x < 1e-10 {
		return -23.025850929940457
	}
	return math.Log(x)
}

func trainingLoop(layers []int, learning_rate float64, epochs int, batch_size int) (test_accuracy float64, err error) {
	// load training and test data
	train_data, err := dataset.ReadCsv("data/mnist_train.csv")
	if err != nil {
		panic(fmt.Sprintf("Error loading training data: %v", err))
	}

	testData, err := dataset.ReadCsv("data/mnist_test.csv")
	if err != nil {
		panic(fmt.Sprintf("Error loading test data: %v", err))
	}

	r := rand.New(rand.NewSource(time.Now().UnixNano()))

	// shuffle training data
	r.Shuffle(len(train_data), func(i, j int) {
		train_data[i], train_data[j] = train_data[j], train_data[i]
	})
	// initialize neural network
	net, err := nn.NewNetwork(layers)
	if err != nil {
		panic(fmt.Sprintf("Error creating network: %v", err))
	}

	// training loop

	fmt.Printf("Starting training for parameters:  hidden layer one %d neurons, hidden layer two %d neurons, learning rate %.2f, epochs %d, batch size %d \n", layers[1], layers[2], learning_rate, epochs, batch_size)
	for epoch := 0; epoch < epochs; epoch++ {
		start_time := time.Now()
		total_loss := 0.0
		correct := 0

		for i := 0; i < len(train_data); i += batch_size {
			end := i + batch_size
			if end > len(train_data) {
				end = len(train_data)
			}
			batch := train_data[i:end]

			var batchGradW [][][]float64
			var batchGradB [][]float64

			// initialize batch gradients
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

			// process each sample in the batch
			for _, sample := range batch {
				gradW, gradB, err := net.Backward(sample.Input, sample.Target)
				if err != nil {
					panic(fmt.Sprintf("Error during backpropagation: %v", err))
				}

				for l := range gradW {
					for i := range gradB[l] {
						batchGradB[l][i] += gradB[l][i]
						for j := range gradW[l][i] {
							batchGradW[l][i][j] += gradW[l][i][j]
						}
					}
				}

				// calculate loss and accuracy for this sample
				output, _, _, err := net.Forward(sample.Input)
				if err != nil {
					panic(fmt.Sprintf("Error during forward pass: %v", err))
				}
				// cross-entropy loss
				loss := 0.0
				for k := range output {
					loss += sample.Target[k]*safeLog(output[k]) + (1-sample.Target[k])*safeLog(1-output[k])
				}
				total_loss += -loss / float64(len(sample.Target))
				total_loss += -loss / float64(len(sample.Target))

				// Accuracy
				if argmax(output) == sample.Label {
					correct++
				}
			}

			// average gradients over the batch and update weights
			for l := range batchGradW {
				for i := range batchGradB[l] {
					batchGradB[l][i] /= float64(len(batch))
					for j := range batchGradW[l][i] {
						batchGradW[l][i][j] /= float64(len(batch))
					}
				}
			}

			net.UpdateWeights(batchGradW, batchGradB, learning_rate)
		}

		avgLoss := total_loss / float64(len(train_data))
		accuracy := float64(correct) / float64(len(train_data)) * 100
		elapsed := time.Since(start_time)

		fmt.Printf("Epoch %d/%d - avgLoss: %.4f - Accuracy: %.2f%% - Time: %s\n",
			epoch+1, epochs, avgLoss, accuracy, elapsed)

		// evaluate on test set after each epoch
		test_accuracy = evaluate(net, testData)
		fmt.Printf("Test Accuracy: %.2f%%\n", test_accuracy)
	}

	fmt.Println("Training completed!")
	fmt.Print("\n")

	err = net.SaveNetwork("network_parameters")
	if err != nil {
		fmt.Println(err)
	}
	fmt.Println("weights saved")

	return test_accuracy, err
}

// tests some parameters and saves the results to a csv (can adjust the parameters)
func testParameters() {
	// set the epochs count
	epochs := 12
	layers := []int{784, 16, 16, 10} // input layer (28x28=784), hidden layers, output layer (10 digits)
	var learning_rate float64
	var batch_size int

	file, err := os.Create("results/output.csv")
	if err != nil {
		panic(err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	err = writer.Write([]string{"Accuracy", "Neurons in hidden layer 1", "Neurons in hidden layer 2", "Learning rate", "Batch size"})
	if err != nil {
		log.Fatal("Failed to write header:", err)
	}

	// lo - neurons in hidden layer 1, lt - neurons in hidden layer two, lr - learning rate, bs - batch size
	// the parameters are adjustable by changing the for loops
	for lo := 3; lo <= 8; lo++ {
		layers[1] = lo * 16
		for lt := 2; lt <= 4; lt++ {
			layers[2] = lt * 16
			for lr := 3; lr <= 4; lr++ {
				learning_rate = float64(lr) * 0.2
				for bs := 1; bs <= 2; bs++ {
					batch_size = 32 * bs
					acc, err := trainingLoop(layers, learning_rate, epochs, batch_size)
					if err != nil {
						fmt.Println("Training error:", err)
						continue
					}
					row := []string{
						strconv.FormatFloat(acc, 'f', 2, 64),
						strconv.Itoa(layers[1]),
						strconv.Itoa(layers[2]),
						strconv.FormatFloat(learning_rate, 'f', 1, 64),
						strconv.Itoa(batch_size),
					}
					err = writer.Write(row)
					if err != nil {
						fmt.Println(err)
					}
					writer.Flush()

				}
			}

		}
	}
	fmt.Println("Data saved to output.csv")
}

func predictFromFile(filename string, network nn.Network) (int, error) {

	path := filepath.Join("custom_pics", filename)
	fmt.Println(path)
	// script_path := filepath.Join(exe_dir, "custom_pics", "preprocess.py")

	cmd := exec.Command("python3", "custom_pics/preprocess.py", path)
	output, err := cmd.CombinedOutput()
	if err != nil {
		fmt.Printf("Command failed with error: %v\n", err)
		fmt.Printf("Output: %s\n", string(output))
	}

	var input []float64
	json.Unmarshal(output, &input)

	predictions, _, _, err := network.Forward(input)
	if err != nil {
		fmt.Println(err)
	}
	prediction := argmax(predictions)
	return prediction, nil
}
