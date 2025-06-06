package nn

import (
	"errors"
	"math/rand"
	"time"
)

type Network struct {
	Layers  []int
	Weights [][]float64
	Biases  [][]float64
}

func NewNetwork(layers []int) (*Network, error) {
	if len(layers) < 2 {
		return nil, errors.New("network must have at least 2 layers")
	}

	r := rand.New(rand.NewSource(time.Now().UnixNano()))

	weights := make([][]float64, len(layers)-1)
	biases := make([][]float64, len(layers)-1)

	for i := 0; i < len(layers)-1; i++ {
		inputs := layers[i]
		outputs := layers[i+1]

		weights[i] = make([]float64, outputs*inputs)
		for j := range weights[i] {
			weights[i][j] = r.NormFloat64()
		}

		biases[i] = make([]float64, outputs)
		for j := range biases[i] {
			biases[i][j] = r.NormFloat64()
		}
	}

	return &Network{
		Layers:  layers,
		Weights: weights,
		Biases:  biases,
	}, nil
}

func (net *Network) Forward(input []float64) (output []float64, zs [][]float64, activations [][]float64, err error) {
	if len(input) != net.Layers[0] {
		return nil, nil, nil, errors.New("input length must match size of input layer")
	}

	activation := input
	activations = append(activations, activation)

	for layer := 0; layer < len(net.Layers)-1; layer++ {
		currentSize := net.Layers[layer]
		nextSize := net.Layers[layer+1]

		z := make([]float64, nextSize)
		nextActivation := make([]float64, nextSize)

		for i := 0; i < nextSize; i++ {
			sum := 0.0
			for j := 0; j < currentSize; j++ {
				weightIndex := i*currentSize + j
				sum += net.Weights[layer][weightIndex] * activation[j]
			}
			sum += net.Biases[layer][i]
			z[i] = sum
			nextActivation[i] = Sigmoid(sum)
		}

		zs = append(zs, z)
		activations = append(activations, nextActivation)
		activation = nextActivation
	}

	return activation, zs, activations, nil
}

func (net *Network) Backward(input, target []float64) (gradW [][][]float64, gradB [][]float64, err error) {
	output, zs, activations, err := net.Forward(input)
	if err != nil {
		return nil, nil, err
	}

	// allocating slices
	gradW = make([][][]float64, len(net.Layers)-1)
	gradB = make([][]float64, len(net.Layers)-1)

	// init for each layer
	for l := 0; l < len(net.Layers)-1; l++ {
		current := net.Layers[l]
		next := net.Layers[l+1]

		gradW[l] = make([][]float64, next)
		for i := 0; i < next; i++ {
			gradW[l][i] = make([]float64, current)
		}
		gradB[l] = make([]float64, next)
	}

	// calculating delta for last layer
	deltas := make([][]float64, len(net.Layers)-1)
	last_layer_index := len(zs) - 1

	deltas[last_layer_index] = make([]float64, len(output))
	for i := range output {
		deltas[last_layer_index][i] = (output[i] - target[i]) * SigmoidPrime(zs[last_layer_index][i])
	}

	// deltas for each neuron from other layers
	for l := len(net.Layers) - 2; l > 0; l-- {
		layerSize := net.Layers[l]
		nextSize := net.Layers[l+1]

		deltas[l-1] = make([]float64, layerSize)

		for i := 0; i < layerSize; i++ {
			sum := 0.0
			for j := 0; j < nextSize; j++ {
				weight := net.Weights[l][j*layerSize+i]
				sum += weight * deltas[l][j]
			}
			deltas[l-1][i] = sum * SigmoidPrime(zs[l-1][i])
		}
	}

	// computing gradients for each bias and weight
	// gradW[l][i][j] - gradient of weight from [j][l] to [i][l+1]
	for l := 0; l < len(deltas); l++ {
		for i := range gradB[l] {
			gradB[l][i] = deltas[l][i]
			for j := range gradW[l][i] {
				gradW[l][i][j] = deltas[l][i] * activations[l][j]
			}
		}
	}

	return gradW, gradB, nil
}

func (net *Network) UpdateWeights(gradW [][][]float64, gradB [][]float64, learningRate float64) {

	for l := 0; l < len(net.Weights); l++ {
		inputs := net.Layers[l]
		outputs := net.Layers[l+1]

		for i := 0; i < outputs; i++ {
			net.Biases[l][i] -= learningRate * gradB[l][i]

			for j := 0; j < inputs; j++ {
				index := i*inputs + j
				//fmt.Printf("Before: %f\n", net.Weights[l][index])
				net.Weights[l][index] -= learningRate * gradW[l][i][j]
				//fmt.Printf("After: %f\n", net.Weights[l][index])
			}
		}
	}
}
