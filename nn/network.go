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

func (net *Network) Forward(input []float64) ([]float64, error) {
	if len(input) != net.Layers[0] {
		return nil, errors.New("input length must match size of input layer")
	}

	activation := input

	for layer := 0; layer < len(net.Layers)-1; layer++ {
		current_size := net.Layers[layer]
		next_size := net.Layers[layer+1]

		next := make([]float64, next_size)

		for i := 0; i < next_size; i++ {
			sum := 0.0
			for j := 0; j < current_size; j++ {
				weight_index := i*current_size + j
				sum += net.Weights[layer][weight_index] * activation[j]
			}
			sum += net.Biases[layer][i]
			next[i] = ReLu(sum)
		}

		activation = next
	}

	return activation, nil
}
