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
