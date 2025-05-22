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
			nextActivation[i] = ReLu(sum)
		}

		zs = append(zs, z)
		activations = append(activations, nextActivation)
		activation = nextActivation
	}

	return activation, zs, activations, nil
}

func (net *Network) Backward(input, target []float64) (gradW [][][]float64, gradB [][]float64, err error) {
	activations := [][]float64{input}
	zs := [][]float64{}
	output, zs, activations, err := net.Forward(input)
	if err != nil {
		return nil, nil, err
	}

	// looping through net to get lengths of slices for gradients
	for l := 0; l < len(net.Layers)-1; l++ {
		current := net.Layers[l]
		next := net.Layers[l+1]

		gradW[l] = make([][]float64, next)
		for i := 0; i < next; i++ {
			gradW[l][i] = make([]float64, current)
		}

		gradB[l] = make([]float64, next)
	}

	last := len(zs) - 1
	delta := make([]float64, len(output))
	// calculating the delta slice values
	for i := range delta {
		delta[i] = (output[i] - target[i]) * ReLUPrime(zs[last][i])
	}
	//skonczylo sie na tym ze delta chyba zle sie liczy bo powinna miec 2 wymiary,
	//potem po liczeniu delt trzeba dodac gradB i gradW, a potem zwrocic i jest g
	return gradW, gradB, nil
}
