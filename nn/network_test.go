package nn

import (
	"math"
	"testing"
)

func almostEqual(a, b, epsilon float64) bool {
	return math.Abs(a-b) < epsilon
}

func TestNewNetwork(t *testing.T) {
	layers := []int{2, 3, 1}
	net, err := NewNetwork(layers)
	if err != nil {
		t.Fatalf("Failed to create network: %v", err)
	}
	if len(net.Weights) != len(layers)-1 {
		t.Errorf("Expected %d weight layers, got %d", len(layers)-1, len(net.Weights))
	}
}

func TestForward(t *testing.T) {
	net, _ := NewNetwork([]int{2, 3, 1})
	input := []float64{0.5, -1.2}

	output, zs, activations, err := net.Forward(input)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}
	if len(output) != 1 {
		t.Errorf("Expected output of size 1, got %d", len(output))
	}
	if len(zs) != 2 || len(activations) != 3 {
		t.Errorf("Incorrect number of layers in zs or activations")
	}
}

func TestBackward(t *testing.T) {
	net, _ := NewNetwork([]int{2, 3, 1})
	input := []float64{0.5, -1.2}
	target := []float64{0.0}

	gradW, gradB, err := net.Backward(input, target)
	if err != nil {
		t.Fatalf("Backward failed: %v", err)
	}
	if len(gradW) != len(net.Weights) || len(gradB) != len(net.Biases) {
		t.Errorf("Gradient size mismatch")
	}
}

func TestUpdateWeights(t *testing.T) {
	net, _ := NewNetwork([]int{2, 3, 1})
	input := []float64{0.5, -1.2}
	target := []float64{0.0}

	// save original weights
	origW := make([][]float64, len(net.Weights))
	origB := make([][]float64, len(net.Biases))
	for l := range net.Weights {
		origW[l] = make([]float64, len(net.Weights[l]))
		copy(origW[l], net.Weights[l])

		origB[l] = make([]float64, len(net.Biases[l]))
		copy(origB[l], net.Biases[l])
	}

	gradW, gradB, err := net.Backward(input, target)
	if err != nil {
		t.Fatalf("Backward failed before update: %v", err)
	}

	net.UpdateWeights(gradW, gradB, 0.01)

	// compare weights to ensure they have changed
	changed := false
	for l := range net.Weights {
		for i := range net.Weights[l] {
			if !almostEqual(origW[l][i], net.Weights[l][i], 1e-6) {
				changed = true
				break
			}
		}
	}
	if !changed {
		t.Errorf("Weights did not change after update")
	}
}
