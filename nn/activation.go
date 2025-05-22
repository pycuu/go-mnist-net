package nn

import "math"

func ReLu(x float64) float64 {
	return max(0, x)
}

func ReLUPrime(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}

func Sigmoid(x float64) float64 {
	return (1 / (1 + math.Pow(math.E, (-x))))
}

func SigmoidPrime(x float64) float64 {
	return (Sigmoid(x) * (1 - Sigmoid(x)))
}
