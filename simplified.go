package main

import (
	"fmt"
	"math"
	"math/rand"
)

func sigmoid_func(x float64) float64 {
	return (1 / (1 + math.Exp(-x)))
}

// Used chain rule, derivative of sigmoid with respect to inner
// times derivative of inner with respect to weight
func derivative(actual float64) float64 {
	return actual * (1 - actual)
}

type Neuron struct {
	x1 int
	x2 int

	w1 float64
	w2 float64

	b float64
}

func (neuron Neuron) fire() float64 {
	x1 := float64(neuron.x1)
	x2 := float64(neuron.x2)
	return sigmoid_func(neuron.w1*x1 + neuron.w2*x2 + neuron.b)
}

func main() {
	var neuron = Neuron{
		x1: 0,
		x2: 0,
		w1: rand.Float64(),
		w2: rand.Float64(),
		b:  rand.Float64(),
	}

	data := [...][]int{
		{0, 0, 0},
		{0, 1, 0},
		{1, 0, 0},
		{1, 1, 1},
	}
	rate := 0.3

	for range 1000 {
		for i := range 4 {
			neuron.x1 = data[i][0]
			neuron.x2 = data[i][1]
			actual := neuron.fire()
			expected := float64(data[i][2])

			output_delta := (actual - expected) * derivative(actual)

			w1_grad := output_delta * float64(neuron.x1)
			w2_grad := output_delta * float64(neuron.x2)

			bias_grad := output_delta

			neuron.w1 -= rate * w1_grad
			neuron.w2 -= rate * w2_grad
			neuron.b -= rate * bias_grad
		}
	}

	for i := range 4 {
		neuron.x1 = data[i][0]
		neuron.x2 = data[i][1]

		expected := float64(data[i][2])
		actual := neuron.fire()

		cost := expected - actual

		fmt.Printf("(%d|%d) => %f [%f]\n", neuron.x1, neuron.x2, actual, cost)
	}

}
