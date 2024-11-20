package main

import (
	"fmt"
	"math"
)

type Point struct {
	input    []float64
	expected []float64
}

type Training_Set []Point

type Layer struct {
	incoming_nodes, layer_nodes int

	weights [][]float64
	biases  []float64

	weight_cost_gradient [][]float64
	biases_cost_gradient []float64
}

type Neural_Network struct {
	layers []Layer
}

func activation_function(x float64) float64 {
	// sigmoid function
	return 1 / (1 + math.Exp(-x))
}

func activation_function_derivative(x float64) float64 {
	activation := activation_function(x)
	return activation * (1 - activation)
}

func calculate_outputs(layer Layer, inputs []float64) (outputs []float64) {

	for layer_node_i := range layer.layer_nodes {
		sum := layer.biases[layer_node_i]

		for incoming_node := range layer.incoming_nodes {
			sum += inputs[incoming_node] * layer.weights[incoming_node][layer_node_i]
		}

		outputs = append(outputs, activation_function(sum))
	}
	return
}

func create_neural_network(layer_sizes []int) (nn Neural_Network) {
	for i := range layer_sizes {
		var hidden_layer Layer
		hidden_layer.incoming_nodes = layer_sizes[i]
		hidden_layer.layer_nodes = layer_sizes[i+1]

		nn.layers = append(nn.layers, hidden_layer)
	}
	return
}

func forward_propogate(nn Neural_Network, inputs []float64) (outputs []float64) {
	outputs = inputs
	for _, layer := range nn.layers {
		outputs = calculate_outputs(layer, outputs)
	}
	return
}

func apply_gradient_descent(layer Layer, learn_rate float64) {
	for node_i := range layer.layer_nodes {
		layer.biases[node_i] -= layer.biases_cost_gradient[node_i] * learn_rate
		for incoming_node := range layer.incoming_nodes {
			layer.weights[incoming_node][node_i] -= layer.weight_cost_gradient[incoming_node][node_i] * learn_rate
		}
	}
}

func loss(actual float64, expected float64) float64 {
	error := actual - expected
	return error * error
}

func loss_derivative(actual float64, expected float64) float64 {
	return 2 * (actual - expected)
}

func network_loss(nn Neural_Network, training_set Training_Set) float64 {
	var cost float64

	for i := range len(training_set) {
		output := forward_propogate(nn, training_set[i].input)
		for j := range len(output) {
			cost += loss(output[j], training_set[i].expected[j])
		}
	}

	return cost / float64(len(training_set))
}

func main() {
	fmt.Println("Hello world!")
}
