package main

import (
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

type Layer_Data struct {
	inputs           []float64
	weighted_inputs  []float64
	activated_inputs []float64
	layer_node_value []float64
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

func loss(actual float64, expected float64) float64 {
	error := actual - expected
	return error * error
}

func loss_derivative(actual float64, expected float64) float64 {
	return 2 * (actual - expected)
}

func (layer Layer) calculate_outputs(inputs []float64) (outputs []float64) {
	for layer_node_i := range layer.layer_nodes {
		sum := layer.biases[layer_node_i]

		for incoming_node := range layer.incoming_nodes - 1 {
			sum += inputs[incoming_node] * layer.weights[incoming_node][layer_node_i]
		}
		outputs = append(outputs, activation_function(sum))
	}
	return
}

func (layer Layer) calculate_outputs_and_store_info(inputs []float64, learn_store Layer_Data) (outputs []float64) {

	copy(learn_store.inputs, inputs)

	for layer_node_i := range layer.layer_nodes {
		sum := layer.biases[layer_node_i]

		for incoming_node := range layer.incoming_nodes {
			sum += inputs[incoming_node] * layer.weights[incoming_node][layer_node_i]
		}

		learn_store.weighted_inputs[layer_node_i] = sum
		outputs = append(outputs, activation_function(sum))
	}

	copy(learn_store.activated_inputs, outputs)

	return
}

func create_neural_network(layer_sizes []int) (nn Neural_Network) {
	for i := range len(layer_sizes) - 2 {
		var hidden_layer Layer
		hidden_layer.incoming_nodes = layer_sizes[i]
		hidden_layer.layer_nodes = layer_sizes[i+1]

		hidden_layer.weights = make([][]float64, hidden_layer.incoming_nodes)
		for j := range hidden_layer.incoming_nodes {
			hidden_layer.weights[j] = make([]float64, hidden_layer.layer_nodes)
		}

		hidden_layer.biases = make([]float64, hidden_layer.layer_nodes)

		hidden_layer.weight_cost_gradient = make([][]float64, hidden_layer.incoming_nodes)
		for j := range hidden_layer.incoming_nodes {
			hidden_layer.weight_cost_gradient[j] = make([]float64, hidden_layer.layer_nodes)
		}

		hidden_layer.biases_cost_gradient = make([]float64, hidden_layer.layer_nodes)

		nn.layers = append(nn.layers, hidden_layer)
	}

	output_layer := Layer{
		incoming_nodes: layer_sizes[len(layer_sizes)-2],
		layer_nodes:    layer_sizes[len(layer_sizes)-1],

		weights: make([][]float64, layer_sizes[len(layer_sizes)-2]),
		biases:  make([]float64, layer_sizes[len(layer_sizes)-1]),

		weight_cost_gradient: make([][]float64, layer_sizes[len(layer_sizes)-2]),
		biases_cost_gradient: make([]float64, layer_sizes[len(layer_sizes)-1]),
	}

	for i := range output_layer.incoming_nodes {
		output_layer.weights[i] = make([]float64, output_layer.layer_nodes)
	}

	for i := range output_layer.incoming_nodes {
		output_layer.weight_cost_gradient[i] = make([]float64, output_layer.layer_nodes)
	}

	nn.layers = append(nn.layers, output_layer)
	return
}

func (nn Neural_Network) forward_propogate(inputs []float64) (outputs []float64) {
	outputs = inputs
	for _, layer := range nn.layers {
		outputs = layer.calculate_outputs(outputs)
	}
	return
}

func (layer Layer) apply_gradient(learn_rate float64) {
	for node_i := range layer.layer_nodes {
		layer.biases[node_i] -= layer.biases_cost_gradient[node_i] * learn_rate
		for incoming_node := range layer.incoming_nodes {
			layer.weights[incoming_node][node_i] -= layer.weight_cost_gradient[incoming_node][node_i] * learn_rate
		}
	}
}

func calculate_partial_derivative_of_output_nodes(layer_data Layer_Data, expected_output []float64) {
	for i := range layer_data.layer_node_value {
		cost_d := loss_derivative(layer_data.activated_inputs[i], expected_output[i])
		activation_d := activation_function_derivative(layer_data.weighted_inputs[i])
		layer_data.layer_node_value[i] = cost_d * activation_d
	}
}

func (layer Layer) calculate_partial_derivative_of_hidden_nodes(layer_data Layer_Data, next_layer Layer, next_node_values []float64) {
	for node_i := range layer.layer_nodes {
		var node_val float64 = 0
		for next_node_i := range next_layer.layer_nodes {
			weighted_input_d := next_layer.weights[node_i][next_node_i]
			node_val += weighted_input_d * next_node_values[next_node_i]
		}
		node_val *= activation_function_derivative(layer_data.weighted_inputs[node_i])
		layer_data.layer_node_value[node_i] = node_val
	}
}

func (layer Layer) update_gradient(layer_data Layer_Data) {
	for node_out := range layer.layer_nodes {
		node_value := layer_data.layer_node_value[node_out]
		for node_in := range layer.incoming_nodes {
			d_cost_weight := layer_data.inputs[node_in] * node_value // chain rule
			layer.weight_cost_gradient[node_in][node_out] += d_cost_weight
		}
	}

	for node_out := range layer.layer_nodes {
		d_cost_bias := layer_data.layer_node_value[node_out]
		layer.biases_cost_gradient[node_out] += d_cost_bias
	}
}

func (nn Neural_Network) network_loss(training_set Training_Set) float64 {
	var cost float64

	for i := range len(training_set) {
		output := nn.forward_propogate(training_set[i].input)
		for j := range len(output) {
			cost += loss(output[j], training_set[i].expected[j])
		}
	}

	return cost / float64(len(training_set))
}

// func main() {
// 	layer_szs := [...]int{2, 2}
// 	var nn Neural_Network = create_neural_network(layer_szs[:])

// 	training_set := Training_Set{
// 		Point{
// 			input:    []float64{0, 0},
// 			expected: []float64{0, 0},
// 		},
// 		Point{
// 			input:    []float64{0, 1},
// 			expected: []float64{0, 1},
// 		},
// 		Point{
// 			input:    []float64{1, 0},
// 			expected: []float64{1, 0},
// 		},
// 		Point{
// 			input:    []float64{1, 1},
// 			expected: []float64{1, 1},
// 		},
// 	}

// 	layer_data := []Layer_Data{
// 		{
// 			inputs:           make([]float64, nn.layers[0].incoming_nodes),
// 			weighted_inputs:  make([]float64, nn.layers[0].layer_nodes),
// 			activated_inputs: make([]float64, nn.layers[0].layer_nodes),
// 			layer_node_value: make([]float64, nn.layers[0].layer_nodes),
// 		},
// 	}

// 	for layer_i, layer := range nn.layers {

// 		for i := range layer.incoming_nodes {
// 			for j := range layer.layer_nodes {
// 				nn.layers[layer_i].weights[i][j] = rand.Float64()
// 			}
// 		}

// 		for i := range layer.layer_nodes {
// 			nn.layers[layer_i].biases[i] = rand.Float64()
// 		}
// 	}

// 	nn.layers[0].calculate_outputs_and_store_info(training_set[0].input, layer_data[0])
// 	calculate_partial_derivative_of_output_nodes(layer_data[0], training_set[0].expected)
// 	for range 100 {
// 		nn.layers[0].calculate_partial_derivative_of_hidden_nodes(layer_data[0], nn.layers[0], layer_data[0].layer_node_value)
// 		nn.layers[0].update_gradient(layer_data[0])
// 		nn.layers[0].apply_gradient(0.2)
// 		fmt.Println("Cost: ", nn.network_loss(training_set))
// 	}
// }
