#! /usr/bin/python
from numpy import *

class NeuronLayer():
	def __init__(self, number_of_neuron, number_of_inputs_per_neuron):
		self.synaptic_weights = 2*random.random((number_of_inputs_per_neuron, number_of_neuron))-1

class NeuralNetwork():
	def __init__(self, layer1,layer2, layer3):
		self.layer1 =layer1
		self.layer2 =layer2
		self.layer3 =layer3

	def _Relu_(self,x):
		return x * (x>0)

	def _Relu_derivative(self,x):
		return 1.*(x>0)

	def _train_(self, training_inputs, training_outputs, number_of_iterations):
		for iteration in xrange(number_of_iterations):
			output_layer1, output_layer2, output_layer3 = self.learn(training_inputs)
			
			layer3_error = training_outputs-output_layer3
			layer3_delta = layer3_error*self._Relu_derivative(output_layer3)

			layer2_error = layer3_delta.dot(self.layer3.synaptic_weights.T)
			layer2_delta = layer2_error*self._Relu_derivative(output_layer2)

			layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
			layer1_delta = layer1_error*self._Relu_derivative(output_layer1)

			layer1_adjustment = training_inputs.T.dot(layer1_delta)
			layer2_adjustment = output_layer1.T.dot(layer2_delta)
			layer3_adjustment = output_layer2.T.dot(layer3_delta)

			self.layer1.synaptic_weights +=layer1_adjustment
			self.layer2.synaptic_weights +=layer2_adjustment
			self.layer3.synaptic_weights +=layer3_adjustment

	def learn(self, inputs):
		output_layer1 = self._Relu_(dot(inputs, self.layer1.synaptic_weights))
		output_layer2 = self._Relu_(dot(output_layer1, self.layer2.synaptic_weights))
		output_layer3 = self._Relu_(dot(output_layer2, self.layer3.synaptic_weights))
		return output_layer1, output_layer2, output_layer3

	def print_weights(self):
		print "layer1 weights"
		print self.layer1.synaptic_weights

		print "layer2 weights"
		print self.layer2.synaptic_weights

		print "layer3 weights"
		print self.layer3.synaptic_weights

if __name__ == "main":
	
	random.seed(1)

	layer1 = NeuronLayer(4,3)
	layer2 = NeuronLayer(4,4)
	layer3 = NeuronLayer(1,4)

	nn = NeuralNetwork(layer1, layer2, layer3)
	nn.print_weights()

	training_inputs = array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
	training_outputs =array([[0, 1, 1, 1, 1, 0, 0]]).T

	nn._train_(training_inputs,training_outputs,1000)

	print "New Synaptic weights"
	print nn.print_weights()

	output = nn.learn(array([1,1,0]))
	print output
