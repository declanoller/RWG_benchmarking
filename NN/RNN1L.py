import numpy as np

class RNN1L:
    """Recurrent neural network, single layer

    - `self.state` represents: last inputs + fixed 1 (bias) + last activation
    - weights are thus order the same way in each row of the weights matrix
    - rows in the weight matrix correspond to weights of connections entering a same neuron
    - cols in the weight matrix correspond to connections from the same input
    """

    def __init__(self, ninputs, nneurs, act_fn='tanh', init_weights=None, output_fn='argmax'):
        self.ninputs = ninputs
        self.nneurs = nneurs
        self.state_size = self.ninputs + 1 + self.nneurs
        self.reset_state()
        self.weights_matrix = self.init_weights()
        if init_weights: self.set_weights(init_weights)
        # state index accessors
        self.input_idxs = range(0, ninputs)
        self.bias_idxs = range(ninputs + 1, ninputs + 1)
        self.act_idxs = range(ninputs + 1, ninputs + 1 + nneurs)

        activation_fn_d = {
                    'tanh' : np.tanh,
                    'linear' : lambda x: x,
                    'relu' : lambda x: np.maximum(0, x)
        }
        assert act_fn in activation_fn_d.keys(), 'Must supply valid activation function name!'
        self.act_fn = activation_fn_d[act_fn] # make sure it applies element-wise to a np array

        output_fn_d = {
                    'argmax' : np.argmax,
                    'continuous' : lambda x: x,
                    'softmax' : self.softmax_choice
                    }
        assert output_fn in output_fn_d.keys(), 'Must supply valid output function name'
        self.output_fn = output_fn_d[output_fn]


    def softmax_choice(self, x):
        x_softmax = np.exp(x)/sum(np.exp(x))
        return np.choice(len(x), p=x_softmax)


    def init_weights(self):
        return np.random.randn(self.nneurs, self.state_size)

    def set_weights(self, weights):
        assert weights.size == self.weights_matrix.size, "Wrong number of weights"
        self.weights_matrix = weights.reshape(self.weights_matrix.shape)
        self.reset_state()

    def set_random_weights(self):
        self.set_weights(np.random.randn(self.nweights()))

    def reset_state(self):
        self.state = np.zeros(self.state_size)
        self.state[self.ninputs] = 1 # bias -- should never be changed!

    def activate(self, inputs):
        """Activate the neural network

        - Overwrite the new inputs in the initial part of the state
        - Execute dot product with weight matrix
        - Pass result to activation function
        """
        self.state[self.input_idxs] = inputs
        net = np.dot(self.weights_matrix, self.state)
        self.state[self.act_idxs] = self.act_fn(net)
        return self.get_act()

    def last_input(self):
        return self.state[self.input_idxs]

    def get_act(self):
        return self.state[self.act_idxs]

    def get_action(self, inputs):
        action_vec = self.activate(inputs)
        return self.output_fn(action_vec)

    def nweights(self):
        return self.weights_matrix.size

    def nweights_per_neur(self):
        return self.weights_matrix.shape[1]
