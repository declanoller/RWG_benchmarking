import numpy as np

class FFNN_multilayer:
    """Feedforward neural network, multi layer

    - weights are thus order the same way in each row of the weights matrix
    - rows in the weight matrix correspond to weights of connections entering a same neuron
    - cols in the weight matrix correspond to connections from the same input
    """

    def __init__(self, N_inputs, N_outputs, **kwargs):

        act_fn = kwargs.get('act_fn', 'tanh')
        output_fn = kwargs.get('output_fn', 'argmax')
        N_hidden_units = kwargs.get('N_hidden_units', 2*N_inputs)
        N_hidden_layers = kwargs.get('N_hidden_layers', 1)


        self.N_inputs = N_inputs
        self.N_outputs = N_outputs
        self.N_hidden_layers = N_hidden_layers
        self.N_hidden_units = N_hidden_units

        self.init_weights()
        self.print_NN_matrices()

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
        assert output_fn in output_fn_d, 'Must supply valid output function name'
        self.output_fn = output_fn_d[output_fn]


    def reset_state(self):
        pass

    def print_NN_matrices(self):

        print('N_inputs: ', self.N_inputs)

        for i,w in enumerate(self.weights_matrix):
            print(f'W_{i} shape: {w.shape}')

        print('N_outputs: ', self.N_outputs)

        print('\nWeight matrices:')
        for i,w in enumerate(self.weights_matrix):
            print(f'\nW_{i}:')
            print(w)



    def softmax_choice(self, x):
        x_softmax = np.exp(x)/sum(np.exp(x))
        return np.random.choice(len(x), p=x_softmax)

    def init_weights(self):

        self.weights_matrix = []

        mat_input_size = self.N_inputs + 1

        '''
        Here, we're using the convention of doing W*x, if W is the weight matrix
        and x is vector. So W has dimensions [N_outputs x N_inputs] and x has
        dimensions [N_inputs x 1]. So W*x has dimensions [N_outputs x 1].

        So it composes right to left, like W_2*tanh(W_1*x).

        Note also that we insert a bias unit in each layer.
        '''
        for i in range(self.N_hidden_layers):

            mat_output_size = self.N_hidden_units
            self.weights_matrix.append(np.random.randn(mat_output_size, mat_input_size))
            mat_input_size = mat_output_size + 1

        # And for the last layer:
        self.weights_matrix.append(np.random.randn(self.N_outputs, mat_input_size))



    def set_weights(self, weights):

        self.weights_matrix = weights


    def set_random_weights(self):
        self.init_weights()

    def activate(self, inputs):
        """Activate the neural network

        """

        x = inputs

        for i,w in enumerate(self.weights_matrix):
            x = np.concatenate((x, [1.0]))
            x = np.dot(w, x)
            x = self.act_fn(x)
            #if i < len(self.weights_matrix)-1:

        return x

    def get_action(self, inputs):
        action_vec = self.activate(inputs)
        return self.output_fn(action_vec)
