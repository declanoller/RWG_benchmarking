from FFNN1L import FFNN1L
from RNN1L import RNN1L
import numpy as np

'''
A wrapper class for whichever NN you want to use. Handles
issues like the right output function, scaling, etc, to use for
different action spaces.
'''


class Agent:

    def __init__(self, env, **kwargs):

        self.ninputs = env.reset().size

        # Figure out correct output function (i.e., argmax or identity)
        # and output scaling, given the env type.
        if type(env.action_space).__name__ == 'Discrete':
            self.action_space_type = 'discrete'
            output_fn = 'argmax'
            self.nactions = env.action_space.n
        elif type(env.action_space).__name__ == 'Box':
            self.action_space_type = 'continuous'
            output_fn = 'continuous'
            self.nactions = len(env.action_space.sample())

            # This is because right now, RNN1L outputs tanh, which goes from -1 to 1,
            # but some env's (Pendulum-v0, for example) have an action space of -2, 2.
            # So I figure out the max action value a continuous action space could need
            # and just scale the actions by that. There's certainly a better way that
            # should be implemented.
            self.action_scale = env.action_space.high.max()

        # Select the NN class to use.
        NN_types_dict = {
            'RNN' : RNN1L,
            'FFNN' : FFNN1L
        }

        NN_type = kwargs.get('NN', 'RNN')
        assert NN_type in NN_types_dict.keys(), 'Must supply a valid NN type!'
        self.NN = NN_types_dict[NN_type](self.ninputs, self.nactions, output_fn=output_fn)

        self.reset()



    def reset(self):
        self.NN.set_random_weights()


    def init_episode(self):
        self.NN.reset_state()


    def get_action(self, state):
        a = self.NN.get_action(state)
        if self.action_space_type == 'continuous':
            a *= self.action_scale
        return a


    def get_weight_matrix(self):
        return self.NN.weights_matrix

    def set_weight_matrix(self, w):
        self.NN.set_weights(w)



    def mutate_gaussian_noise(self, sd=0.1):
        noise = sd*np.random.randn(self.NN.nweights())
        new_weights = self.get_weight_matrix() + noise
        self.set_weight_matrix(new_weights)




#
