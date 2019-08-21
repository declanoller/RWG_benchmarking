import path_utils
import Statistics

'''
For getting statistics with various combos of parameters.

'''


############################### Basic runs
Statistics.run_vary_params(
    {
        'NN' : 'FFNN'
    },
    {
        'env_name' : ['CartPole-v0', 'Pendulum-v0'],
        'N_hidden_layers' : [0, 1],
        'N_hidden_units' : [2, 4]
    },
    N_gen=200,
    N_trials=5
)
