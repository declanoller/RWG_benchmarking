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
        'env_name' : ['CartPole-v0', 'Acrobot-v1', 'MountainCar-v0', 'MountainCarContinuous-v0', 'Pendulum-v0'],
        'N_hidden_layers' : [0, 1],
        'N_hidden_units' : [2, 4, 8]
    },
    N_gen=20000,
    N_trials=20
)


################################# Two layer, 4 units each
Statistics.run_vary_params(
    {
        'NN' : 'FFNN',
        'N_hidden_layers' : 2,
        'N_hidden_units' : 4
    },
    {
        'env_name' : ['CartPole-v0', 'Acrobot-v1', 'MountainCar-v0', 'MountainCarContinuous-v0', 'Pendulum-v0'],

    },
    N_gen=20000,
    N_trials=20
)


################################# No hidden layer, linear activation
Statistics.run_vary_params(
    {
        'NN' : 'FFNN',
        'N_hidden_layers' : 0,
        'act_fn' : 'linear'
    },
    {
        'env_name' : ['CartPole-v0', 'Acrobot-v1', 'MountainCar-v0', 'MountainCarContinuous-v0', 'Pendulum-v0'],

    },
    N_gen=20000,
    N_trials=20
)



################################# No hidden layer, RNN
Statistics.run_vary_params(
    {
        'NN' : 'RNN',
        'N_hidden_layers' : 0
    },
    {
        'env_name' : ['CartPole-v0', 'Acrobot-v1', 'MountainCar-v0', 'MountainCarContinuous-v0', 'Pendulum-v0'],

    },
    N_gen=20000,
    N_trials=20
)
