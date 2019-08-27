import path_utils
import Statistics
import ray, psutil, time

ray.init(num_cpus=psutil.cpu_count(), ignore_reinit_error=True, include_webui=False)
print('\nInitializing ray... Waiting for workers before starting...')
time.sleep(2.0)
print('Starting!\n')


'''
For getting statistics with various combos of parameters.
'''

N_GEN = 10000
N_TRIALS = 20

############################### Basic runs
Statistics.run_vary_params(
    {
        'NN' : 'FFNN',
        'random_dist' : 'uniform'
    },
    {
        'env_name' : ['CartPole-v0', 'Acrobot-v1', 'MountainCar-v0', 'MountainCarContinuous-v0', 'Pendulum-v0'],
        'N_hidden_layers' : [0, 1],
        'N_hidden_units' : [2, 4, 8]
    },
    N_gen=N_GEN,
    N_trials=N_TRIALS
)


################################# Two layer, 4 units each
Statistics.run_vary_params(
    {
        'NN' : 'FFNN',
        'N_hidden_layers' : 2,
        'N_hidden_units' : 4,
        'random_dist' : 'uniform'
    },
    {
        'env_name' : ['CartPole-v0', 'Acrobot-v1', 'MountainCar-v0', 'MountainCarContinuous-v0', 'Pendulum-v0'],

    },
    N_gen=N_GEN,
    N_trials=N_TRIALS
)


################################# No hidden layer, RNN
Statistics.run_vary_params(
    {
        'NN' : 'RNN',
        'N_hidden_layers' : 0,
        'random_dist' : 'uniform'
    },
    {
        'env_name' : ['CartPole-v0', 'Acrobot-v1', 'MountainCar-v0', 'MountainCarContinuous-v0', 'Pendulum-v0'],

    },
    N_gen=N_GEN,
    N_trials=N_TRIALS
)


################################# 1 hidden layer, 4 HU, RNN
Statistics.run_vary_params(
    {
        'NN' : 'RNN',
        'N_hidden_layers' : 1,
        'N_hidden_units' : 4,
        'random_dist' : 'uniform'
    },
    {
        'env_name' : ['CartPole-v0', 'Acrobot-v1', 'MountainCar-v0', 'MountainCarContinuous-v0', 'Pendulum-v0'],

    },
    N_gen=N_GEN,
    N_trials=N_TRIALS
)
