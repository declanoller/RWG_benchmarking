import path_utils
import Statistics

'''
For getting statistics with various combos of parameters.

'''


Statistics.run_vary_params(
    {
        'NN' : 'FFNN_multilayer'
    },
    {
        'env_name' : ['CartPole-v0', 'MountainCar-v0', 'MountainCarContinuous-v0', 'Pendulum-v0'],
        'N_hidden_layers' : [0, 1],
        'N_hidden_units' : [2, 4, 8]
    },
    N_gen=3000,
    N_trials=20
)

exit()







#['CartPole-v0', 'MountainCar-v0', 'MountainCarContinuous-v0', 'Pendulum-v0']
Statistics.run_multi_envs(
                            ['CartPole-v0', 'MountainCar-v0', 'MountainCarContinuous-v0', 'Pendulum-v0'],
                            N_gen=10000,
                            N_trials=10,
                            NN='FFNN_multilayer',
                            N_hidden_layers=0,
                            N_hidden_units=0,
                            act_fn='linear'
                            )


exit()








Statistics.benchmark_envs(['CartPole-v0'], N_dist=5, N_gen=100)
