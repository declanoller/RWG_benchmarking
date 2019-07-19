import path_utils
import Benchmark

'''
For testing various benchmarking examples.

'''

Benchmark.benchmark_vary_params(
    {
        'env_name' : 'CartPole-v0'
    },
    {
        'act_fn' : ['tanh', 'relu'],
        'NN' : ['FFNN', 'RNN']
    }
)

exit()

Benchmark.benchmark_envs(['CartPole-v0'], N_dist=5, N_gen=100)


#Benchmark.benchmark_classic_control_envs(N_gen=2000, N_dist=5)
#Benchmark.benchmark_envs(['CartPole-v0', 'LunarLander-v2'], N_dist=5, N_gen=20)
#Benchmark.benchmark_envs(['MountainCar-v0'], N_dist=50, N_gen=2000)
