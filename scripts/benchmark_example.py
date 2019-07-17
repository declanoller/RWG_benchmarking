import path_utils
import Benchmark

'''
For testing various benchmarking examples.

'''

#Benchmark.benchmark_envs(['CartPole-v0'], N_dist=100, N_gen=1000)
#Benchmark.benchmark_classic_control_envs(N_gen=2000)
#Benchmark.benchmark_envs(['CartPole-v0', 'LunarLander-v2'], N_dist=5, N_gen=20)
Benchmark.benchmark_envs(['CartPole-v0'], N_dist=5, N_gen=20)
