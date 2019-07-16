import path_utils
import Benchmark

Benchmark.benchmark_classic_control_envs(N_gen=2000)
#Benchmark.benchmark_envs(['CartPole-v0', 'LunarLander-v2'], N_gen=300)
