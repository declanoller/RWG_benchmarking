import path_utils
from Evolve import Evolve
import traceback as tb
import os, json

'''
Will eventually be made into a class that will handle all the benchmarking
details. For now, just holds benchmark_envs.
'''


def benchmark_envs(env_list, **kwargs):

    '''
    Iterates over a list of env names you give it,
    benchmarking it and recording info.
    '''

    # Create dir for the results of this benchmark.
    benchmark_dir = os.path.join(path_utils.get_output_dir(), 'Benchmark_{}'.format(path_utils.get_date_str()))
    os.mkdir(benchmark_dir)

    # Dict to hold results on timing, etc.
    benchmark_dict = {}

    for env_name in env_list:

        print(f'\nBenchmarking env {env_name} now...\n')

        try:
            e = Evolve(env_name, output_dir=benchmark_dir, **kwargs)
            evo_dict = e.evolve(kwargs.get('N_gen', 1000))
            e.plot_scores(evo_dict)
            #e.record_best_episode(evo_dict['best_weights'])
            benchmark_dict[env_name] = evo_dict

        except:
            print(f'\n\nError in evolve with env {env_name}. Traceback:\n')
            print(tb.format_exc())




def benchmark_classic_control_envs(**kwargs):

    with open(os.path.join(path_utils.get_src_dir(), 'gym_envs_info.json'), 'r') as f:
        envs_dict = json.load(f)

    env_list = [k for k,v in envs_dict.items() if v['env_type']=='classic_control']

    print(f'Benchmarking: {env_list}')

    benchmark_envs(env_list, **kwargs)


#
