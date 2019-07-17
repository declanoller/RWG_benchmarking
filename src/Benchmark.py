import path_utils
from Evolve import Evolve
import traceback as tb
import os, json
import numpy as np
import matplotlib.pyplot as plt


'''
Will eventually be made into a class that will handle all the benchmarking
details. For now, just holds benchmark_envs.
'''


def benchmark_envs(env_list, **kwargs):

    '''
    Iterates over a list of env names you give it,
    benchmarking it and recording info.
    '''

    N_dist = kwargs.get('N_dist', 10) # How many evolutions to run, to form a distribution
    N_gen = kwargs.get('N_gen', 1000)

    # Create dir for the results of this benchmark.
    benchmark_dir = os.path.join(path_utils.get_output_dir(), 'Benchmark_{}'.format(path_utils.get_date_str()))
    os.mkdir(benchmark_dir)

    # Dict to hold results on timing, etc.
    benchmark_dict = {}

    for env_name in env_list:

        print(f'\nBenchmarking env {env_name} now...\n')
        solve_gen_dist = [] # To hold the solve times distribution for this env.
        try:

            # Create a dir for this env.
            env_dir = os.path.join(benchmark_dir, env_name)
            os.mkdir(env_dir)

            for _ in range(N_dist):
                e = Evolve(env_name, output_dir=env_dir, **kwargs)
                evo_dict = e.evolve(N_gen)
                e.plot_scores(evo_dict)
                #e.record_best_episode(evo_dict['best_weights'])
                #benchmark_dict[env_name] = evo_dict
                # If it didn't solve it, give it the max number (which
                # may actually still be an underestimate).
                if evo_dict['solved']:
                    solve_gen_dist.append(evo_dict['solve_gen'])
                else:
                    solve_gen_dist.append(N_gen)


            benchmark_dict[env_name] = solve_gen_dist


        except:
            print(f'\n\nError in evolve with env {env_name}. Traceback:\n')
            print(tb.format_exc())

    # Save distributions to file
    with open(os.path.join(benchmark_dir, 'solve_time_dists.json'), 'w+') as f:
        json.dump(benchmark_dict, f, indent=4)

    # Plot each env dist.
    for k,v in benchmark_dict.items():
        fname = os.path.join(benchmark_dir, f'{k}_solve_gen_dist.png')
        plot_solve_gen_dist(v, k, fname)




def benchmark_classic_control_envs(**kwargs):

    with open(os.path.join(path_utils.get_src_dir(), 'gym_envs_info.json'), 'r') as f:
        envs_dict = json.load(f)

    env_list = [k for k,v in envs_dict.items() if v['env_type']=='classic_control']

    print(f'Benchmarking: {env_list}')

    benchmark_envs(env_list, **kwargs)



def plot_solve_gen_dist(dist, env, fname, **kwargs):

    plt.close('all')
    mu = np.mean(dist)
    sd = np.std(dist)
    plt.hist(dist, color='dodgerblue', edgecolor='gray')
    plt.axvline(mu, linestyle='dashed', color='tomato', linewidth=2)
    plt.xlabel('Solve time (episodes)')
    plt.ylabel('Counts')
    plt.title(f'Solve time distribution for {env}\n$\mu = {mu:.1f}$, $\sigma = {sd:.1f}$')
    plt.savefig(fname)






#
