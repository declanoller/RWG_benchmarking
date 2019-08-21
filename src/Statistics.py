import path_utils
from Evolve import Evolve, replot_evo_dict_from_dir
import traceback as tb
import os, json, shutil
import numpy as np
import matplotlib.pyplot as plt
import itertools
from copy import deepcopy
import pprint as pp
import pandas as pd
from tabulate import tabulate
import seaborn as sns
import ray, psutil, time

ray.init(num_cpus=psutil.cpu_count(), ignore_reinit_error=True, include_webui=False)
print('\nInitializing ray... Waiting for workers before starting...')
time.sleep(2.0)
print('Starting!\n')

'''

This is very similar to Benchmark.py, but that one was designed (when I had a
previous setup in mind) to run a set of parameters MULTIPLE times each. I.e.,
it would create an Evolve object and do evo_obj.evolve() several times to create
a distribution. Now, there's no "time dependence", so we really just want to be
able to look at separate parameter settings, but only running them once each.

I'm also getting rid of the whole "solved" aspect for now because it's based
on numbers that are hard to explain, making it a bit pointless.

run_param_dict() is the most basic function, just doing an evolution for a passed
param_dict. Other functions basically involve calling it given various inputs.

'''

################################ Statistics functions


@path_utils.timer
def run_param_dict(param_dict, N_gen, N_trials, base_dir):

    '''
    Pass a single params dict to run an evolve() of, including the env_name.

    Also pass an output_dir, or it will use the default output folder.

    This only runs each setting ONCE.
    '''

    # deepcopy, just to be safer
    params = deepcopy(param_dict)

    assert 'env_name' in params.keys(), 'Must supply an env_name!'

    env_name = params['env_name']
    params.pop('env_name')
    params['base_dir'] = base_dir

    try:
        # Run a single parameters setting
        e = Evolve(env_name, **params)
        evo_dict = e.evolve(N_gen, N_trials=N_trials)
        e.save_all_evo_stats(evo_dict)

        return evo_dict

    except:
        print(f'\n\nError in evolve with params: {params}. Traceback:\n')
        print(tb.format_exc())
        print('\n\nAttempting to continue...\n\n')

        return {}

@ray.remote
def run_param_dict_wrapper(param_dict, N_gen, N_trials, base_dir):

    # If a run_fname_label is provided, use that to create a more informative dir name.
    # Otherwise, just use the date.
    if 'run_fname_label' in param_dict.keys():
        run_fname_label = param_dict['run_fname_label']
    else:
        run_fname_label = 'vary_params'

    # Make plots for this params set
    if 'run_plot_label' in param_dict.keys():
        run_plot_label = param_dict['run_plot_label']
    else:
        run_plot_label = run_fname_label

    # Base dir for this set of params
    params_dir = os.path.join(base_dir, '{}_{}'.format(run_fname_label, path_utils.get_date_str()))
    os.mkdir(params_dir)

    print('\n\nNow running with params:')
    pp.pprint(param_dict, width=1)
    print('\n\n')

    stats_dict = run_param_dict(param_dict, N_gen, N_trials, params_dir)
    return stats_dict



@path_utils.timer
def run_multi_envs(env_list, **kwargs):

    '''
    Iterates over a list of env names you give it,
    running them and recording info.
    '''

    N_gen = kwargs.get('N_gen', 1000)
    N_trials = kwargs.get('N_trials', 1000)

    # Create dir for the results of this stats set.
    stats_dir = os.path.join(path_utils.get_output_dir(), 'Stats_{}'.format(path_utils.get_date_str()))
    os.mkdir(stats_dir)

    # Dict to hold results on timing, etc.
    stats_dict = {}

    for env_name in env_list:

        print(f'\nGetting stats for env {env_name} now...\n')

        param_dict = deepcopy(kwargs)
        param_dict['env_name'] = env_name

        stats_dict[env_name] = run_param_dict(param_dict, N_gen, N_trials, stats_dir)


    # Save distributions to file
    with open(os.path.join(stats_dir, 'multi_env_stats.json'), 'w+') as f:
        json.dump(stats_dict, f, indent=4)



def run_classic_control_envs(**kwargs):

    '''
    Loads gym_envs_info.json. This contains info about the envs we want to analyze.

    It then calls run_multi_envs() for the classic control envs.
    '''

    with open(os.path.join(path_utils.get_src_dir(), 'gym_envs_info.json'), 'r') as f:
        envs_dict = json.load(f)

    env_list = [k for k,v in envs_dict.items() if v['env_type']=='classic_control']

    print(f'Getting stats for: {env_list}')

    run_multi_envs(env_list, **kwargs)


def run_param_dict_list(params_dict_list, **kwargs):

    '''
    Pass this a list of dicts, where each has the different parameters you want
    to gather stats for.

    It then iterates through this list, doing a run for each dict.

    Note that it modifies the passed params_dict_list to add the results to it.

    '''

    # Create dir for the results of this stats run if one isn't provided.
    stats_dir = kwargs.get('stats_dir', None)
    if stats_dir is None:
        stats_dir = os.path.join(path_utils.get_output_dir(), 'Stats_{}'.format(path_utils.get_date_str()))
        os.mkdir(stats_dir)

    # Produce results in parallel
    for d in params_dict_list:
        d['result_ID'] = run_param_dict_wrapper.remote( d,
                                                        kwargs.get('N_gen', 100),
                                                        kwargs.get('N_trials', 10),
                                                        stats_dir)

    # Retrieve results from ID
    for d in params_dict_list:
        d['stats_dict'] = ray.get(d['result_ID'])

    # Return passed list, which should have dicts
    # modified with the results
    return params_dict_list



@path_utils.timer
def run_vary_params(constant_params_dict, vary_params_dict, **kwargs):

    '''
    This is a convenience function to easily vary parameters for analysis.
    You pass it constant_params_dict, which is a dict with the values that
    you want to remain constant between runs. Then, pass it vary_params_dict,
    which should have each parameter that you want to vary as a list of the values
    it should take.

    Example:

    constant_params_dict = {
        'env_name' : 'CartPole-v0',
        'N_gen' : 1000,
        'N_dist' : 100,
        'NN' : 'FFNN_multilayer'
    }

    vary_params_dict = {
        'N_hidden_units' : [2, 4, 8],
        'act_fn' : ['tanh', 'relu']
    }

    This will do 3*2 = 6 runs, for each of the combinations of varying parameters.
    '''

    # Create informative dir name
    vary_params = list(vary_params_dict.keys())
    stats_dir = os.path.join(
                        path_utils.get_output_dir(),
                        'Stats_vary_{}_{}'.format('_'.join(vary_params), path_utils.get_date_str()))
    print(f'\nSaving statistics run to {stats_dir}')
    os.mkdir(stats_dir)

    combined_params = {**constant_params_dict, **vary_params_dict}
    # Save params to file
    with open(os.path.join(stats_dir, 'vary_params.json'), 'w+') as f:
        json.dump(combined_params, f, indent=4)


    # Flatten list, pass to other function
    flat_param_list = vary_params_cross_products(constant_params_dict, vary_params_dict)
    flat_param_list = run_param_dict_list(flat_param_list, stats_dir=stats_dir, **kwargs)

    perc_cutoff = 99
    perc_cutoff_str = f'all_scores_{perc_cutoff}_perc'
    # Parse results
    for d in flat_param_list:
        stats_dict = d['stats_dict']

        all_scores = stats_dict['all_scores']
        d['mu_all_scores'] = np.mean(all_scores)
        d['sigma_all_scores'] = np.std(all_scores)

        d[perc_cutoff_str] = np.percentile(all_scores, perc_cutoff)

        #pp.pprint(d, width=1)
        # Get rid of this now
        d.pop('stats_dict')

    # Save results to csv for later parsing/plotting
    df = pd.DataFrame(flat_param_list)
    print(tabulate(df, headers=df.columns.values, tablefmt='psql'))
    df_fname = os.path.join(stats_dir, 'vary_params_stats.csv')
    df.to_csv(df_fname, index=False)

    # Only need to do if more than 2 params were varied.
    if len(vary_params) >= 2:

        # Create heatmap plots dir
        heatmap_dir = os.path.join(stats_dir, 'heatmap_plots')
        print(f'\nSaving heatmap plots to {heatmap_dir}')
        os.mkdir(heatmap_dir)

        # Iterate over all unique pairs of vary params, plot heatmaps of them
        for pair in itertools.combinations(vary_params, 2):

            print(f'Making heatmaps for {pair}')

            other_params_flat = [(k, v) for k,v in vary_params_dict.items() if k not in pair]
            other_params = [x[0] for x in other_params_flat]
            other_vals = [x[1] for x in other_params_flat]
            print(f'other params: {other_params}')

            # Create dir for specific pivot
            pivot_name = 'vary_{}_{}'.format(*pair)
            pivot_dir = os.path.join(heatmap_dir, pivot_name)
            os.mkdir(pivot_dir)

            # Select for each of the combos of the other params.
            for other_params_set in itertools.product(*other_vals):
                other_sel_dict = dict(zip(other_params, other_params_set))
                fname_label = path_utils.param_dict_to_fname_str(other_sel_dict)
                df_sel = df.loc[(df[list(other_sel_dict)] == pd.Series(other_sel_dict)).all(axis=1)]

                heatmap_plot(df_sel, *pair, 'mu_all_scores', pivot_dir, label=fname_label)
                heatmap_plot(df_sel, *pair, perc_cutoff_str, pivot_dir, label=fname_label)






def heatmap_plot(df, xvar, yvar, zvar, output_dir, **kwargs):

    #df = pd.read_csv(csv_fname)
    df = df.pivot(yvar, xvar, zvar)


    plt.close('all')
    plt.figure()
    ax = plt.gca()

    label = kwargs.get('label', '')

    sns.heatmap(df, annot=True, fmt=".1f", cmap='viridis', ax=ax)
    ax.set_title(f'{zvar} for constant {label}')
    plt.savefig(os.path.join(output_dir, f'vary_{xvar}_{yvar}__{zvar}_heatmap__const_{label}.png'))
    if kwargs.get('show_plot', False):
        plt.show()



def make_total_score_df(stats_dir):

    # Load csv that holds the names of all the dirs
    stats_overview_fname = os.path.join(stats_dir, 'vary_params_stats.csv')
    overview_df = pd.read_csv(stats_overview_fname)

    # unique fname labels in table
    run_fname_labels = overview_df.run_fname_label.unique()

    # Get the params that are varied
    vary_params_fname = os.path.join(stats_dir, 'vary_params.json')
    with open(vary_params_fname, 'r') as f:
        vary_params_dict = json.load(f)

    const_params = [k for k,v in vary_params_dict.items() if not isinstance(v, list)]
    vary_params = [k for k,v in vary_params_dict.items() if isinstance(v, list)]
    print(f'Params varied: {vary_params}')

    all_row_dfs = []

    # Iterate through table, for each run label, find its corresponding dir,
    # walk through it, get all its scores, create a dataframe from them,
    # then concatenate all these df's into a big one, that we can plot.
    for index, row in overview_df.iterrows():
        # Only get the params varied, turn them into a dict
        #row_dict = row[vary_params].to_dict()
        row_dict = row[vary_params + const_params].to_dict()
        #row_dict = row.to_dict()
        run_label = row['run_fname_label']
        # Get the one dir that has the run_fname_label in its name
        match_dirs = [x for x in os.listdir(stats_dir) if run_label in x]
        assert len(match_dirs)==1, 'Must only have one dir matching label!'
        vary_dir = match_dirs[0]
        # Clumsy, but: walk through this dir until you find the evo_stats.json,
        # then add its scores to the row_dict
        for root, dirs, files in os.walk(os.path.join(stats_dir, vary_dir)):
            if 'evo_stats.json' in files:
                with open(os.path.join(root, 'evo_stats.json'), 'r') as f:
                    evo_dict = json.load(f)

                row_dict['all_scores'] = evo_dict['all_scores']

        # pandas has the nice perk that if you create a df from a dict where
        # some of the entries are constants and one entry is a list, it duplicates
        # the constant values.
        row_df = pd.DataFrame(row_dict)
        all_row_dfs.append(row_df)


    all_scores_df = pd.concat(all_row_dfs)
    all_scores_fname = os.path.join(stats_dir, 'all_scores.csv')
    all_scores_df.to_csv(all_scores_fname, index=False)



def violin_plot(df, xvar, yvar, output_dir, **kwargs):

    #df = pd.read_csv(csv_fname)
    #df = df.pivot(yvar, xvar, zvar)


    plt.close('all')
    plt.figure()
    ax = plt.gca()

    label = kwargs.get('label', '')
    sns.violinplot(x=xvar, y="all_scores", hue=yvar, data=df)

    #sns.heatmap(df, annot=True, fmt=".1f", cmap='viridis', ax=ax)
    ax.set_title(f'all_scores for constant {label}')
    plt.savefig(os.path.join(output_dir, f'vary_{xvar}_{yvar}__all_scores_violinplot__const_{label}.png'))
    if kwargs.get('show_plot', False):
        plt.show()


def all_violin_plots(stats_dir):

    all_scores_fname = os.path.join(stats_dir, 'all_scores.csv')

    if not os.path.exists(all_scores_fname):
        make_total_score_df(stats_dir)

    # Load csv that holds the names of all the dirs
    df = pd.read_csv(all_scores_fname)

    # Get the params that are varied
    vary_params_fname = os.path.join(stats_dir, 'vary_params.json')
    with open(vary_params_fname, 'r') as f:
        all_params_dict = json.load(f)

    vary_params = [k for k,v in all_params_dict.items() if isinstance(v, list)]
    vary_params_dict = {k:v for k,v in all_params_dict.items() if isinstance(v, list)}

    print(f'Params varied: {vary_params}')


    # Only need to do if more than 2 params were varied.
    if len(vary_params) >= 2:

        # Create violin plots dir
        violin_plots_dir = os.path.join(stats_dir, 'violin_plots')
        print(f'\nSaving violin plots to {violin_plots_dir}')
        if os.path.exists(violin_plots_dir):
            shutil.rmtree(violin_plots_dir)
        os.mkdir(violin_plots_dir)

        # Iterate over all unique pairs of vary params, plot violin plots of them
        for pair in itertools.combinations(vary_params, 2):

            print(f'Making violin plots for {pair}')

            other_params_flat = [(k, v) for k,v in vary_params_dict.items() if k not in pair]
            other_params = [x[0] for x in other_params_flat]
            other_vals = [x[1] for x in other_params_flat]
            print(f'other params: {other_params}')

            # Create dir for specific pivot
            pivot_name = 'vary_{}_{}'.format(*pair)
            pivot_dir = os.path.join(violin_plots_dir, pivot_name)
            os.mkdir(pivot_dir)

            # Select for each of the combos of the other params.
            for other_params_set in itertools.product(*other_vals):
                other_sel_dict = dict(zip(other_params, other_params_set))
                fname_label = path_utils.param_dict_to_fname_str(other_sel_dict)
                df_sel = df.loc[(df[list(other_sel_dict)] == pd.Series(other_sel_dict)).all(axis=1)]
                print(other_sel_dict)
                violin_plot(df_sel, *pair, pivot_dir, label=fname_label)


################################# Helper functions


def vary_params_cross_products(constant_params_dict, vary_params_dict):

    '''
    Gets and returns the "cross product" of several lists in vary_params_dict,
    as a large flat dict.

    '''

    param_dict_list = []

    # Do this because we want to make sure to preserve order. Might not be necessary?
    vary_params_flat = [(k, v) for k,v in vary_params_dict.items()]
    vary_args = [x[0] for x in vary_params_flat]
    vary_vals = [x[1] for x in vary_params_flat]

    # Iterate over all combinations of vary params.
    for cur_vals in itertools.product(*vary_vals):

        current_vary_dict = dict(zip(vary_args, cur_vals))

        full_params = {**deepcopy(constant_params_dict), **current_vary_dict}

        # Get a run label, will be useful for varying lots.
        full_params['run_fname_label'] = path_utils.param_dict_to_fname_str(current_vary_dict)
        full_params['run_plot_label'] = path_utils.linebreak_every_n_spaces(
                                            path_utils.param_dict_to_label_str(
                                                            current_vary_dict))

        param_dict_list.append(full_params)

    return param_dict_list



def replot_whole_stats_dir(stats_dir):

    for root, dirs, files in os.walk(stats_dir):
        if 'run_params.json' in files:
            print('Replotting for dir {}'.format(root.split('/')[-1]))
            replot_evo_dict_from_dir(root)


#
