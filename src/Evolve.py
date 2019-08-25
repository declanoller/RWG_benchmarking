import numpy as np
import gym
import matplotlib.pyplot as plt
import os, json

import path_utils
import Agent
gym.logger.set_level(40)

'''

Evolve class
--------------------

Used to run an agent in a gym env over a number (N_gen) of generations. Each
generation, the same agent is run for N_trials, because for many envs, different
initial conditions will give different scores for the same agent.

After every generation, Evolve.get_next_generation() is called, which gets
the next agent. For RWG, this simply involves picking a new set of random
weights, so there's no real "progression" or relation between generations.
However, this offers the opportunity to use other methods (CMA-ES, etc).

Uses the Agent class, which handles the NN and related stuff.

'''

class Evolve:

    def __init__(self, env_name, **kwargs):

        # The search method used. Default is Random Weight Guessing (RWG).
        self.search_method = kwargs.get('search_method', 'RWG')
        assert self.search_method in [
                                        'RWG',
                                        'gaussian_noise_hill_climb',
                                        'grid_search',
                                        'bin_grid_search',
                                        'sparse_bin_grid_search'
                                    ], 'Must supply valid search_method!'

        # Create env, create agent
        self.setup_env(env_name)
        self.agent = Agent.Agent(self.env, **kwargs)

        self.noise_sd = 1.0
        self.max_episode_steps = kwargs.get('max_episode_steps', 500)

        # Get the base dir, which is where runs will be saved to. Default
        # is /output/
        self.base_dir = kwargs.get('base_dir', path_utils.get_output_dir())

        # Datetime string for labeling the run
        self.dt_str = path_utils.get_date_str()

        # If you don't pass anything, it will create a dir in self.base_dir to
        # hold the results of this run, but you can supply your own externally.
        self.run_dir = kwargs.get('run_dir', None)
        if self.run_dir is None:
            self.run_dir = os.path.join(self.base_dir, f'{self.env_name}_evo_{self.dt_str}')
            os.mkdir(self.run_dir)

        # For saving the parameters used for the run. Run last in __init__().
        if kwargs.get('load_params_from_dir', False):
            self.load_params_dict()
        else:
            self.run_params = kwargs.copy()
            self.save_params_dict()


        #### Plot params
        self.plot_pt_alpha = 0.2
        self.plot_label_params = {
            'fontsize' : 14
        }
        self.plot_tick_params = {
            'fontsize' : 11
        }
        self.plot_title_params = {
            'fontsize' : 16
        }


    def setup_env(self, env_name):

        '''
        For setting up the env and getting the info
        about when it's solved, etc.
        '''

        with open(os.path.join(path_utils.get_src_dir(), 'gym_envs_info.json'), 'r') as f:
            envs_dict = json.load(f)

        assert env_name in envs_dict.keys(), f'Env {env_name} not in envs_dict!'

        self.env_name = env_name
        self.env = gym.make(env_name)

        # Two details for being considered "solved"
        self.solved_avg_reward = envs_dict[env_name]['solved_avg_reward']
        self.N_eval_trials = envs_dict[env_name]['N_eval_trials']


    def evolve(self, N_gen, **kwargs):

        '''
        Evolve the agent for N_gen generations,
        improving it with the selection mechanism.

        Each generation, use the same agent for N_trials to try and get a more
        representative score from it.
        '''

        N_trials = kwargs.get('N_trials', 3)

        all_scores = []
        best_scores = []
        all_trials = []
        L0_weights = []
        L1_weights = []
        L2_weights = []
        all_weights = []
        best_score = None
        best_weights = self.agent.get_weight_matrix()

        # Gameplay loop
        for gen in range(N_gen):

            if kwargs.get('print_gen', False):
                if gen % max(1, N_gen // 10) == 0:
                    print(f'\nGeneration {gen}/{N_gen}')

            score_trials = []
            for _ in range(N_trials):
                # Run episode, get score, record score
                score_trials.append(self.run_episode())

            # Take mean score of N_trials, record if best score yet
            mean_score = np.mean(score_trials)
            if (best_score is None) or (mean_score > best_score):
                best_score = mean_score
                #print(f'New best score {best_score:.3f} in generation {gen}')
                best_weights = self.agent.get_weight_matrix()

            # Get stats about the weights of the NN
            weight_sum_dict = self.agent.get_weight_sums()
            L0_weights.append(weight_sum_dict['L0'])
            L1_weights.append(weight_sum_dict['L1'])
            L2_weights.append(weight_sum_dict['L2'])

            all_scores.append(mean_score)
            best_scores.append(best_score)
            all_trials.append(score_trials)

            if kwargs.get('save_all_weights', False):
                if mean_score >= 180:
                    all_weights.append(self.agent.get_weights_as_list())

            if self.agent.search_done():
                print(f'Search done in gen {gen}\n\n')
                break

            # Get next agent.
            self.get_next_generation(all_scores, best_scores, best_weights)



        ret_dict = {
            'best_scores' : best_scores,
            'all_scores' : all_scores,
            'all_trials' : all_trials,
            'best_weights' : [bw.tolist() for bw in best_weights],
            'L0_weights' : L0_weights,
            'L1_weights' : L1_weights,
            'L2_weights' : L2_weights,
        }

        if kwargs.get('save_all_weights', False):
            ret_dict['all_weights'] = all_weights

        return ret_dict


    def run_episode(self, **kwargs):

        '''
        Run episode with gym env. Returns the total score
        for the episode. Pass show_ep=True to render the episode.
        '''

        show_ep = kwargs.get('show_ep', False)

        if kwargs.get('record_ep', False):
            self.env = gym.wrappers.Monitor(self.env, self.run_dir, force = True)

        obs = self.env.reset()
        self.agent.init_episode()
        score = 0
        steps = 0
        done = False
        while not done:
            if show_ep:
                self.env.render()
                if steps % 10 == 0:
                    print(f'step {steps}, score {score:.2f}')

            action = self.agent.get_action(obs)
            obs, rew, done, info = self.env.step(action)
            score += rew
            steps += 1
            if steps >= self.max_episode_steps:
                done = True

        if show_ep:
            self.env.close()
            print(f'Score = {score:.3f}')

        return score


    def get_next_generation(self, all_scores, best_scores, best_weights):

        '''
        For getting the next generation using some selection criteria.
        Right now it's just RWG, which resets the weights randomly.
        '''

        if self.search_method == 'RWG':
            self.agent.set_random_weights()

        elif self.search_method == 'gaussian_noise_hill_climb':
            if all_scores[-1] < best_scores[-1]:
                self.agent.set_weight_matrix(best_weights)

            self.agent.mutate_gaussian_noise(sd=self.noise_sd)

        elif self.search_method in ['grid_search', 'bin_grid_search', 'sparse_bin_grid_search']:

            self.agent.mutate_grid_search()



    ################################ Plotting/saving functions

    def plot_scores(self, evo_dict, **kwargs):

        '''
        For plotting results. Pass it a dict of the form
        returned by evolve().

        Plots several versions of the same data (only mean, in the order they're
        run, mean, but ordered by increasing value, and then the mean and the scores
        for each trial).
        '''

        ###################### In time order

        plt.close('all')
        plt.plot(evo_dict['all_scores'], color='dodgerblue', label='All mean scores')

        plt.xlabel('Generation', **self.plot_label_params)
        plt.ylabel('Generation mean score', **self.plot_label_params)

        plt.xticks(**self.plot_tick_params)
        plt.yticks(**self.plot_tick_params)

        #plt.legend()
        plt.title(f'{self.env_name} environment', **self.plot_title_params)
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, '{}_score_mean_timeseries_{}.png'.format(self.env_name, self.dt_str)))

        ###################### In mean order

        all_scores = evo_dict['all_scores']
        all_scores = sorted(all_scores)

        plt.close('all')
        plt.plot(all_scores, color='mediumseagreen')

        plt.xlabel('Sorted by generation mean score', **self.plot_label_params)
        plt.ylabel('Generation mean score', **self.plot_label_params)

        plt.xticks(**self.plot_tick_params)
        plt.yticks(**self.plot_tick_params)

        #plt.legend()
        plt.title(f'{self.env_name} environment', **self.plot_title_params)
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, '{}_score_mean_ordered_{}.png'.format(self.env_name, self.dt_str)))

        ###################### In mean order, with all trials

        all_trials = evo_dict['all_trials']
        all_trials = sorted(all_trials, key=lambda x: np.mean(x))

        all_trials_mean = np.mean(all_trials, axis=1)
        # Wait, is this right? all_trials has the form [[8, 4, 5], [1, 9, 2], ...]
        # Just tested, I think it is, but it's not ideal. Each N_trials list is
        # actually treated as a separate dataset... ahh actually, it's treating
        # the first of each as a dataset together, then the second of each, etc...
        plt.close('all')
        plt.plot(all_trials, 'o', color='tomato', alpha=self.plot_pt_alpha)
        plt.plot(all_trials_mean, color='black')

        plt.xlabel('Sorted by generation mean score', **self.plot_label_params)
        plt.ylabel('Generation trials score', **self.plot_label_params)

        plt.xticks(**self.plot_tick_params)
        plt.yticks(**self.plot_tick_params)

        #plt.legend()
        plt.title(f'{self.env_name} environment', **self.plot_title_params)
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, '{}_score_trials_ordered_{}.png'.format(self.env_name, self.dt_str)))


    def plot_weight_stats(self, evo_dict, **kwargs):


        '''
        For plotting episode mean scores and the corresponding L1 or L2 sums
        of the weight matrix that produced those scores.

        '''


        L0_weights = evo_dict['L0_weights']
        L1_weights = evo_dict['L1_weights']
        L2_weights = evo_dict['L2_weights']
        all_scores = evo_dict['all_scores']

        ###################### L0
        plt.close('all')
        plt.plot(all_scores, L0_weights, 'o', color='forestgreen', alpha=self.plot_pt_alpha)

        plt.xlabel('Generation mean score', **self.plot_label_params)
        plt.ylabel('L0/N_weights', **self.plot_label_params)

        plt.xticks(**self.plot_tick_params)
        plt.yticks(**self.plot_tick_params)

        #plt.legend()
        plt.title(f'{self.env_name} environment,\n L0 sum of weights', **self.plot_title_params)
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, '{}_L0_vs_meanscore_{}.png'.format(self.env_name, self.dt_str)))


        ###################### L1
        plt.close('all')
        plt.plot(all_scores, L1_weights, 'o', color='forestgreen', alpha=self.plot_pt_alpha)

        plt.xlabel('Generation mean score', **self.plot_label_params)
        plt.ylabel('L1/N_weights', **self.plot_label_params)

        plt.xticks(**self.plot_tick_params)
        plt.yticks(**self.plot_tick_params)

        #plt.legend()
        plt.title(f'{self.env_name} environment,\n L1 sum of weights', **self.plot_title_params)
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, '{}_L1_vs_meanscore_{}.png'.format(self.env_name, self.dt_str)))

        ######################## L2
        plt.close('all')
        plt.plot(all_scores, L2_weights, 'o', color='forestgreen', alpha=self.plot_pt_alpha)

        plt.xlabel('Generation mean score', **self.plot_label_params)
        plt.ylabel('L2/N_weights', **self.plot_label_params)

        plt.xticks(**self.plot_tick_params)
        plt.yticks(**self.plot_tick_params)

        #plt.legend()
        plt.title(f'{self.env_name} environment,\n L2 sum of weights', **self.plot_title_params)
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, '{}_L2_vs_meanscore_{}.png'.format(self.env_name, self.dt_str)))


    def plot_all_trial_stats(self, evo_dict, **kwargs):

        '''
        Plots the variance, min, and max of the scores for the N_trials of
        each episode, as a function of the mean score for that episode.

        '''

        ####################### Episode score variance
        plt.close('all')

        sigma = np.std(evo_dict['all_trials'], axis=1)
        N_trials = len(evo_dict['all_trials'][0])

        plt.plot(evo_dict['all_scores'], sigma, 'o', color='dodgerblue', alpha=self.plot_pt_alpha)

        plt.xlabel('Generation mean score', **self.plot_label_params)
        plt.ylabel('Variance of generation scores', **self.plot_label_params)

        plt.xticks(**self.plot_tick_params)
        plt.yticks(**self.plot_tick_params)

        plt.title(f'{self.env_name} environment,\n variance of N_trials = {N_trials}', **self.plot_title_params)
        plt.tight_layout()
        fname = os.path.join(self.run_dir, '{}_variance_meanscore_{}.png'.format(self.env_name, self.dt_str))
        plt.savefig(fname)


        ####################### Min generation score
        plt.close('all')

        trial_min = np.min(evo_dict['all_trials'], axis=1)
        N_trials = len(evo_dict['all_trials'][0])

        plt.plot(evo_dict['all_scores'], trial_min, 'o', color='dodgerblue', alpha=self.plot_pt_alpha)

        plt.xlabel('Generation mean score', **self.plot_label_params)
        plt.ylabel('Min of generation scores', **self.plot_label_params)

        plt.xticks(**self.plot_tick_params)
        plt.yticks(**self.plot_tick_params)

        plt.title(f'{self.env_name} environment,\n min score of N_trials = {N_trials}', **self.plot_title_params)
        plt.tight_layout()
        fname = os.path.join(self.run_dir, '{}_min_score_{}.png'.format(self.env_name, self.dt_str))
        plt.savefig(fname)

        ####################### Max episode score
        plt.close('all')

        trial_max = np.max(evo_dict['all_trials'], axis=1)
        N_trials = len(evo_dict['all_trials'][0])

        plt.plot(evo_dict['all_scores'], trial_max, 'o', color='dodgerblue', alpha=self.plot_pt_alpha)

        plt.xlabel('Generation mean score', **self.plot_label_params)
        plt.ylabel('Max of generation scores', **self.plot_label_params)

        plt.xticks(**self.plot_tick_params)
        plt.yticks(**self.plot_tick_params)

        plt.title(f'{self.env_name} environment,\n max score of N_trials = {N_trials}', **self.plot_title_params)
        plt.tight_layout()
        fname = os.path.join(self.run_dir, '{}_max_score_{}.png'.format(self.env_name, self.dt_str))
        plt.savefig(fname)


        ####################### Min and max episode score
        plt.close('all')

        trial_min = np.min(evo_dict['all_trials'], axis=1)
        trial_max = np.max(evo_dict['all_trials'], axis=1)
        N_trials = len(evo_dict['all_trials'][0])

        plt.plot(evo_dict['all_scores'], trial_min, 'o', color='mediumturquoise', alpha=self.plot_pt_alpha)
        plt.plot(evo_dict['all_scores'], trial_max, 'o', color='plum', alpha=self.plot_pt_alpha)

        plt.xlabel('Generation mean score', **self.plot_label_params)
        plt.ylabel('Min and max of generation scores', **self.plot_label_params)

        plt.xticks(**self.plot_tick_params)
        plt.yticks(**self.plot_tick_params)

        plt.title(f'{self.env_name} environment, min (turquoise) and \nmax (purple) score of N_trials = {N_trials}', **self.plot_title_params)
        plt.tight_layout()
        fname = os.path.join(self.run_dir, '{}_min_max_score_{}.png'.format(self.env_name, self.dt_str))
        plt.savefig(fname)


    def show_best_episode(self, weights):

        '''
        Pass it the weights matrix you want to run it with,
        e.g., best_weights returned from evolve(). It runs
        an episode and renders it.
        '''

        self.agent.set_weight_matrix(weights)
        ep_score = self.run_episode(show_ep=True, record_ep=False)


    def record_best_episode(self, weights):

        '''
        Pass it the weights matrix you want to run it with,
        e.g., best_weights returned from evolve(). It runs
        an episode and renders it.
        '''

        self.agent.set_weight_matrix(weights)
        ep_score = self.run_episode(show_ep=True, record_ep=True)


    def save_all_evo_stats(self, evo_dict):
        '''
        For saving all the stats and plots for the evolution, just a collector
        function.

        '''

        self.plot_scores(evo_dict)
        self.plot_all_trial_stats(evo_dict)
        self.plot_evo_histogram(evo_dict['all_scores'], 'Generation mean score', f'{self.env_name}_all_scores_dist_{self.dt_str}.png', plot_log=True)
        self.save_evo_dict(evo_dict)
        self.plot_weight_stats(evo_dict)


    def save_evo_dict(self, evo_dict):
        '''
        For saving the results of the run in a .json file, for later analysis.
        '''

        # Maybe not necessary, but to be careful to not modify the original
        evo_dict_copy = evo_dict.copy()
        if 'best_weights' in evo_dict_copy.keys():
            evo_dict_copy.pop('best_weights')

        fname = os.path.join(self.run_dir, 'evo_stats.json')
        # Save distributions to file

        with open(fname, 'w+') as f:
            json.dump(evo_dict_copy, f, indent=4)


    def save_params_dict(self):
        '''
        For saving the parameters used in a .json file, for later analysis.
        '''

        self.run_params['env_name'] = self.env_name
        self.run_params['search_method'] = self.search_method
        self.run_params['base_dir'] = self.base_dir
        self.run_params['dt_str'] = self.dt_str
        self.run_params['run_dir'] = self.run_dir
        self.run_params['NN_type'] = self.agent.NN_type

        fname = os.path.join(self.run_dir, 'run_params.json')
        # Save distributions to file

        with open(fname, 'w+') as f:
            json.dump(self.run_params, f, indent=4)


    def load_params_dict(self):
        '''
        For loading the parameters from a saved .json file, for later analysis.
        '''
        fname = os.path.join(self.run_dir, 'run_params.json')
        # Save distributions to file

        with open(fname, 'r') as f:
            self.run_params = json.load(f)

        self.env_name = self.run_params['env_name']
        self.search_method = self.run_params['search_method']
        self.base_dir = self.run_params['base_dir']
        self.dt_str = self.run_params['dt_str']
        self.run_dir = self.run_params['run_dir']
        self.agent.NN_type = self.run_params['NN_type'] # not necessary probably? Be careful



    def plot_evo_histogram(self, dist, dist_label, fname, **kwargs):

        '''
        For plotting the distribution of various benchmarking stats for self.env_name.
        Plots a vertical dashed line at the mean.

        kwarg plot_log = True also plots one with a log y axis, which is often
        better because the number of best solutions are very small.
        '''

        fname = os.path.join(self.run_dir, fname)

        plt.close('all')
        mu = np.mean(dist)
        sd = np.std(dist)

        if kwargs.get('N_bins', None) is None:
            plt.hist(dist, color='dodgerblue', edgecolor='gray')
        else:
            plt.hist(dist, color='dodgerblue', edgecolor='gray', bins=kwargs.get('N_bins', None))

        plt.axvline(mu, linestyle='dashed', color='tomato', linewidth=2)
        plt.xlabel(dist_label, **self.plot_label_params)
        plt.ylabel('Counts', **self.plot_label_params)

        plt.xticks(**self.plot_tick_params)
        plt.yticks(**self.plot_tick_params)

        plt.title(f'{dist_label} distribution for {self.env_name}\n$\mu = {mu:.1f}$, $\sigma = {sd:.1f}$', **self.plot_title_params)
        plt.savefig(fname)

        if kwargs.get('plot_log', False):
            if kwargs.get('N_bins', None) is None:
                plt.hist(dist, color='dodgerblue', edgecolor='gray', log=True)
            else:
                plt.hist(dist, color='dodgerblue', edgecolor='gray', bins=kwargs.get('N_bins', None), log=True)

            plt.axvline(mu, linestyle='dashed', color='tomato', linewidth=2)
            plt.xlabel(dist_label, **self.plot_label_params)
            plt.ylabel('log(Counts)', **self.plot_label_params)

            plt.xticks(**self.plot_tick_params)
            plt.yticks(**self.plot_tick_params)

            plt.title(f'{dist_label} distribution for {self.env_name}\n$\mu = {mu:.1f}$, $\sigma = {sd:.1f}$', **self.plot_title_params)
            plt.savefig(fname.replace('dist', 'log_dist'))


def replot_evo_dict_from_dir(dir):

    '''
    Minor fix: originally this would open run_params.json and read the run_dir
    field, and pass that to the Evolve() object. However, that caused trouble
    if the run was done on another machine, because the path was absolute,
    so it would then be looking for a path that might not exist on the machine
    that this function is being run on.

    Instead, since we're already assuming this dir is a run dir, it should just
    take this dir and rewrite run_dir and base_dir.

    '''


    assert os.path.exists(dir), f'Dir must exist to load from! Dir {dir} DNE.'

    run_params_json_fname = os.path.join(dir, 'run_params.json')
    assert os.path.exists(run_params_json_fname), f'run_params.json must exist in dir to load from! {run_params_json_fname} DNE.'

    evo_dict_fname = os.path.join(dir, 'evo_stats.json')
    assert os.path.exists(evo_dict_fname), f'evo_stats.json must exist in dir to load from! {evo_dict_fname} DNE.'

    # Get run_params to recreate the object
    with open(run_params_json_fname, 'r') as f:
        run_params = json.load(f)

    # Rewrite run_params in case it was originally run on another machine.
    run_params['run_dir'] = dir
    base_dir = os.path.abspath(os.path.join(dir, os.pardir))
    run_params['base_dir'] = base_dir

    with open(run_params_json_fname, 'w+') as f:
        json.dump(run_params, f, indent=4)


    # Recreate Evolve object, get evo_dict, replot
    # Have to pass run_dir so it doesn't automatically create a new dir.
    e = Evolve(run_params['env_name'], run_dir=run_params['run_dir'], load_params_from_dir=True)

    # Get evo_dict to replot statistics found
    with open(evo_dict_fname, 'r') as f:
        evo_dict = json.load(f)

    # Replot
    e.save_all_evo_stats(evo_dict)




#
