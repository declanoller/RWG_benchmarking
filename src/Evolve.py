import numpy as np
import gym
import matplotlib.pyplot as plt
import os
import json

import path_utils
import Agent
gym.logger.set_level(40)

class Evolve:

    def __init__(self, env_name, **kwargs):

        # Create env, create agent
        self.setup_env(env_name)
        self.agent = Agent.Agent(self.env, **kwargs)

        # The search method used. Default is Random Weight Guessing (RWG).
        self.search_method = kwargs.get('search_method', 'RWG')
        assert self.search_method in ['RWG', 'gaussian_noise_hill_climb'], 'Must supply valid search_method!'
        self.noise_sd = 1.0

        # Create or get output dir, where plots/etc will save to
        self.output_dir = kwargs.get('output_dir', path_utils.get_output_dir())


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

        If a new best score is found, the agent is tried for N_eval_trials
        episodes to see if can achieve the average solved score.
        '''

        N_trials = kwargs.get('N_trials', 3)

        all_scores = []
        best_scores = []
        best_score = None
        solved = False
        solve_gen = None
        best_weights = self.agent.get_weight_matrix()

        # Gameplay loop
        for gen in range(N_gen):

            score_trials = []
            for _ in range(N_trials):
                # Run episode, get score, record score
                score_trials.append(self.run_episode())

            # Take mean score of N_trials, record if best score yet
            mean_score = np.mean(score_trials)
            if (best_score is None) or (mean_score > best_score):
                best_score = mean_score
                print(f'New best score {best_score:.3f} in generation {gen}')
                best_weights = self.agent.get_weight_matrix()

                if mean_score > 0.8*self.solved_avg_reward:
                    # If it achieved a new best score, test for self.N_eval_trials episode average score.
                    # If self.N_eval_trials ep mean score is >= self.solved_avg_reward, it's considered solved.
                    eval_trials = []
                    for _ in range(self.N_eval_trials):
                        eval_trials.append(self.run_episode())

                    eval_mean = np.mean(eval_trials)
                    if eval_mean >= self.solved_avg_reward:
                        print(f'\t==> Solved! {self.N_eval_trials} ep mean score = {eval_mean:.2f} in gen {gen}')
                        solved = True
                        solve_gen = gen
                    else:
                        print(f'\t==> Unsolved. {self.N_eval_trials} ep mean score = {eval_mean:.2f} in gen {gen}')

            all_scores.append(mean_score)
            best_scores.append(best_score)

            if solved: break

            # Get next agent.
            self.get_next_generation(all_scores, best_scores, best_weights)

        if not solved:
            print(f'\nReached max gen {N_gen} without being solved.\n')

        return {
            'best_scores' : best_scores,
            'all_scores' : all_scores,
            'best_weights' : best_weights,
            'solved' : solved,
            'solve_gen' : solve_gen
        }


    def run_episode(self, **kwargs):

        '''
        Run episode with gym env. Returns the total score
        for the episode. Pass show_ep=True to render the episode.
        '''

        show_ep = kwargs.get('show_ep', False)

        if kwargs.get('record_ep', False):
            self.env = gym.wrappers.Monitor(self.env, self.output_dir, force = True)

        obs = self.env.reset()
        self.agent.init_episode()
        score = 0
        done = False
        while not done:
            if show_ep:
                self.env.render()

            action = self.agent.get_action(obs)
            obs, rew, done, info = self.env.step(action)
            score += rew

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
            self.agent.reset()

        elif self.search_method == 'gaussian_noise_hill_climb':
            if all_scores[-1] < best_scores[-1]:
                self.agent.set_weight_matrix(best_weights)

            self.agent.mutate_gaussian_noise(sd=self.noise_sd)


    def plot_scores(self, score_dict, **kwargs):

        '''
        For plotting results. Pass it a dict of the form
        returned by evolve().
        '''

        plt.close('all')
        plt.plot(score_dict['best_scores'], color='tomato', label='Best FF found')
        plt.plot(score_dict['all_scores'], color='dodgerblue', label='All FF')

        plt.xlabel('Episode')
        plt.ylabel('Fitness Function (FF)')
        plt.legend()
        plt.title(f'{self.env_name} environment')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '{}_FF_{}.png'.format(self.env_name, path_utils.get_date_str())))
        if kwargs.get('show_plot', False):
            plt.show()


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
