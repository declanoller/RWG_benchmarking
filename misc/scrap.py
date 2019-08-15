


################ other mutation stuff from Agent.py:

    def mutate_gaussian_noise(self, sd=0.1):
        #
        noise = sd*np.random.randn(self.NN.nweights())
        new_weights = self.get_weight_matrix() + noise
        self.set_weight_matrix(new_weights)





    def softmax_choice(self, x):
        x_softmax = np.exp(x)/sum(np.exp(x))
        return np.random.choice(len(x), p=x_softmax)



################## FFNN_multilayer:

    def activate(self, inputs):
        """Activate the neural network

        """

        x = inputs

        for i,w in enumerate(self.weights_matrix):
            x = np.concatenate((x, [1.0]))
            x = np.dot(w, x)
            x = self.act_fn(x)

        return x


    def get_action(self, inputs):
        action_vec = self.activate(inputs)
        return self.output_fn(action_vec)














##################################### "solved" stuff from Evolve.py:

############# Taken from evolve(), checking if it's been solved and stuff

            if solved:
                print(f'\nSolved in gen {gen}!\n')
                break


        if not solved:
            print(f'\nReached max gen {N_gen} without being solved.\n')



                if mean_score > 1.8*self.solved_avg_reward:
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
                        #print(f'\t==> Unsolved. {self.N_eval_trials} ep mean score = {eval_mean:.2f} in gen {gen}')
                        pass
