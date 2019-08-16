
################### readme:



## Issues, upgrades, expansions

See Issues tab, which will be a general To-do list.


To run it for the cartpole environment, do:

```
python3 scripts/cartpole_example.py
```

This will run it for a maximum of 1000 generations. Each generation, it will reset the weights of a 1 layer RNN (from class `RNN1L`), and use it to play 3 episodes of CartPole. It takes the average of these 3 episodes to get a representative score. If this score is better than the current best score, it runs this NN for 100 trials (`CartPole-v0`'s "solved" condition is having an average score of 195.0 over 100 trials).

If it's solved, it breaks. When it's done, it plots the best FF at each generation, as well as the FF from each generation. Then, it runs an episode with the best weights set found, and saves the video.


## Higher order benchmarking

The above solves the `CartPole-v0` env. However, we want to benchmark more envs. In addition, a single solve isn't representative, especially given the randomness inherent in many of these environments. What would give more specific info is to do the above benchmarking, but run it a number of times to create a distribution.

This is easily doable with the `Evolve` class and the `Benchmark.py` functions. Briefly, an `Evolve` object creates an `Agent` class object for a given env, and then `Evolve.evolve()` does RWG to try and solve that env. It returns the solve time (in number of generations).

`Benchmark.benchmark_envs()` takes a list of envs. For each, it creates a dist of the solve times, by (some specified number of times) creating a new `Evolve` object and solving the env.

Here's a simple usage, from `scripts/benchmark_example.py`:

```
import path_utils
import Benchmark

Benchmark.benchmark_envs(['CartPole-v0'], N_dist=100, N_gen=1000)
```

This will only benchmark `CartPole-v0`. It will create a distribution of solve times from `N_dist` instances of that env. Each one will have a max number of generations `N_gen` (if it doesn't solve in that time, it gets marked as the maximum time; this might be suboptimal because it's underestimating these outliers).

This produces:

<p align="center">
  <img width="640" height="480" src="misc/CartPole-v0_solve_gen_dist.png">
</p>

Something curious: even though it seems to have a well-defined Gamma-like (?) distribution shape, there are always some at the maximum `N_gen` (meaning they didn't solve). This is curious, since every iteration of `evolve()` is independent. However, since we're just testing for `mean_score` > `best_score`, it's possible that it gets a "lucky" set of weights that got a high score for its 3 episode trials, but couldn't solve it. Then, later sets that might not get as high a 3-episode score, but *would* solve it, don't get tested. This has to be looked at more.

In addition, it creates a timestamped directory in the `outputs` directory for the benchmarking run. Within that, it creates:

* For each env benchmarked, a directory with the FF plot for each run
* A distribution plot for each env
* A .json file with the solve times for each env, `solve_time_dists.json`

Example structure:

```
├── output
│   ├── Benchmark_17-07-2019_10-52-41
│   │   ├── CartPole-v0
│   │   ├── CartPole-v0_solve_gen_dist.png
│   │   ├── LunarLander-v2
│   │   ├── LunarLander-v2_solve_gen_dist.png
│   │   └── solve_time_dists.json
│   └── Benchmark_17-07-2019_10-57-01
│       ├── CartPole-v0
│       ├── CartPole-v0_solve_gen_dist.png
│       └── solve_time_dists.json
```

(the FF plots for each env have been omitted, as there are `N_dist` many of them.)

Similarly, `Benchmark.benchmark_classic_control_envs()` will call `Benchmark.benchmark_envs()` with only the "classic control" envs.

Other simple variations have been added, to be tested:

* Softmax vs argmax outputs for discrete action spaces
* Different activation functions (even linear can solve many of these environments! See CartPole and LunarLander solves here: https://www.declanoller.com/2019/01/25/beating-openai-games-with-neuroevolution-agents-pretty-neat/ )
* Feedforward vs RNN networks (currently using FF as default, because it seems to solve much quicker)




## Higher level comparisons

`CartPole-v0` (and other envs) is so simple that even a NN without a hidden layer can solve it. However, more complicated envs might need more




I also added a feedforward NN (class `FFNN1L.py`), since the default RNN isn't crucial for most classic control problems, and adds extra weights that increase the search space. It seems to solve it lots faster on average, as expected.

The [OpenAI gym leaderboard score](https://github.com/openai/gym/wiki/Leaderboard#cartpole-v0) for `CartPole-v0` gives a fastest solve time of 24 episodes before solve. This is pretty fast, but also hard to compare to for several reasons:

* lots of the top scores are tailor made to the environment (see the "0 episodes" solve time for the top `MountainCar-v0`)
* What are their statistics? Did they run it for 1000 and choose the fastest one? In that case, RWG can sometimes solve it in 1 or 2 episodes (if it gets lucky)



























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
