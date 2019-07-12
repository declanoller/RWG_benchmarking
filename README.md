
# Random Weight Guessing (RWG) benchmarking

For benchmarking, using the [tinynet](https://github.com/giuse/tinynet) library.

RWG is simply randomly guessing the weights of a neural network (NN) until it gives the desired behavior. It's strangely effective at many tasks. A likely hypothesis that's been put forward is that valid sets of weights are "dense" in weight-space; i.e., a relatively large portion of the possible weight sets solve the problem.

Although RWG would probably have trouble with very complex tasks, the fact that it does so reasonably well a these simpler ones indicates that at the very least, it should be a good benchmark to compare other algorithms to.

## CartPole-v0

Here, we're just testing it for the `CartPole-v0` environment. It's incredibly simple, and thus, it solves it very quickly:

<p align="center">
  <img width="600" height="600" src="misc/NE_cartpole_FF.png">
</p>

You can see that because it's doing RWG, the results of runs are independent of the previous ones, i.e., nothing "builds off" of previous runs.

<p align="center">
  <img width="600" height="600" src="misc/cartpole-v0_ep.gif">
</p>

Being random, it has a lot of variance in how many generations it takes to find a solution. A more rigorous experiment would run this many times to get a distribution of the solve times.

I also added a feedforward NN (class `FFNN1L.py`), since the default RNN isn't crucial for most classic control problems, and adds extra weights that increase the search space. It seems to solve it lots faster on average, as expected.

The [OpenAI gym leaderboard score](https://github.com/openai/gym/wiki/Leaderboard#cartpole-v0) for `CartPole-v0` gives a fastest solve time of 24 episodes before solve. This is pretty fast, but also hard to compare to for several reasons:

* lots of the top scores are tailor made to the environment (see the "0 episodes" solve time for the top `MountainCar-v0`)
* What are their statistics? Did they run it for 1000 and choose the fastest one? In that case, RWG can sometimes solve it in 1 or 2 episodes (if it gets lucky)

## Use

To run it for the cartpole environment, do:

```
python3 scripts/cartpole_example.py
```

This will run it for a maximum of 1000 generations. Each generation, it will reset the weights of a 1 layer RNN (from class `RNN1L`), and use it to play 3 episodes of CartPole. It takes the average of these 3 episodes to get a representative score. If this score is better than the current best score, it runs this NN for 100 trials (`CartPole-v0`'s "solved" condition is having an average score of 195.0 over 100 trials).

If it's solved, it breaks. When it's done, it plots the best FF at each generation, as well as the FF from each generation. Then, it runs an episode with the best weights set found, and saves the video.


## Relevant links

* https://github.com/giuse/tinynet
* https://www.bioinf.jku.at/publications/older/ch9.pdf
* https://gist.github.com/giuse/3d16c947259173d571cf82e28a2f7a7e
