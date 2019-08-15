import path_utils
from FFNN1L import FFNN1L
import gym
import numpy as np


env = gym.make("MountainCar-v0")
# env = gym.wrappers.Monitor(env, 'video', force = True) # Uncomment to save video
nactions = env.action_space.n
ninputs = env.reset().size

# Network setup
net = FFNN1L(ninputs, nactions)

a = 1.0
b = 0.00001
c = 0.00001
b = 0
c = 0
net.set_weights(
np.array([
    [c, -a, -b],
    [0, 0, 0],
    [-c, a, b]
])
)
print(net.weights_matrix)

scores = []
for i in range(10000):
    # Gameplay loop
    obs = env.reset()
    score = 0
    done = False
    while not done:
        #env.render()
        action = net.get_action(obs)
        obs, rew, done, info = env.step(action)
        score += rew
    #print(f"Fitness: {score}")
    scores.append(score)
    last_100_score = np.mean(scores[-100:])
    if last_100_score >= -110.0 and len(scores) >= 100:
        print(f'Solved in generation {i}. Mean score over last 100 eps: {last_100_score:.2f}')
        break

env.close()
print('mean score: ', np.mean(scores))
