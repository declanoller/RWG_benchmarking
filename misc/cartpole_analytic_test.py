#import path_utils
#from FFNN1L import FFNN1L
import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("CartPole-v0")
# env = gym.wrappers.Monitor(env, 'video', force = True) # Uncomment to save video
nactions = env.action_space.n
ninputs = env.reset().size
'''
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
'''

state_hist = []

#ang_vel_cutoff_vals = np.linspace(-2, 1, 4)
#ang_vel_cutoff_vals = np.linspace(-5, 5, 40)
w1_div_w2 = np.linspace(-4, 4, 100)
scores_mean = []
scores_sd = []

for w in w1_div_w2:

    scores = []
    for i in range(100):
        # Gameplay loop
        obs = env.reset()
        score = 0
        done = False
        while not done:
            x, v, ang, ang_vel = obs
            #print(f'x={x:.2f}, v={v:.2f}, ang={ang:.2f}, ang_vel={ang_vel:.2f}')
            '''if (ang >= 0) and (ang_vel > ang_vel_cutoff):
                action = 1
            else:
                action = 0'''

            if ang_vel > -w*ang:
                action = 1
            else:
                action = 0



            if i == 0:
                state_hist.append([x, v, ang, ang_vel, action])
            #print(f'action = {action}')
            #env.render()
            #action = net.get_action(obs)
            obs, rew, done, info = env.step(action)
            score += rew

        scores.append(score)
        #print(f'Fitness: {score}')

    print('\nmean score: ', np.mean(scores))
    scores_mean.append(np.mean(scores))
    scores_sd.append(np.std(scores))


env.close()


#plt.plot(ang_vel_cutoff_vals, scores_mean, 'o-')
plt.plot(w1_div_w2, scores_mean, 'o-')
plt.xlabel('w1/w2')
plt.ylabel('mean score')
plt.savefig('misc/ang_vel_cutoff_performance_2.png')
plt.show()



exit()


state_hist = np.array(state_hist)
print(state_hist.shape)

labels = ['x', 'v', 'ang', 'ang_vel', 'action']

for i,l in enumerate(labels):

    plt.plot((state_hist[:,i]-np.mean(state_hist[:,i]))/np.std(state_hist[:,i]), label=l)

plt.legend()
plt.show()
