import path_utils
from Evolve import Evolve


e = Evolve('CartPole-v0')
evo_dict = e.evolve(1000, N_trials=3)
e.plot_scores(evo_dict, show_plot=True)
e.show_best_episode(evo_dict['best_weights'])
