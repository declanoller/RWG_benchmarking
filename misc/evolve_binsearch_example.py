import path_utils
from Evolve import Evolve


#e = Evolve('MountainCar-v0', NN='FFNN_multilayer')
#e = Evolve('CartPole-v0', NN='FFNN', search_method='bin_grid_search')
e = Evolve('MountainCar-v0', NN='FFNN', search_method='sparse_bin_grid_search')
#e = Evolve('MountainCar-v0', NN='FFNN')
evo_dict = e.evolve(20000, N_trials=3)
e.plot_scores(evo_dict, show_plot=True)
e.show_best_episode(evo_dict['best_weights'])
