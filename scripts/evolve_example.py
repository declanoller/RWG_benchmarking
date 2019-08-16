import path_utils
from Evolve import Evolve

e = Evolve('CartPole-v0', NN='FFNN', N_hidden_layers=0)
evo_dict = e.evolve(2000, N_trials=20, print_gen=True)
e.save_all_evo_stats(evo_dict)
#e = Evolve('MountainCar-v0', NN='FFNN_multilayer', N_hidden_layers=1)
#e = Evolve('Pendulum-v0', NN='FFNN_multilayer', N_hidden_layers=1, search_method='sparse_bin_grid_search')
#e = Evolve('Pendulum-v0', NN='FFNN_multilayer', N_hidden_layers=1, N_hidden_units=2, search_method='RWG')
#e = Evolve('MountainCar-v0', NN='FFNN_multilayer', N_hidden_layers=0, search_method='RWG')
#e = Evolve('MountainCar-v0', NN='FFNN_multilayer', N_hidden_layers=0, search_method='sparse_bin_grid_search')
#e = Evolve('MountainCar-v0', NN='FFNN')

exit()
