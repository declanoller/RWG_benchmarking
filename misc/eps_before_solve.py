import matplotlib.pyplot as plt
import numpy as np



'''
P(N(t) = n) = exp(-lambda*t) * ((lambda*t)**n/n!)


We want P(N(t) = 1), to equal, say, 0.5.



'''


frac_solved = 0.0001
P_target = 0.5

N_solve = np.log(1 - P_target)/np.log(1 - frac_solved)

# only want discrete
eps = np.arange(1, 4.0/frac_solved)

# identical, dumber
#plt.plot(eps, np.cumsum(frac_solved*(1 - frac_solved)**(eps - 1)))
plt.plot(eps, 1 - (1 - frac_solved)**eps)


ax = plt.gca()
ax.axhline(P_target, linestyle='dashed', color='gray')
ax.axvline(N_solve, linestyle='dashed', color='gray')
plt.show()
