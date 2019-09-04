import numpy as np
import itertools



N_weights = 4
#nonzero_generator = itertools.combinations_with_replacement([-1,1], N_nonzero)
#nonzero_generator = itertools.combinations_with_replacement([-1,1], N_nonzero)
#nz_gen = itertools.product([-1,1], repeat=N_nonzero)

#nonzero_w = next(nz_gen)
#weight_generator = itertools.permutations(list(nonzero_w) + [0]*(N_weights - N_nonzero), N_weights)
#nz_ind_gen = itertools.combinations(range(N_weights), N_nonzero)
counter = 0
all_sets = []

def get_set(nz_tuple, nz_ind_tuple, N_weights):
    arr = np.zeros(N_weights)
    for val,ind in zip(nz_tuple, nz_ind_tuple):
        arr[ind] = val
    return arr


N_nonzero = 0
nz_gen = itertools.product([-1,1], repeat=N_nonzero)
nz_ind_gen = itertools.combinations(range(N_weights), N_nonzero)

nz_tuple = next(nz_gen)
#nz_ind_tuple = next(nz_ind_gen)
#arr = get_set(nz_tuple, nz_ind_tuple, N_weights)
#all_sets.append(arr)
#counter += 1

def get_next_weight_set():
    global nz_gen, nz_ind_gen, nz_tuple, nz_ind_tuple, N_nonzero, N_weights, counter
    try:
        nz_ind_tuple = next(nz_ind_gen)
        arr = get_set(nz_tuple, nz_ind_tuple, N_weights)
        all_sets.append(arr)
        counter += 1
        print(f'\t\t\t=========> {arr}')
        return arr

    except StopIteration:

        try:
            nz_tuple = next(nz_gen)
            nz_ind_gen = itertools.combinations(range(N_weights), N_nonzero)
            #print(f'nz_ind_tuple = {nz_ind_tuple}')
            return get_next_weight_set()
            #counter += 1
            #arr = get_set(nz_tuple, nz_ind_tuple, N_weights)
            #all_sets.append(arr)
            #print(f'\t\t\t=========> {arr}')

        except StopIteration:
            #print('Done with nz_ind_gen!\n')
            if N_nonzero < N_weights:
                N_nonzero += 1
                nz_gen = itertools.product([-1,1], repeat=N_nonzero)
                return get_next_weight_set()
            else:
                return 0


#print(s) for s in all_sets]
for i in range(100):
    get_next_weight_set()
print(f'{counter} total sets')
exit()
get_next_weight_set()




while True:
    try:
        print(next(nonzero_generator))
    except:
        print('Done with nonzero_gen!')
        break


while True:
    try:
        print(next(nonzero_indices_generator))
    except:
        print('Done with nonzero_gen!')
        break

exit()


'''
Want nonzero_gen to produce all N_nonzero-ples -1's and 1's, where order
DOES matter. So for N_nonzero = 2:
(-1, -1),
(-1, 1),
(1, -1),
(1, 1)

Then, I want nonzero_ind_gen to produce all N_nonzero-ples of indices between 0
and N_weights-1, which will be set with the values produced from nonzero_gen.

So for N_nonzero = 2, N_weights = 4:
(0,1),
(0,2),
(0,3),
(1,2),
(1,3),
(2,3),

So this will give 4x6=24 combos.
'''














def get_next_weight_set():
    global weight_generator, nonzero_tuples_tried, nonzero_generator, N_weights, N_nonzero
    try:
        # Try to just get the next weights set
        while True:
            inds = next(weight_generator)
            print(w)
            if w not in nonzero_tuples_tried:
                nonzero_tuples_tried.add(w)
                return w
            else:
                return get_next_weight_set()

    except StopIteration:
        # If it was at the end of that generator, you need to get the next
        # set of nonzero_w
        try:
            # Call the nonzero_generator
            nonzero_w = next(nonzero_generator)
            weight_generator = itertools.permutations(list(nonzero_w) + [0]*(N_weights - N_nonzero), N_weights)
            return get_next_weight_set()
        except StopIteration:
            # If the nonzero_generator is at the end, you need to increase
            # N_nonzero and create it again
            N_nonzero += 1
            print('\n\nNow doing weights with {N_nonzero} nonzero vals')
            if N_nonzero > N_weights:
                print('N_nonzero is bigger than N_weights, weight search done!')
                search_done = True
                return None
            else:
                nonzero_generator = itertools.combinations_with_replacement([-1,1], N_nonzero)
                return get_next_weight_set()



while True:

    get_next_weight_set()


























#
