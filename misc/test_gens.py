import itertools, time

N = 8
gen = itertools.permutations([-1]*N + [1]*N, N)

already_seen = set([])
N_unique = 0
counter = 0
unique_lim = 500
start = time.time()

while N_unique < unique_lim:
    counter += 1
    try:
        t = next(gen)
        #print(t)
    except:
        print(f'broke with {N_unique} unique ones!')
        break
    if t not in already_seen:
        print(f'adding {t} to already_seen')
        already_seen.add(t)
        N_unique += 1

print(f'counter = {counter}')
end = time.time()

print(f'Took {end - start:.4f} seconds to find {unique_lim} ones!')
