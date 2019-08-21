import ray
import time
import psutil

print(f'there are {psutil.cpu_count()} CPUs')


ray.init(num_cpus=4, ignore_reinit_error=True, include_webui=False)

time.sleep(2.0)

def cool_fn(x):
    return x*2

@ray.remote
def slow_function(i, j):
    i = cool_fn(i)
    return i

start_time = time.time()

#result_IDs = [slow_function.remote(i) for i in range(6)]
result_IDs = []
test_dict = {}
for i in range(6):
    print(f'Iteration {i} of loop')
    id = slow_function.remote(i, i**2)
    #result_IDs.append(id)
    test_dict[i] = id
    print(f'end of iteration {i}')
#print(result_IDs)
results = {i:ray.get(id) for i,id in test_dict.items()}

end_time = time.time()
duration = end_time - start_time

print('The results are {}. This took {} seconds.'.format(results, duration))
print('results: ', results)









#
