import os, sys, functools, time
from datetime import datetime

'''
For adding the dirs to the system path, so we can reference
classes from wherever we run a script. Import this first for any
file you want to use it for.

src/ classes will only be called from scripts, so assume that this has
already been called and dirs added to the system path.
'''


ROOT_DIR = os.path.join(os.path.dirname(__file__), '../')
SRC_DIR = os.path.join(ROOT_DIR, 'src')
NN_DIR = os.path.join(ROOT_DIR, 'NN')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')

sys.path.append(ROOT_DIR)
sys.path.append(SRC_DIR)
sys.path.append(NN_DIR)

def get_output_dir():
    return OUTPUT_DIR

def get_src_dir():
    return SRC_DIR

def timer(func):
    # Print the runtime of the decorated function
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        print(f'\n\nFinished function {func.__name__!r} in {run_time:.2f} secs\n')
        return value
    return wrapper_timer


def get_date_str():
    # Returns the date and time for labeling output.
	return datetime.now().strftime('%d-%m-%Y_%H-%M-%S')


#
