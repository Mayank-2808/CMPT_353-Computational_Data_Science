import time
import pandas as pd
import numpy as np
from implementations import all_implementations

def make_random_array(size):
    return np.random.randint(0, 10000, size=size)

def calc_sorting_time(sort_function, arr):
    
    start_time = time.time()
    _ = sort_function(arr)
    end_time = time.time()
    return end_time - start_time

def main():
    data = {'Implementation': [], 'Array Size': [], 'Sorting Time(s)': []}
    array_sizes = [100, 1000, 5000, 10000, 50000]

    for array_size in array_sizes:
        for sort_function in all_implementations:
            for _ in range(10):
                random_array = make_random_array(array_size)
                sorting_time = calc_sorting_time(sort_function, random_array)
                data['Implementation'].append(sort_function.__name__)
                data['Array Size'].append(array_size)
                data['Sorting Time(s)'].append(sorting_time)

    df = pd.DataFrame(data)
    df.to_csv('data.csv', index=False)

if __name__ == '__main__':
    main()
