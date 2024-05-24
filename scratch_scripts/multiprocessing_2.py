# This was an experiment to see the effects of different numbers of processes on calculating all prime numbers in a
# range

import multiprocessing as mp
import os
import pdb
import time
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


def all_primes_in_range(lower, upper):
    """
    Calculates all prime numbers in range lower, upper
    :param lower: lower range
    :param upper: upper range
    :return: list of prime numbers
    """
    all_primes = []
    for i in range(lower, upper):
        if all([i % j != 0 for j in range(2, int(i**.5)+1)]):
            all_primes.append(i)

    return all_primes


def all_primes_multiprocess(upper_range, num_processes):
    """
    Calculates all primes from 1 to upper_range using multiprocessing
    :param upper_range: upper range to look for primes
    :param num_processes: number of processes to use
    :return: total time
    """
    # dividing into sublists
    size_of_sublist = upper_range // num_processes
    sublist_idxs = [(i, i + size_of_sublist) for i in range(0, upper_range, size_of_sublist)]

    start = time.time()

    processes = []

    for i in range(num_processes):
        lower, upper = sublist_idxs[i]
        p = mp.Process(target=all_primes_in_range, args=(lower, upper,))
        p.start()
        processes.append(p)

    # if you join them within the previous loop then each pass thru the loop waits, so its the same as non-parallel
    for p in processes:
        p.join()

    total_time = time.time() - start
    print(f'time for prime number task with {num_processes} processes: {total_time}')

    return total_time


def main():
    upper_range = 1_000_000

    # running an experiment to see effect of different numbers of processes
    max_num_processes = 20  # will run experiments for numbers of processes in range(1, max_num_processes)

    all_process_times = []
    for i in range(1, max_num_processes+1):
        time_for_process = all_primes_multiprocess(upper_range, i)
        all_process_times.append(time_for_process)

    # creating a figure
    plt.figure()
    sns.lineplot(x=range(1, max_num_processes+1), y=all_process_times)
    plt.xticks(range(1, max_num_processes+1))

    # adding a red dashed line to show the number of CPU cores I have
    plt.axvline(x=8, color='r', linestyle='--')
    plt.annotate(text='number of cpu cores', xy=(os.cpu_count(), max(all_process_times)/2), rotation=90, color='r')

    plt.xlabel('number of processes')
    plt.ylabel('total time (seconds)')
    plt.title('Performance effects of multiprocessing on prime number task')
    plt.savefig(f'prime_number_task_multiprocessing_experiment_{datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}.png',
                dpi=300)


if __name__ == '__main__':
    main()
