from generate_safe_system import *
from lu_factorisation import *
from f_and_b_subs import *

import numpy as np
import time
import matplotlib.pyplot as plt

sizes = [2 ** j for j in range(1, 10)]

# List of dicts. Each dict has keys: size, lu_time, fs_time, bs_time, total_time
results = []

n_repeats = 10

for n in sizes:
    # Generate a random system of linear equations of size n
    A, b, x = generate_safe_system(n)

    # Total time across all repeats at size n
    fs_total_time = 0
    bs_total_time = 0
    lu_total_time = 0

    start_time = time.perf_counter()
    for i in range(n_repeats):
        lu_start_time = time.perf_counter()
        L, U = lu_factorisation(A)
        lu_end_time = time.perf_counter()

        lu_total_time += lu_end_time - lu_start_time

        # Solve L*z* = *b*
        fs_start_time = time.perf_counter()
        z = forward_substitution(L, b)
        fs_end_time = time.perf_counter()

        fs_total_time += fs_end_time - fs_start_time

        # Solve U*x* = *z*
        bs_start_time = time.perf_counter()
        x = backward_substitution(U, z)
        bs_end_time = time.perf_counter()

        bs_total_time += bs_end_time - bs_start_time

    end_time = time.perf_counter()
    
    avg_total_time = (end_time - start_time) / n_repeats
    avg_fs_time = fs_total_time / n_repeats
    avg_bs_time = bs_total_time / n_repeats
    avg_lu_time = lu_total_time / n_repeats

    results.append( 
        {
            "size": n,
            "lu_time": avg_lu_time,
            "fs_time": avg_fs_time,
            "bs_time": avg_bs_time,
            "total_time": avg_total_time,
        }
    )

# Extract times for plotting
lu_times = [result["lu_time"] for result in results]
fs_times = [result["fs_time"] for result in results]
bs_times = [result["bs_time"] for result in results]
total_times = [result["total_time"] for result in results]

marker_size = 5
plt.plot(sizes, lu_times, "-o", color = "green", markersize = marker_size, label = "LU factorisation")
plt.plot(sizes, fs_times, "-o", color = "red", markersize = marker_size, label = "Forward substitution")
plt.plot(sizes, bs_times, "-o", color = "blue", markersize = marker_size, label = "Backward substitution")
plt.plot(sizes, total_times, "-o", color = "black", markersize = marker_size, label = "Total")

plt.xscale('log')
plt.yscale('log')
plt.title("Runtime with LU factorisation")
plt.xlabel("n")
plt.ylabel("time (s)")
plt.legend(loc = "upper left")
plt.grid()

plt.show()
