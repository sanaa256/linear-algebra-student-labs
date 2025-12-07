from generate_safe_system import *
from lu_factorisation import *
from f_and_b_subs import *

import numpy as np
import time

sizes = [2**j for j in range(1, 6)]

# list of dicts. each dict has keys: size, fs_time, bs_time, total_time
results = []

n_repeats = 10

for n in sizes:
    # generate a random system of linear equations of size n
    A, b, x = generate_safe_system(n)
    fs_total_time = 0
    bs_total_time = 0

    start_time = time.perf_counter()
    for i in range(n_repeats):
        # do the solve
        L, U = lu_factorisation(A)

        # solve L*z* = *b*
        fs_start_time = time.perf_counter()
        z = forward_substitution(L, b)
        fs_end_time = time.perf_counter()

        fs_total_time += fs_end_time - fs_start_time

        # solve U*x* = *z*
        bs_start_time = time.perf_counter()
        x = backward_substitution(U, z)
        bs_end_time = time.perf_counter()

        bs_total_time += bs_end_time - bs_start_time

    end_time = time.perf_counter()
    
    avg_total_time = (end_time - start_time) / n_repeats
    avg_fs_time = fs_total_time / n_repeats
    avg_bs_time = bs_total_time / n_repeats

    results.append( 
        {
            "size": n,
            "fs_time": avg_fs_time,
            "bs_time": avg_bs_time,
            "total_time": avg_total_time,
        }
    )

print(results)