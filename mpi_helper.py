# from mpi4py import MPI
import math

# similar to orphics.mpi.mpi_distribute except tasks is a list of values
# to be for looped over
# returns a list of subsets of tasks
def distribute(tasks, size):
    per_core, rem = divmod(len(tasks), size)
    all_tasks = [tasks[tasknum * per_core:(tasknum + 1) * per_core]
                 for tasknum in range(size)]

    # add the remainder tasks in reverse so we don't add extra tasks to rank 0
    for r in range(rem):
        all_tasks[-(r+1)] += [tasks[per_core * size + r]]

    return all_tasks
    

