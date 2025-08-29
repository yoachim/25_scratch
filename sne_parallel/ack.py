import multiprocessing
import numpy as np

# will hold the (implicitly mem-shared) data
data_array = None

# child worker function
def job_handler(num):
    # built-in id() returns unique memory ID of a variable
    return id(data_array), np.sum(data_array)

def launch_jobs(data, num_jobs=5, num_worker=4):
    global data_array
    data_array = data

    pool = multiprocessing.Pool(num_worker)
    return pool.map(job_handler, range(num_jobs))


if __name__ == "__main__":



    # create some random data and execute the child jobs
    mem_ids, sumvals = zip(*launch_jobs(np.random.rand(10)))

    import pdb ; pdb.set_trace()

    # this will print 'True' on POSIX OS, since the data was shared
    print(np.all(np.asarray(mem_ids) == id(data_array)))

