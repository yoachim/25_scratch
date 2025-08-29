import numpy as np
from multiprocessing import Pool

data = None
my_class = None


def set_globals(n1, n2):
    global data
    data = np.arange(n1).reshape(n2, n2)
    global my_class
    my_class = MyClass()


class MyClass():
    def __call__(self, indx):
        return np.sum(data[:, indx])


def call_single_indx(indx):
    result = my_class(indx)
    return result


def launch_jobs(nmap=10, num_jobs=3):
    with Pool(processes=num_jobs) as pool:
        result = pool.map(call_single_indx, range(nmap))
    result = np.array(result)
    return result


if __name__ == "__main__":
    set_globals(n1=400, n2=20)
    launch_jobs()
