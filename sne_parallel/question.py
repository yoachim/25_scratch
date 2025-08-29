import numpy as np
from multiprocessing import Pool


data = np.arange(400).reshape(20, 20)


class MyClass():
    def __call__(self, indx):
        return np.sum(data[:, indx])


my_class = MyClass()


def call_single_indx(indx):
    result = my_class(indx)
    return result


def launch_jobs(nmap=10, num_jobs=3):

    call_single_indx(0)
    with Pool(processes=num_jobs) as pool:
        result = pool.map(call_single_indx, range(nmap))
    result = np.array(result)
    return result


if __name__ == "__main__":

    result = launch_jobs()
    print(result)
