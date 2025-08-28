import numpy as np
from multiprocessing import Pool, shared_memory, Manager


class SharedNumpyArray:
    # https://e-dorigatti.github.io/python/2020/06/19/multiprocessing-large-objects.html
    def __init__(self, array):
        # create the shared memory location of the same size of the array
        self._shared = shared_memory.SharedMemory(create=True, size=array.nbytes)
        # save data type and shape, necessary to read the data correctly
        self._dtype, self._shape = array.dtype, array.shape
        # create a new numpy array that uses the shared memory we created.
        # at first, it is filled with zeros
        res = np.ndarray(
            self._shape, dtype=self._dtype, buffer=self._shared.buf
        )
        # copy data from the array to the shared memory. numpy will
        # take care of copying everything in the correct format
        np.copyto(res, array)

    def read(self):
        """ Read array without copy.
        """
        return np.ndarray(self._shape, self._dtype, buffer=self._shared.buf)

    def copy(self):
        """Copy arrray
        """
        return np.copy(self.read_array())

    def unlink(self):
        """Unlink when done with data
        """
        self._shared.close()
        self._shared.unlink()


class MyClass():
    def __call__(self, shared_data, indx):
        return np.sum(shared_data.read()[:, indx])


def call_single_indx(shared_data, shared_class, indx):
    result = shared_class(shared_data, indx)
    return result


def launch_jobs(shared_data, nmap=10, num_jobs=3):
    with Manager() as manager:
        manager.shared_class = MyClass()
        args = [[shared_data, manager.shared_class, i] for i in range(nmap)]
        with Pool(processes=num_jobs) as pool:
            result = pool.starmap(call_single_indx, args)
    result = np.array(result)
    return result


if __name__ == "__main__":

    data = np.arange(400).reshape(20, 20)
    shared_data = SharedNumpyArray(data)
    result = launch_jobs(shared_data)
    print(result)

    shared_data.unlink()
