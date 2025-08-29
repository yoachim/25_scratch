import numpy as np

from multiprocessing import Pool, shared_memory, Manager
import rubin_sim.maf_proto as maf
import sqlite3
import pandas as pd
from itertools import repeat

from rubin_sim.data import get_baseline

metric = maf.MeanMetric(col="airmass")


class SharedNumpyArray:
    # from https://e-dorigatti.github.io/python/2020/06/19/multiprocessing-large-objects.html
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


def call_single_indx(shared_data, slice_i):

    print(slice_i["slice_point"]["sid"])
    if len(slice_i["idxs"]) != 0:
        #result = np.atleast_1d(5.)
        #result = np.atleast_1d(np.mean(shared_data.read()["airmass"][slice_i["idxs"]]))
        result = np.atleast_1d(metric(shared_data.read()[slice_i["idxs"]]))
        #result = np.atleast_1d(metric(shared_data.read()[slice_i["idxs"]],
                                      #slice_point=slice_i["slice_point"]))
    else:
        result = np.atleast_1d(np.nan) #np.atleast_1d(metric.badval)
    return result


def launch_jobs(shared_data, slicer, metric, num_jobs=6):

    with Manager() as manager:
        manager.shared_metric = metric

        with Pool(processes=num_jobs) as pool:
            result = pool.starmap(call_single_indx, zip(repeat(shared_data), slicer))
    result = np.concatenate(result)
    return result


if __name__ == "__main__":

    nside = 4
    fast_metric = True

    # Read in observations
    # Read in some visits
    observations = get_baseline()
    con = sqlite3.connect(observations)
    df = pd.read_sql("select * from observations where night < 365;", con)
    visits_array = df.to_records(index=False)

    shared_data = SharedNumpyArray(visits_array)
    slicer = maf.Slicer(nside=nside)
    slicer.setup_slicer(visits_array)
    # Use faster metric for testing
    if fast_metric:
        metric = maf.MeanMetric(col="airmass")
    else:
        metric = maf.SNNSNMetric()

    result = launch_jobs(shared_data, slicer, metric)
    shared_data.unlink()

    import pdb ; pdb.set_trace()

