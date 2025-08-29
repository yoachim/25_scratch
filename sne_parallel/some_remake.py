import numpy as np
import healpy as hp
import rubin_sim.maf_proto as maf
from multiprocessing import Pool, shared_memory
import matplotlib.pylab as plt
import datetime
import sqlite3
import pandas as pd

from rubin_sim.data import get_baseline


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


# Should be able to do this then promote things to be globals?
# or find out how to share visits, slicer, metric to
# each process


# Looks like this gets run by each process. So that's not the most 
# efficent way to do things clearly. But, it is only getting done once.


# set up metric and slicer. 
#slicer = maf.Slicer(nside=8)
#slicer.setup_slicer(visits_array)
#metric = maf.SNNSNMetric()

def set_globals(visit_array, nside=8):
    global slicer
    slicer = maf.Slicer(nside=nside)
    slicer.setup_slicer(visit_array)
    global metric
    metric = maf.SNNSNMetric()


def call_single_indx(shared_data, indx, nside=8):

    # If this is the first call, set up a slicer and metric
    # just for this processor
    if "slicer" not in globals():
        set_globals(shared_data.read(), nside=nside)
    result = slicer(shared_data.read(),
                    metric,
                    info=None,
                    indx=[indx],
                    skip_setup=True)
    return result


def launch_jobs(shared_data, nside=8, num_jobs=6):
    npix = hp.nside2npix(nside)

    args = [[shared_data, i] for i in range(npix)]
    with Pool(processes=num_jobs) as pool:
        result = pool.starmap(call_single_indx, args)

    #pool = Pool(processes=num_jobs)
    #jobs = [pool.apply_async(call_single_indx, [[indx]]) for indx in range(nmap)]
    #result = [res.get() for res in jobs]
    result = np.concatenate(result)
    pool.close()
    return result


if __name__ == '__main__':
    nside = 8

    # Read in some visits
    observations = get_baseline()
    con = sqlite3.connect(observations)
    df = pd.read_sql("select * from observations where night < 365;", con)
    visits_array = df.to_records(index=False)

    shared_data = SharedNumpyArray(visits_array)


    t1 = datetime.datetime.now()

    parallel_results = launch_jobs(shared_data, nside=nside)
    
    info = maf.empty_info()
    info["run_name"] = "temp"
    info["observations_subset"] = "Year 1"

    t2 = datetime.datetime.now()
    print("time to run in parallel", t2-t1)
    # XXX--need to manually update the info dict since we went parallel and dropped that

    # run classic style
    sl = maf.Slicer(nside=nside)
    metric = maf.SNNSNMetric()

    sn_array, info = sl(visits_array, metric, info=info)

    t3 = datetime.datetime.now()
    print("p2, time to run single core", t3-t2)

    import pdb ; pdb.set_trace()

    assert np.allclose(sn_array["n_sn"], parallel_results["n_sn"], equal_nan=True)
    assert np.allclose(sn_array["zlim"], parallel_results["zlim"], equal_nan=True)


