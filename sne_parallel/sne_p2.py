import numpy as np
import healpy as hp
import rubin_sim.maf_proto as maf
from multiprocessing import Pool
import matplotlib.pylab as plt
import datetime
import sqlite3
import pandas as pd

from rubin_sim.data import get_baseline



# Looks like this gets run by each process. So that's not the most 
# efficent way to do things clearly. But, it is only getting done once.

# Read in some visits
observations = get_baseline()
con = sqlite3.connect(observations)
df = pd.read_sql("select * from observations where night < 365;", con)
visits_array = df.to_records(index=False)

# set up metric and slicer. 
slicer = maf.Slicer(nside=8)
slicer.setup_slicer(visits_array)
metric = maf.SNNSNMetric()


def call_single_indx(indx):
    result = slicer(visits_array, metric, indx=[indx], skip_setup=True)
    return result


def launch_jobs(nside=8, num_jobs=4):
    nmap = hp.nside2npix(nside)

    with Pool(processes=num_jobs) as pool:
        result = pool.map(call_single_indx, range(nmap))
    result = np.concatenate(result)
    pool.close()
    return result


if __name__ == '__main__':
    nside = 8

    t1 = datetime.datetime.now()

    parallel_results = launch_jobs(nside=nside)

    t2 = datetime.datetime.now()
    print("time to run in parallel", t2-t1)


    # XXX--need to manually update the info dict since we went parallel and dropped that

    # run classic style
    sl = maf.Slicer(nside=nside)
    metric = maf.SNNSNMetric()

    sn_array = sl(visits_array, metric)

    t3 = datetime.datetime.now()
    print("p2, time to run single core", t3-t2)

    
    # Check that parallel results match
    assert np.allclose(sn_array["n_sn"], parallel_results["n_sn"], equal_nan=True)
    assert np.allclose(sn_array["zlim"], parallel_results["zlim"], equal_nan=True)


