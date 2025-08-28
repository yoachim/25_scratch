import numpy as np
import healpy as hp
import rubin_sim.maf_proto as maf
from multiprocessing import Pool
import matplotlib.pylab as plt
import datetime

# Should be able to do this then promote things to be globals?
visits_array, df, run_name, subset, fig_saver = maf.batch_preamble(
            observations=None,
            run_name=None,
            quick_test=False,
            fig_saver=None,
        )

good = np.where(visits_array["night"] < 365)
visits_array = visits_array[good]
slicer = maf.Slicer(nside=8)
slicer.setup_slicer(visits_array)
metric = maf.SNNSNMetric()


def call_single_indx(indx):
    result = slicer(visits_array, metric, info=None, indx=[indx], skip_setup=True)
    return result


def launch_jobs(nside=8, num_jobs=6):
    nmap = hp.nside2npix(nside)

    with Pool(processes=num_jobs) as pool:
        result = pool.map(call_single_indx, range(nmap))

    #pool = Pool(processes=num_jobs)
    #jobs = [pool.apply_async(call_single_indx, [[indx]]) for indx in range(nmap)]
    #result = [res.get() for res in jobs]
    result = np.concatenate(result)
    pool.close()
    return result


if __name__ == '__main__':
    nside = 8

    t1 = datetime.datetime.now()

    parallel_results = launch_jobs(nside=nside)
    
    info = maf.empty_info()
    info["run_name"] = run_name
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


