import numpy as np
import healpy as hp
import rubin_sim.maf_proto as maf
from multiprocessing import Pool, Manager
import matplotlib.pylab as plt
import datetime



def call_single_indx(shared_dict, indx):
    result = shared_dict["slicer"](shared_dict["visits_array"], 
                                   shared_dict["metric"], 
                                   info=None, indx=[indx], skip_setup=True)
    return result


def launch_jobs(visits_array, nside=8, num_jobs=3):
    nmap = hp.nside2npix(nside)

    with Manager() as manager:
        shared_dict = manager.dict()
        metric = maf.SNNSNMetric()
        shared_dict["metric"] = metric
        shared_dict["visits_array"] = visits_array
        slicer = maf.Slicer(nside=nside)
        slicer.setup_slicer(visits_array)
        shared_dict["slicer"] = slicer
        with Pool(processes=num_jobs) as pool:
            result = pool.map(shared_dict, call_single_indx, range(nmap))

    #pool = Pool(processes=num_jobs)
    #jobs = [pool.apply_async(call_single_indx, [[indx]]) for indx in range(nmap)]
    #result = [res.get() for res in jobs]
    result = np.concatenate(result)
    pool.close()
    return result


if __name__ == '__main__':
    nside = 8

    visits_array, df, run_name, subset, fig_saver = maf.batch_preamble(
            observations=None,
            run_name=None,
            quick_test=False,
            fig_saver=None,
        )

    good = np.where(visits_array["night"] < 365)
    visits_array = visits_array[good]

    t1 = datetime.datetime.now()

    parallel_results = launch_jobs(visits_array, nside=nside)
    
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


