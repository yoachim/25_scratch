import numpy as np
import healpy as hp
import rubin_sim.maf_proto as maf
from multiprocessing import Pool
import matplotlib.pylab as plt
import datetime

DAYS_IN_YEAR = 365.25


def call_single_indx(visits, metric, info, indx, nside):
    slicer = maf.Slicer(nside=nside)
    result, info = slicer(visits, metric, info=info, indx=indx, skip_setup=False)
    return result


def run_sne():
    
    summary_stats = []

    info = maf.empty_info()
    info["run_name"] = run_name
    info["observations_subset"] = subset
    sl = maf.Slicer(nside=nside)
    metric = maf.SNNSNMetric()

    sn_array, info = sl(visits_array, metric, info=info, skip_setup=True)

    pm = maf.PlotMoll(info=info)

    fig = pm(sn_array["n_sn"], unit="N SNe to z limit")
    fig_saver(fig, info=info)
    fig = pm(sn_array["zlim"], unit="z limit")
    fig_saver(fig, info=info)

    summary_stats.append(maf.gen_summary_row(info, "sum N SNe", np.nansum(sn_array["n_sn"])))
    summary_stats.append(maf.gen_summary_row(info, "mean z limit", np.nansum(sn_array["zlim"])))
    summary_stats.append(maf.gen_summary_row(info, "median z limit", np.nanmedian(sn_array["zlim"])))

    return summary_stats


if __name__ == '__main__':


   

    npool = 4
    nside = 4
    nmap = hp.nside2npix(nside)

    observations = None
    run_name = None
    quick_test = False
    fig_saver = None

    visits_array, df, run_name, subset, fig_saver = maf.batch_preamble(
            observations=observations,
            run_name=run_name,
            quick_test=quick_test,
            fig_saver=fig_saver,
        )

    good = np.where(visits_array["night"] < 365)
    visits_array = visits_array[good]


    t1 = datetime.datetime.now()
    info = maf.empty_info()
    info["run_name"] = run_name
    info["observations_subset"] = "Year 1"
    metric = maf.SNNSNMetric()

    pool = Pool(processes=npool)
    jobs = [pool.apply_async(call_single_indx, [visits_array, metric, info, [indx], nside]) for indx in range(nmap)]
    result = [res.get() for res in jobs]
    result = np.concatenate(result)

    pool.close()

    t2 = datetime.datetime.now()
    print("time to run in parallel", t2-t1)
    # XXX--need to manually update the info dict since we went parallel

    # run classic style
    sl = maf.Slicer(nside=nside)
    metric = maf.SNNSNMetric()

    sn_array, info = sl(visits_array, metric, info=info)

    t3 = datetime.datetime.now()
    print("time to run single core", t3-t2)

    assert np.array_equal(sn_array["n_sn"], result["n_sn"])
    assert np.array_equal(sn_array["zlim"], result["zlim"])

    import pdb ; pdb.set_trace()

    # check that things worked properly



