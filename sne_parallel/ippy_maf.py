# Try out ip

import ipyparallel as ipp
import numpy as np
import rubin_sim.maf_proto as maf
import sqlite3
import pandas as pd

from rubin_sim.data import get_baseline



def call_single_indx(visits_array, metric, slicer, indx):
    
    result = slicer(visits_array,
                    metric, 
                    indx=[indx], skip_setup=True)
    return result



if __name__ == "__main__":

    nside = 4

    observations = get_baseline()
    con = sqlite3.connect(observations)
    df = pd.read_sql("select * from observations where night < 365;", con)
    visits_array = df.to_records(index=False)


    slicer = maf.Slicer(nside=nside)
    slicer.setup_slicer(visits_array)

    metric = maf.MeanMetric(col="airmass")

    with ipp.Cluster(n=4) as rc:
        # get a view on the cluster
        view = rc.load_balanced_view()

        dview = rc[:] # Get a DirectView of all engines

        # put the numpy array on all the 
        dview.execute("global visits_array")
        dview.push(dict(visits_array=visits_array))

        args = [[visits_array, metric, slicer, i] for i in range(len(slicer))]

        # submit the tasks
        asyncresult = view.map_async(call_single_indx, args)
        # wait interactively for results
        asyncresult.wait_interactive()
        # retrieve actual results
        result = asyncresult.get()
    
    print(result)
