import numpy as np
import healpy as hp
import rubin_sim.maf_proto as maf
from multiprocessing import Pool


def my_func(n1, n2):
    return n1 + n2


if __name__ == '__main__':

    nmap = 5
    npool = 3

    pool = Pool(processes=npool)
    import pdb ; pdb.set_trace()
    results = [pool.apply_async(my_func, [indx, indx+1]) for indx in range(nmap)]
    output = [res.get() for res in results]
