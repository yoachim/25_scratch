Trying out some different codes. Made an env rubin_future to try things out in rebound/assist

Need openorb to do some unit conversions, so let's do a python 11 for that


conda create -n rubin_oo python=3.11
conda activate rubin_oo
conda install -c conda-forge openorb
conda install -c conda-forge sbpy
conda install -c conda-forge notebook
pip install -e . --no-deps # for rubin_sim, and rubin_scheduler
conda install -c conda-forge healpy

