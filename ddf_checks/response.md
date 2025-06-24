

* The DDFs should be centered at the positions listed in Sec 3.2 of Br18.

Done. Has been since at least v3.5 sims. Feel free to make a PR here if there should be changes to the DDF locations: https://github.com/lsst/rubin_scheduler/blob/main/rubin_scheduler/utils/ddf_locations.py

* DDF observations should minimize nightly lunar sky-brightness effects to the extent reasonably possible (see "Cadence" of Br21).

We (assuming a nominal value for atmospheric seeing) compute the 5-sigma limiting depth for each field in 15 min timesteps for the entire survey. We set a minimum depth to avoid scheduling observations in particularly bad times. Once a night has been selected as good for a DDF sequence, we schedule it for the best depth. The timing can be shifted slightly from that best time to avoid extra filter changes (e.g., we shift XMM observations to a slightly different time so we can take r band observations in XMM and ECDFS sequentially).

* DDF observations should extend over the very longest observing seasons possible (7-8.5 months; see Br18 and Br21). An accordion cadence approach could be used to enable this at lower cost, if needed.

We are currently setting the season length to 225 days. Spot checking, COSMOS has one 232 day season. Going longer seems impracticle, usually because of lunar phase issues. Feel free to play with this notebook to look at season length limits:  https://github.com/lsst-sims/sims_featureScheduler_runs4.3/blob/main/ddf_ocean/season_check.ipynb

* To mitigate weather or other losses, if no observations can be made in a given night then we request elevated priority for the next night to make up the loss (see Sec 3.7 of Br18).

DDF observations are already one of the highest priority tiers in the scheduler and should only be preemted by ToO triggers, or Roman field observations (which seems unlikely since the Roman fields are in the bulge). 

