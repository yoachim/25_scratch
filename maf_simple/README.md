
Trying out a refactor of MAF that does not use `MetricBundle`, `MetricBundleGroup`, or `PlotHandler`

The main drawback of this:

* Slicers can no longer efficiently pass information to multiple metrics that are running on the same grid with the same sql constriant. So if one wants maps of both coadded depth per filter, and number of visits per filter, there is a factor of ~2 penalty in the refactor (takes 5 min instead of 2). 

Advantages

* Much less code to maintain. Those three classes are monsters
* Can get to a sky map plot in 4 lines! 
* More explicit and direct control of labels. No more long confusing auto-generated dictionary keys for names
* No framework constraints on how things are computed anymore (a metric could even be a labmda function), so easier to make strange new data structures.

