from rubin_scheduler.scheduler.schedulers import BaseQueueManager
from rubin_scheduler.utils import (DEFAULT_NSIDE, SURVEY_START_MJD,
                                   rotation_converter, _approx_ra_dec2_alt_az,
                                   _approx_altaz2pa)
import rubin_scheduler.scheduler.basis_functions as bf
import rubin_scheduler.scheduler.detailers as detailers
from rubin_scheduler.scheduler.model_observatory import ModelObservatory
import numpy as np
from rubin_scheduler.site_models import CloudMap


class RotSkyPosUpdateDetailer(detailers.BaseDetailer):
    """Update the RotSkyPos given the MJD
    """

    def __init__(self, telescope="rubin"):

        self.rc = rotation_converter(telescope=telescope)

    def __call__(self, observation_array, conditions):

        alt, az = _approx_ra_dec2_alt_az(
            observation_array["RA"],
            observation_array["dec"],
            conditions.site.latitude_rad,
            conditions.site.longitude_rad,
            conditions.mjd,
        )
        obs_pa = _approx_altaz2pa(alt, az, conditions.site.latitude_rad)

        observation_array["rotSkyPos"] = self.rc._rottelpos2rotskypos(observation_array["rotTelPos"], obs_pa)
        return observation_array


def generate_qm():

    detailer_list = []
    #detailer_list.append(detailers.RotspUpdateDetailer())
    detailer_list.append(RotSkyPosUpdateDetailer())

    bf_list = []
    # This should get zenith masked
    bf_list.append(bf.SlewtimeBasisFunction())
    # Any clouds that have rolled in
    bf_list.append(bf.MapCloudBasisFunction())

    qm = BaseQueueManager(detailers=detailer_list, basis_functions=bf_list)

    return qm


class CloudyMO(ModelObservatory):

    def return_conditions(self):
        conditions = super().return_conditions()

        cloud_map = CloudMap()
        extinction_map = conditions.az * 0
        extinction_map[np.where(conditions.az < (np.pi/2))] = 5.
        cloud_map.add_frame(extinction_map, conditions.mjd)

        conditions.cloud_maps = cloud_map

        return conditions

