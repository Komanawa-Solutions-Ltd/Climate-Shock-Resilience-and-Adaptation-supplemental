"""
created matt_dumont 
on: 12/07/22
"""
from pathlib import Path
from numbers import Number
from collections.abc import Iterable
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

#todo this is wrong!!!!, need to re-calculate the data including all of the scenarios (100/storyline)
def calc_non_exceedence_prob(pasture_growth, mod: bool, site: str, mode: str, pg_in_tons=False):
    """
    calculate the probability of having as much or more pasture production as the specified data
    :param pasture_growth: annual pasture growth in kg DM/ha/year or tons DM/ha/year (single value or iterable)
    :param mod: bool If true then the farm consultants corrections are applied, if False then use the raw BASGRA
                data
    :param pg_in_tons: bool if True then expects pasture growth data in tons if false then expects it in kg
    :return: float or np.array (shape of pasture_growth) of probability in percent (0-100)
    """
    site = site.lower()
    mode = mode.lower()
    base_dir = Path(__file__).parent.joinpath('exceedence_datasets')
    return_num = False
    if isinstance(pasture_growth, Number):
        return_num = True
    pasture_growth = np.atleast_1d(pasture_growth)

    if not np.issubdtype(pasture_growth.dtype, np.number):
        raise ValueError('input must be a number or numeric datatype')

    if mod:
        mod_key = 'mod'
    else:
        mod_key = 'raw'
    use_path = base_dir.joinpath(f"{site}-{mode}-{mod_key}_1yr_cumulative_exceed_prob.csv")
    exceedence = pd.read_csv(use_path)
    probs = exceedence.prob.values * 100
    impacts = exceedence.pg.values * 1000
    predictor = interp1d(impacts, probs, fill_value='extrapolate')
    if pg_in_tons:
        out = 100 - predictor(pasture_growth.astype(float) * 1000)
    else:
        out = 100 - predictor(pasture_growth.astype(float))
    if return_num:
        return out[0]
    else:
        return out


def calc_exceedence_prob(pasture_growth, mod: bool, site: str, mode: str, pg_in_tons=False):
    """
    calculate the probability of having as much or less pasture production as the specified data
    :param pasture_growth: annual pasture growth in kg DM/ha/year or tons DM/ha/year (single value or iterable)
    :param mod: bool If true then the farm consultants corrections are applied, if False then use the raw BASGRA
                data
    :param pg_in_tons: bool if True then expects pasture growth data in tons if false then expects it in kg
    :return: float or np.array (shape of pasture_growth) of probability in percent (0-100)
    """
    return 100 - calc_non_exceedence_prob(pasture_growth=pasture_growth, mod=mod, site=site, mode=mode,
                                      pg_in_tons=pg_in_tons)

