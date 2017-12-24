import numpy as np
import pandas as pd

from configure import *

def abs_max(data):
    """
    Get maximum absolute values in data along the axis=1.

    Inputs:
    - data: A Numpy array.

    Return:
    - A Numpy array, maximum absolute values.
    """
    if data.ndim != 2: # No processing for non-2D array
        return data

    return data[np.arange(data.shape[0]), abs(data).argmax(axis=1)]


def decompose_wind(flight_df):
    """
    Decompose wind data to longitude wind speed and latitude wind speed with
    respect to the airplane axis.

    Input:
    - flight_df: A Pandas dataframe containing columns that _WIND_SPD", _WINDIR
      and _WIND_SPD.

    Returns:
    - wind_data: A Numpy array, of shape (T, 2). T means length of time range.
      It contains 2 columns following:
      - columns 0: longitude wind speed with respect to the airplane.
      - columns 1: latitude wind speed with respect to the airplane.
    """
    wind_long = (-flight_df['_WIND_SPD'] * np.cos((flight_df['_HEADING_LINEAR'] 
                 - flight_df['_WINDIR']) / 180.0 * np.pi)).values
    wind_lati = (-flight_df['_WIND_SPD'] * np.sin((flight_df['_HEADING_LINEAR'] 
                 - flight_df['_WINDIR']) / 180.0 * np.pi)).values
    wind_data = np.c_[wind_long, wind_lati]
    return wind_data


def compose_sstick(flight_df, sstick=False):
    """
    Compose driving values of side-sticks of Capt. and F.O. togeter.


    Input:
    - flight_df: A Pandas dataframe containing columns following:
      _SSTICK_CAPT _SSTICK_CAPT-1 _SSTICK_CAPT-2 _SSTICK_CAPT-3
      _PITCH_CAPT_SSTICK _PITCH_CAPT_SSTICK-1 _PITCH_CAPT_SSTICK-2	
      _PITCH_CAPT_SSTICK-3	
      _ROLL_CAPT_SSTICK _ROLL_CAPT_SSTICK-1 _ROLL_CAPT_SSTICK-2 
      _ROLL_CAPT_SSTICK-3
      _SSTICK_FO _SSTICK_FO-1 _SSTICK_FO-2 _SSTICK_FO-3
      _PITCH_FO_SSTICK _PITCH_FO_SSTICK-1 _PITCH_FO_SSTICK-2 _PITCH_FO_SSTICK-3
      _ROLL_FO_SSTICK _ROLL_FO_SSTICK-1 _ROLL_FO_SSTICK-2 _ROLL_FO_SSTICK-3

    Returns:
    - wind_data: A Numpy array, of shape (T, 2). T means length of time range.
      It contains 2 columns following:
      - columns 0: longitude wind speed with respect to the airplane.
      - columns 1: latitude wind speed with respect to the airplane.
    """
    data = flight_df.values

    if sstick:
        sscp = data[:, SS_CP]
        ssfo = data[:, SS_FO]
    else:
        sscp, ssfo = 1, 1

    p_cp = data[:, PITH_CP]
    p_fo = data[:, PITH_FO]
    r_cp = data[:, ROLL_CP]
    r_fo = data[:, ROLL_FO]

    pith = abs_max(sscp * p_cp + ssfo * p_fo)
    roll = abs_max(sscp * r_cp + ssfo * r_fo)
   
    return np.c_[pith, roll]

