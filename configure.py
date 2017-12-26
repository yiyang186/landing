import re, platform, os

myplatform = platform.platform()
if 'Windows' in myplatform:
    PATH = 'G:/A320_300_20/'
if 'Darwin' in myplatform:
    PATH = '/Users/pyy/Documents/Data/A320_300_20/'
    if '.DS_Store' in os.listdir(PATH):
        os.remove(PATH + '.DS_Store')

RAW_COLUMNS = """
_ALTITUDE	_GS	_DRIFT	_HEADING_LINEAR	_GLIDE	_LOC	_LONG_ACC	_LONG_ACC-1	
_LONG_ACC-2	_LONG_ACC-3	_LAT_ACC	_LAT_ACC-1	_LAT_ACC-2	_LAT_ACC-3	_VRTG	
_VRTG-1	_VRTG-2	_VRTG-3	_VRTG-4	_VRTG-5	_VRTG-6	_VRTG-7	_PITCH	_PITCH-1	
_PITCH-2	_PITCH-3	_ROLL	_ROLL-1	_LDG_STATUS	_LDG_STATUS-1	
_LDG_STATUS-2	_LDG_STATUS-3	_IVV	_SSTICK_CAPT	_SSTICK_CAPT-1	
_SSTICK_CAPT-2	_SSTICK_CAPT-3	_PITCH_CAPT_SSTICK	_PITCH_CAPT_SSTICK-1	
_PITCH_CAPT_SSTICK-2	_PITCH_CAPT_SSTICK-3	_ROLL_CAPT_SSTICK	
_ROLL_CAPT_SSTICK-1	_ROLL_CAPT_SSTICK-2	_ROLL_CAPT_SSTICK-3	_SSTICK_FO	
_SSTICK_FO-1	_SSTICK_FO-2	_SSTICK_FO-3	_PITCH_FO_SSTICK	
_PITCH_FO_SSTICK-1	_PITCH_FO_SSTICK-2	_PITCH_FO_SSTICK-3	_ROLL_FO_SSTICK	
_ROLL_FO_SSTICK-1	_ROLL_FO_SSTICK-2	_ROLL_FO_SSTICK-3	_DUAL	
_DUAL-1	_DUAL-2	_DUAL-3	_ENG_THR_VAL_1	_WIND_SPD	_WINDIR
"""

COLUMNS = ['FRAME'] + [s.strip() for s in re.split(r'\s', RAW_COLUMNS)
                       if s is not '']

 # 3 cols
SINGLE_COL = [2, 3, 33]

# 3 cols
ACC_LNG = slice(7, 11)
ACC_LAT = slice(11, 15)
ACC_VRT = slice(15, 23)

# 2 cols
PITH = slice(23, 27)
ROLL = slice(27, 29)

# 2cols
SS_CP = slice(34, 38)
PITH_CP = slice(38, 42)
ROLL_CP = slice(42, 46)

SS_FO = slice(46, 50)
PITH_FO = slice(50, 54)
ROLL_FO = slice(54, 58)

# the number of columns that you need
# the sum of cols above and 2 cols for wind
COL_NUM = 12

DRV = slice(34, 58)


# useless
LDG = slice(29, 33)
