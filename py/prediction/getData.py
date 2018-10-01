import pandas as pd
import numpy as np
from util import *
from datetime import datetime
import re
from sklearn.feature_extraction.text import CountVectorizer
import pytz

# Get data
# OUTPUT: raw esdr and smell data
def getData(
    out_p=None, # output file path
    start_dt=None, # starting date for the data
    end_dt=None, # ending data for the data
    logger=None):

    log("Get data...", logger)

    # Get and save ESDR data
    # Feed 26: Lawrenceville ACHD
    # Feed 28: Liberty ACHD
    # Feed 23: Flag Plaza ACHD
    # Feed 43 and 11067: Parkway East ACHD
    # Feed 1: Avalon ACHD
    # Feed 27: Lawrenceville 2 ACHD
    # Feed 29: Liberty 2 ACHD
    # Feed 3: North Braddock ACHD
    # Feed 3506: BAPC 301 39TH STREET BLDG AirNow
    # Feed 5975: Parkway East AirNow
    # Feed 3508: South Allegheny High School AirNow
    # Feed 24: Glassport High Street ACHD
    esdr_source_names = [
        "Feed_1_Avalon_ACHD_PM",
        "Feed_1_Avalon_ACHD_others",
        "Feed_26_Lawrenceville_ACHD",
        "Feed_27_Lawrenceville_2_ACHD",
        "Feed_28_Liberty_ACHD",
        "Feed_29_Liberty_2_ACHD",
        "Feed_3_North_Braddock_ACHD",
        "Feed_23_Flag_Plaza_ACHD",
        "Feed_43_and_Feed_11067_Parkway_East_ACHD",
        "Feed_3506_BAPC_301_39TH_STREET_BLDG_AirNow",
        "Feed_5975_Parkway_East_AirNow",
        "Feed_3508_South_Allegheny_High_School_AirNow",
        "Feed_24_Glassport_High_Street_ACHD"
    ]
    esdr_source = [
        [
            {"feed": "1", "channel": "PM25B_UG_M3"},
            {"feed": "1", "channel": "PM25T_UG_M3"}
        ],
        [{"feed": "1", "channel": "SO2_PPM,H2S_PPM,SIGTHETA_DEG,SONICWD_DEG,SONICWS_MPH"}],
        [{"feed": "26", "channel": "OZONE_PPM,PM25B_UG_M3,PM10B_UG_M3,SONICWS_MPH,SONICWD_DEG,SIGTHETA_DEG"}],
        [{"feed": "27", "channel": "NO_PPB,NOY_PPB,CO_PPB,SO2_PPB"}],
        [{"feed": "28", "channel": "H2S_PPM,SO2_PPM,SIGTHETA_DEG,SONICWD_DEG,SONICWS_MPH"}],
        [{"feed": "29", "channel": "PM10_UG_M3,PM25_UG_M3"}],
        [{"feed": "3", "channel": "SO2_PPM,SONICWD_DEG,SONICWS_MPH,SIGTHETA_DEG,PM10B_UG_M3"}],
        [{"feed": "23", "channel": "CO_PPM,PM10_UG_M3"}],
        [
            {"feed": "11067", "channel": "CO_PPB,NO2_PPB,NOX_PPB,NO_PPB,PM25T_UG_M3,SIGTHETA_DEG,SONICWD_DEG,SONICWS_MPH"},
            {"feed": "43", "channel": "CO_PPB,NO2_PPB,NOX_PPB,NO_PPB,PM25T_UG_M3,SIGTHETA_DEG,SONICWD_DEG,SONICWS_MPH"}
        ],
        [{"feed": "3506", "channel": "PM2_5,OZONE"}],
        [{"feed": "5975", "channel": "PM2_5"}],
        [{"feed": "3508", "channel": "PM2_5"}],
        [{"feed": "24", "channel": "PM10_UG_M3"}]
    ]
    start_time = datetimeToEpochtime(start_dt) / 1000 # ESDR uses seconds
    end_time = datetimeToEpochtime(end_dt) / 1000 # ESDR uses seconds
    df_esdr_array_raw = getEsdrData(esdr_source, start_time=start_time, end_time=end_time)
    
    # Get smell reports
    df_smell_raw = getSmellReports(start_time=start_time, end_time=end_time)
    
    # Check directory and save file
    if out_p is not None:
        for p in out_p: checkAndCreateDir(p)
        for i in range(len(df_esdr_array_raw)):
            df_esdr_array_raw[i].to_csv(out_p[0] + esdr_source_names[i] + ".csv")
        df_smell_raw.to_csv(out_p[1])
        log("Raw ESDR data created at " + out_p[0], logger)
        log("Raw smell data created at " + out_p[1], logger)
    return df_esdr_array_raw, df_smell_raw
