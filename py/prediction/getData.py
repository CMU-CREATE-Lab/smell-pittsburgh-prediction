"""
Get smell and sensor data for training the prediction model
"""


from util import log, datetimeToEpochtime, getEsdrData, getSmellReports, checkAndCreateDir


def getData(out_p=None, start_dt=None, end_dt=None, region_setting=0, logger=None):
    """
    Get and save smell and ESDR data

    Input:
        out_p: the path for storing ESDR and smell data (optional)
        start_dt (datetime.datetime object): starting date that you want to get the data
        end_dt (datetime.datetime object): ending date that you want to get the data
        region_setting: setting of the region that we want to get the smell reports
        logger: the python logger created by the generateLogger() function

    Output:
        df_esdr_array_raw (list of pandas.DataFrame): a list of raw ESDR data for each channel
        df_smell_raw (pandas.DataFrame): raw smell data
    """
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
    # Feed 11067: Parkway East Near Road ACHD
    # Feed 59665: Pittsburgh ACHD
    # Feed 26086: Clariton ACHD
    esdr_source_names = [
        "Feed_1_Avalon_ACHD_PM25",
        "Feed_1_Avalon_ACHD",
        "Feed_26_Lawrenceville_ACHD",
        "Feed_26_and_Feed_59665_Lawrenceville_ACHD_PM25",
        "Feed_26_and_Feed_59665_Lawrenceville_ACHD_PM10",
        "Feed_27_Lawrenceville_2_ACHD",
        "Feed_28_Liberty_ACHD",
        "Feed_29_Liberty_2_ACHD_PM10",
        "Feed_29_Liberty_2_ACHD_PM25",
        "Feed_3_North_Braddock_ACHD",
        "Feed_3_North_Braddock_ACHD_PM10",
        "Feed_23_Flag_Plaza_ACHD_CO",
        "Feed_23_Flag_Plaza_ACHD_PM10",
        "Feed_43_and_Feed_11067_Parkway_East_ACHD_DATA1",
        "Feed_43_and_Feed_11067_Parkway_East_ACHD_DATA2",
        "Feed_3506_BAPC_301_39TH_STREET_BLDG_AirNow",
        "Feed_5975_Parkway_East_AirNow",
        "Feed_3508_South_Allegheny_High_School_AirNow",
        "Feed_24_Glassport_High_Street_ACHD",
        "Feed_26086_Clairton_ACHD"
    ]
    # IMPORTANT: if you add more data that changes the length of the esdr_source list,
    # ...you need to also add a name to the corresponding index of the esdr_source_names list.
    esdr_source = [
        [
            {"feed": "1", "channel": "PM25B_UG_M3"},
            {"feed": "1", "channel": "PM25T_UG_M3"},
            {"feed": "1", "channel": "PM25_640_UG_M3"}
        ], # Feed_1_Avalon_ACHD_PM25
        [
            {"feed": "1", "channel": "SO2_PPM,H2S_PPM,SIGTHETA_DEG,SONICWD_DEG,SONICWS_MPH"}
        ], # Feed_1_Avalon_ACHD
        [
            {"feed": "26", "channel": "OZONE_PPM,SONICWS_MPH,SONICWD_DEG,SIGTHETA_DEG"}
        ], # Feed_26_Lawrenceville_ACHD
        [
            {"feed": "26", "channel": "PM25B_UG_M3"},
            {"feed": "26", "channel": "PM25T_UG_M3"},
            {"feed": "59665", "channel": "PM25_640_UG_M3"}
        ], # Feed_26_and_Feed_59665_Lawrenceville_ACHD_PM25
        [
            {"feed": "26", "channel": "PM10B_UG_M3"},
            {"feed": "59665", "channel": "PM10_640_UG_M3"}
        ], # Feed_26_and_Feed_59665_Lawrenceville_ACHD_PM10
        [
            {"feed": "27", "channel": "NO_PPB,NOY_PPB,CO_PPB,SO2_PPB"}
        ], # Feed_27_Lawrenceville_2_ACHD
        [
            {"feed": "28", "channel": "H2S_PPM,SO2_PPM,SIGTHETA_DEG,SONICWD_DEG,SONICWS_MPH"}
        ], # Feed_28_Liberty_ACHD
        [
            {"feed": "29", "channel": "PM10_UG_M3"},
            {"feed": "29", "channel": "PM10_640_UG_M3"}
        ], # Feed_29_Liberty_2_ACHD_PM10
        [
            {"feed": "29", "channel": "PM25_UG_M3"},
            {"feed": "29", "channel": "PM25T_UG_M3"},
            {"feed": "29", "channel": "PM25_640_UG_M3"}
        ], # Feed_29_Liberty_2_ACHD_PM25
        [
            {"feed": "3", "channel": "SO2_PPM,SONICWD_DEG,SONICWS_MPH,SIGTHETA_DEG"}
        ], # Feed_3_North_Braddock_ACHD
        [
            {"feed": "3", "channel": "PM10B_UG_M3"},
            {"feed": "3", "channel": "PM10_640_UG_M3"}
        ], # Feed_3_North_Braddock_ACHD_PM10
        [
            {"feed": "23", "channel": "CO_PPM"},
            {"feed": "23", "channel": "CO_PPB", "factor": 0.001}
        ], # Feed_23_Flag_Plaza_ACHD_CO
        [
            {"feed": "23", "channel": "PM10_UG_M3"}
        ], # Feed_23_Flag_Plaza_ACHD_PM10
        [
            {"feed": "43", "channel": "CO_PPB,SIGTHETA_DEG,SONICWD_DEG,SONICWS_MPH"},
            {"feed": "11067", "channel": "CO_PPB,SIGTHETA_DEG,SONICWD_DEG,SONICWS_MPH"}
        ], # Feed_43_and_Feed_11067_Parkway_East_ACHD_DATA1
        [
            {"feed": "43", "channel": "PM25T_UG_M3,NO2_PPB,NOX_PPB,NO_PPB"},
            {"feed": "11067", "channel": "PM25T_UG_M3,NO2_PPB,NOX_PPB,NO_PPB"},
            {"feed": "11067", "channel": "PM25_640_UG_M3,NO2_200_PPB,NOx_200_PPB,NO_200_PPB"}
        ], # Feed_43_and_Feed_11067_Parkway_East_ACHD_DATA2
        [
            {"feed": "3506", "channel": "PM2_5,OZONE"}
        ], # Feed_3506_BAPC_301_39TH_STREET_BLDG_AirNow
        [
            {"feed": "5975", "channel": "PM2_5"}
        ], # Feed_5975_Parkway_East_AirNow
        [
            {"feed": "3508", "channel": "PM2_5"}
        ], # Feed_3508_South_Allegheny_High_School_AirNow
        [
            {"feed": "24", "channel": "PM10_UG_M3"}
        ], # Feed_24_Glassport_High_Street_ACHD
        [
            {"feed": "26086", "channel": "PAMB_640_KPa,PM25_640_UG_M3,TAMB_640_DEGC"}
        ] # Feed_26086_Clairton_ACHD
    ]

    start_time = datetimeToEpochtime(start_dt) / 1000 # ESDR uses seconds
    end_time = datetimeToEpochtime(end_dt) / 1000 # ESDR uses seconds
    df_esdr_array_raw = getEsdrData(esdr_source, start_time=start_time, end_time=end_time)

    if region_setting == 0:
        # Get smell reports (for the Smell PGH paper, only contains a certain zipcodes)
        df_smell_raw = getSmellReports(start_time=start_time, end_time=end_time, api_version=1)
    else:
        # Get smell reports (for general tasks, contains data from a wider geographical region)
        df_smell_raw = getSmellReports(start_time=start_time, end_time=end_time, allegheny_county=True, api_version=2)

    # Check directory and save file
    if out_p is not None:
        for p in out_p: checkAndCreateDir(p)
        for i in range(len(df_esdr_array_raw)):
            df_esdr_array_raw[i].to_csv(out_p[0] + esdr_source_names[i] + ".csv")
        df_smell_raw.to_csv(out_p[1])
        log("Raw ESDR data created at " + out_p[0], logger)
        log("Raw smell data created at " + out_p[1], logger)
    return df_esdr_array_raw, df_smell_raw
