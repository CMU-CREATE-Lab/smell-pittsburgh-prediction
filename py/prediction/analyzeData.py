from util import *
import numpy as np
import pandas as pd

def analyzeData(
    in_p = None, # input path for raw esdr and smell data
    logger=None):

    log("Analyze data...", logger)
