import numpy as np
import json

import sys  
sys.path.append("../analysis/")
from util import *

# This function is used for the heatmap visualization
# Select only zipcode GeoJSON bounds in the Allegany county in PA
# INPUT: GeoJSON of all zipcode bounds in the US
# OUTPUT: GeoJSON of zipcode bounds (plus latlng bounds) in the Allegany county in PA
def main():
    p = "../../data/"
    file_path_in = p+"geo/zcta5.json"
    file_path_out = p+"geo/zipcode_bound_geoJson_allegany_county_PA.json"

    # Check output directory
    checkAndCreateDir(file_path_out)

    # All unique zipcodes for the Allegany county in PA
    zipcodes_unique = [15006, 15007, 15014, 15015, 15017, 15018, 15020, 15024, 15025,
        15028, 15030, 15031, 15032, 15034, 15035, 15037, 15044, 15045, 15046, 15047,
        15049, 15051, 15056, 15064, 15065, 15071, 15075, 15076, 15082, 15084, 15086,
        15088, 15090, 15091, 15095, 15096, 15101, 15102, 15104, 15106, 15108, 15110,
        15112, 15116, 15120, 15122, 15123, 15126, 15127, 15129, 15131, 15132, 15133,
        15134, 15135, 15136, 15137, 15139, 15140, 15142, 15143, 15144, 15145, 15146,
        15147, 15148, 15201, 15202, 15203, 15204, 15205, 15206, 15207, 15208, 15209,
        15210, 15211, 15212, 15213, 15214, 15215, 15216, 15217, 15218, 15219, 15220,
        15221, 15222, 15223, 15224, 15225, 15226, 15227, 15228, 15229, 15230, 15231,
        15232, 15233, 15234, 15235, 15236, 15237, 15238, 15239, 15240, 15241, 15242,
        15243, 15244, 15250, 15251, 15252, 15253, 15254, 15255, 15257, 15258, 15259,
        15260, 15261, 15262, 15264, 15265, 15267, 15268, 15270, 15272, 15274, 15275,
        15276, 15277, 15278, 15279, 15281, 15282, 15283, 15286, 15289, 15290, 15295]

    # Read the geoJSON file that contains all zipcode bounds in USA
    print "Read the zipcode boundaries from " + file_path_in
    with open(file_path_in, "r") as in_file:
        zipcode_bound_all = json.load(in_file)

    # Select only the boundaries in the Allegany county in PA
    print "Select only the boundaries in the Allegany county in PA"
    zipcode_bound_ac = {
        "type": "FeatureCollection",
        "features": []
    }
    for feature in zipcode_bound_all["features"]:
        zipcode = int(feature["properties"]["ZCTA5CE10"])
        if zipcode in zipcodes_unique:
            zipcode_bound_ac["features"].append(feature)

    # Write to a GeoJSON file
    print "Write the boundaries to GeoJSON file " + file_path_out
    with open(file_path_out, "w") as out_file:
        json.dump(zipcode_bound_ac, out_file)

if __name__ == "__main__":
    main()
