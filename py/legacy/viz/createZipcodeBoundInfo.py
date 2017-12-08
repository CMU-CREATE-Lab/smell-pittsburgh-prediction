import json
from shapely.geometry import mapping, shape

import sys
sys.path.append("../analysis/")
from util import *

# This function reads the GeoJSON zipcode boundaries
# Then outputs a table that maps zipcode, bound, and center location of the polygons
def main():
    p = "../../data/"
    file_path_in = p+"geo/zipcode_bound_geoJson_allegany_county_PA.json"
    file_path_out = p+"geo/zipcode_bound_info_allegany_county_PA.json"

    # Check output directory
    checkAndCreateDir(file_path_out)

    # Read the zipcode boundaries which have Specks
    print "Read the zipcode bound GeoJSON " + file_path_in
    with open(file_path_in, "r") as in_file:
        zipcode_bound = json.load(in_file)
    
    # Compute the bound and center of each polygon
    print "Compute bounds and centers of polygons in the GeoJSON"
    zipcode_bound_info = {
        "format": {"zipcode": ["min_lng", "min_lat", "max_lng", "max_lat", "center_lng", "center_lat"]},
        "data": {}
    }
    for feature in zipcode_bound["features"]:
        zipcode = feature["properties"]["ZCTA5CE10"]
        geometry = shape(feature["geometry"])
        b = geometry.bounds
        c = [(b[0]+b[2])/2, (b[1]+b[3])/2]
        zipcode_bound_info["data"][zipcode] = list(b) + c

    # Write to a file
    print "Create zipcode bound info at " + file_path_out
    with open(file_path_out, "w") as out_file:
        json.dump(zipcode_bound_info, out_file)

if __name__ == "__main__":
    main()
