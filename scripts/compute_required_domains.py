'''
Script to read a txt file of cube/sphere positions and widths/radaii 
and return the domains required to bound the volumes
'''

from enum import Enum
class RegionMode(Enum):
    CUBE = 1
    SPHERE = 2

# Filter cube or spherical regions? -> Spheres will (possible) require less domains
REGION_MODE = RegionMode.CUBE

def main(path, ioutput, txt_fname, out_fname=None):
    '''
    Reads the snapshot (to compute domain decomposition at this time)
    and outputs a set of idomains required
    '''
    import csv, seren3

    # Load the snapshot
    print "Loading snapshot"
    snapshot = seren3.load_snapshot(path, ioutput)

    domains = []
    # Open the csv file and collect the desired regions
    with open(txt_fname, "r") as csvfile:
        reader = csv.reader(csvfile)
        print "Reading csv file"
        for row in reader:
            if (hasattr(row, "__iter__") and row[0].startswith("#")):
                continue
            else:
                x,y,z,l = [float(ri) for ri in row]
                pos = [x,y,z]
                reg = None
                if REGION_MODE == RegionMode.CUBE:
                    reg = snapshot.get_cube(pos, l)
                elif REGION_MODE == RegionMode.SPHERE:
                    reg = snapshot.get_sphere(pos, l)
                else:
                    raise Exception("Unknown region mode: %s" % REGION_MODE)
                # Compute minimal number of domains to bound this region
                bbox = reg.get_bounding_box()
                idomains = snapshot.cpu_list(bbox)
                domains.extend(idomains)

    # Use a set to remove duplicates
    domains = sorted(list(set(domains)))  # Converts to set then back to a list

    if (out_fname is not None):
        print "Writing output"
        with open(out_fname, "w") as f:
            for i in range(len(domains)):
                f.write("%i" % domains[i])
                if (i < len(domains) - 1):
                    f.write("\n")

    print "Done"
    # Return the list
    return domains



if __name__ == "__main__":
    '''
    Main function. Read simulation path, output number
    and txt filename
    '''
    import sys

    def usage():
        print "Usage: python compute_required_domains.py <path> <ioutput> <regions_filename>"
        sys.exit()

    if (len(sys.argv) == 1):
        usage()
    elif (len(sys.argv) < 4):
        usage()
    path = sys.argv[1]
    ioutput = int(sys.argv[2])
    txt_fname = sys.argv[3]
    out_fname = None

    if (len(sys.argv) > 4):
        out_fname = sys.argv[4]

    main(path, ioutput, txt_fname, out_fname)