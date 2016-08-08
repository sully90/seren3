def main(path, iout, NSLOTS):
    import seren3

    print path, iout
    snap = seren3.load_snapshot(path, iout)
    halos = snap.halos(finder='ahf')
    if not halos.can_load(NSLOTS=NSLOTS):
        halos.run()
    else:
        print "Halo catalogue already exists"

if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    iout = int(sys.argv[2])
    NSLOTS = int(sys.argv[3])
    main(path, iout, NSLOTS)