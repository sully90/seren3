def main(path, iout):
    import seren3

    snap = seren3.load_snapshot(path, iout)
    halos = snap.h

    h = halos.sort('mvir')[0]
    subsnap = h.subsnap
    center = subsnap.region.center

    pos = subsnap.g["pos"].flatten(center=center)

if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    iout = int(sys.argv[2])

    main(path, iout)