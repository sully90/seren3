
import sys
import seren3
from seren3.analysis.visualization import EngineMode

path = sys.argv[1]
iout = int(sys.argv[2])

snap = seren3.load_snapshot(path, iout)
halos = snap.halos(finder='ctrees')
h = halos.sort("Mvir")[0]

proj = h.d.projection("mass", mode=EngineMode.SPLATTER)