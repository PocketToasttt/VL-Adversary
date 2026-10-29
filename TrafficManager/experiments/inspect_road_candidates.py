"""Print map topology candidates for dense traffic previews."""
import math
from pathlib import Path
import sumolib

net = sumolib.net.readNet(str(Path(__file__).resolve().parents[1] / "networkFiles/boston-seaport/osm.net.xml"))
rows = []
for edge in net.getEdges():
    if not edge.allows("passenger") or edge.getLength() < 70:
        continue
    shape = edge.getShape()
    direct = math.dist(shape[0], shape[-1])
    sinuosity = edge.getLength() / max(direct, 1)
    target = edge.getToNode()
    incoming = [e for e in target.getIncoming() if e.allows("passenger")]
    outgoing = [e for e in target.getOutgoing() if e.allows("passenger")]
    rows.append((edge.getID(), round(edge.getLength()), len(edge.getLanes()), round(sinuosity, 2), len(incoming), len(outgoing), target.getID()))
print("long edges", len(rows))
for label, filtered in (
    ("wide_straight", [r for r in rows if r[2] >= 3 and r[3] < 1.02]),
    ("wide_curve", [r for r in rows if r[2] >= 2 and r[3] >= 1.04]),
    ("mild_curve", [r for r in rows if r[2] >= 2 and r[3] >= 1.01]),
    ("intersection", [r for r in rows if r[2] >= 2 and r[4] >= 3 and r[5] >= 3]),
    ("merge", [r for r in rows if r[2] >= 2 and r[4] >= 2 and r[5] <= 2]),
):
    print(label, len(filtered))
    for row in sorted(filtered, key=lambda x: -x[1])[:12]:
        print("  ", *row)
