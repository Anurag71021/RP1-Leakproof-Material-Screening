import re
import sys
import xml.etree.ElementTree as ET

path = sys.argv[1]
tree = ET.parse(path)
root = tree.getroot()
for tw in root.iter("table-wrap"):
    label = tw.findtext("label") or ""
    caption = " ".join("".join(tw.find("caption").itertext()).split()) if tw.find("caption") is not None else ""
    print(f"\n=== {label} :: {caption[:220]}")
    table = tw.find("table")
    if table is None:
        continue
    for tr in table.iter("tr"):
        cells = []
        for cell in list(tr):
            if cell.tag in ("td", "th"):
                cells.append(" ".join("".join(cell.itertext()).split()))
        if cells:
            print(" | ".join(cells)[:300])
