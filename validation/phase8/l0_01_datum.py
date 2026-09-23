"""L0-01 Vertical datum reference values (design check, not codebase test).
Computes EGM96 / EGM2008 geoid undulation at Indian points with PROJ.
Also checks whether the transform works OFFLINE (grids bundled?) vs ONLINE (PROJ CDN)."""
import json, os, sys, math
import pyproj
from pyproj import Transformer
PTS = {"Kanyakumari":(77.55,8.08),"Bengaluru":(77.59,12.97),"Mumbai":(72.88,19.08),"Delhi":(77.21,28.61),
       "Kolkata":(88.36,22.57),"Leh":(77.58,34.15),"Guwahati":(91.74,26.14),"Dehradun":(78.03,30.32)}
def run(network: bool):
    pyproj.network.set_network_enabled(network)
    out = {}
    for name,(lon,lat) in PTS.items():
        r = {}
        for tag, epsg in (("EGM96","EPSG:9707"),("EGM2008","EPSG:9518")):
            try:
                t = Transformer.from_crs("EPSG:4979", epsg, always_xy=True)
                x,y,h = t.transform(lon,lat,0.0)
                r[tag] = None if (h is None or math.isinf(h) or math.isnan(h)) else round(-h,3)
            except Exception as e:
                r[tag] = f"ERROR: {e}"
        out[name] = r
    return out
res = {"proj_version": pyproj.proj_version_str, "pyproj": pyproj.__version__,
       "offline": run(False), "online": run(True)}
pyproj.network.set_network_enabled(False)
_desc = Transformer.from_crs("EPSG:4979","EPSG:9518", always_xy=True).description
res["offline_pipeline_description"] = _desc
res["offline_grids_available"] = ("ballpark" not in _desc.lower())
res["FINDING"] = "Offline PROJ silently uses a ballpark (0 m) vertical transform when grids are absent; detection must inspect Transformer.description for 'ballpark' and verify grid files exist." if not res["offline_grids_available"] else "grids present"
res["egm96_minus_egm2008_m"] = {k: round(v["EGM96"]-v["EGM2008"],3) for k,v in res["online"].items() if isinstance(v["EGM96"],float) and isinstance(v["EGM2008"],float)}
json.dump(res, open("results/l0_01_datum.json","w"), indent=2)
print(json.dumps(res, indent=1))
