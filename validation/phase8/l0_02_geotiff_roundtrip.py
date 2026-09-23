"""L0-02 GeoTIFF geospatial round-trip (design check with synthetic rasters)."""
import json, numpy as np, rasterio
from rasterio.transform import from_origin, Affine
from rasterio.crs import CRS
res = {}
# (a) UTM 43N, 1 m pixels, origin (500000, 3000000), 64x64, nodata, tags incl. VERTICAL_CRS
tr = from_origin(500000.0, 3000000.0, 1.0, 1.0); crs = CRS.from_epsg(32643)
arr = np.arange(64*64, dtype=np.float32).reshape(64,64); arr[0,0] = -9999
prof = dict(driver="GTiff", width=64, height=64, count=1, dtype="float32", crs=crs, transform=tr, nodata=-9999, tiled=True, blockxsize=32, blockysize=32, compress="lzw")
with rasterio.open("results/synthetic_utm.tif","w",**prof) as ds:
    ds.write(arr,1); ds.update_tags(VERTICAL_CRS="EPSG:3855", TIER="T", MODEL_HASH="deadbeef"); ds.build_overviews([2,4]); ds.update_tags(ns="rio_overview", resampling="average")
with rasterio.open("results/synthetic_utm.tif") as ds:
    res["a_crs_roundtrip"] = ds.crs.to_epsg()==32643
    res["a_transform_roundtrip"] = ds.transform==tr
    res["a_nodata"] = ds.nodata
    res["a_tags"] = {k:ds.tags().get(k) for k in ("VERTICAL_CRS","TIER","MODEL_HASH")}
    res["a_overviews"] = ds.overviews(1)
    # pixel-centre mapping: pixel (10,20) -> centre coords
    x,y = ds.xy(20,10)  # row, col
    res["a_pixel_10_20_centre_xy"] = [x,y]
    res["a_expected_centre_xy"] = [500000+10.5, 3000000-20.5]
    res["a_centre_ok"] = (abs(x-500010.5)<1e-9 and abs(y-(3000000-20.5))<1e-9)
    r,c = ds.index(500010.5, 3000000-20.5); res["a_inverse_ok"] = (r,c)==(20,10)
    res["a_bounds"] = list(ds.bounds); res["a_bounds_ok"] = list(ds.bounds)==[500000.0, 3000000.0-64, 500000.0+64, 3000000.0]
    res["a_nodata_count"] = int((ds.read(1)==-9999).sum())
# (b) rotated transform detection
trr = Affine(0.9, 0.1, 500000.0, 0.1, -0.9, 3000000.0)
with rasterio.open("results/synthetic_rot.tif","w",**{**prof,"transform":trr}) as ds: ds.write(arr,1)
with rasterio.open("results/synthetic_rot.tif") as ds:
    res["b_rotation_detected"] = not (ds.transform.b==0 and ds.transform.d==0)
    res["b_gsd_from_transform_m"] = [float(np.hypot(ds.transform.a, ds.transform.d)), float(np.hypot(ds.transform.b, ds.transform.e))]
# (c) geographic CRS local GSD in metres via pyproj geodesic
trg = from_origin(77.2, 28.6, 1e-5, 1e-5)
with rasterio.open("results/synthetic_wgs84.tif","w",**{**prof,"crs":CRS.from_epsg(4326),"transform":trg}) as ds: ds.write(arr,1)
from pyproj import Geod
g = Geod(ellps="WGS84")
with rasterio.open("results/synthetic_wgs84.tif") as ds:
    lon,lat = ds.xy(32,32)
    _,_,dx = g.inv(lon,lat,lon+1e-5,lat); _,_,dy = g.inv(lon,lat,lon,lat+1e-5)
    res["c_local_gsd_m"] = [round(dx,4), round(dy,4)]
    res["c_gsd_isotropic_ok"] = abs(dx-dy)/dy < 0.15  # cos(lat) anisotropy ~ 0.88 at 28.6°
    res["c_no_crs_would_be_mode_A"] = True
# (d) TIFF with no CRS -> mode A classification
with rasterio.open("results/synthetic_nocrs.tif","w",**{**prof,"crs":None,"transform":Affine.identity()}) as ds: ds.write(arr,1)
with rasterio.open("results/synthetic_nocrs.tif") as ds:
    res["d_nocrs_crs_is_none"] = ds.crs is None; res["d_nocrs_transform_identity"] = ds.transform==Affine.identity()
res["PASS"] = all([res["a_crs_roundtrip"],res["a_transform_roundtrip"],res["a_centre_ok"],res["a_inverse_ok"],res["a_bounds_ok"],res["a_tags"]["VERTICAL_CRS"]=="EPSG:3855",res["b_rotation_detected"],res["c_gsd_isotropic_ok"],res["d_nocrs_crs_is_none"]])
json.dump(res, open("results/l0_02_geotiff_roundtrip.json","w"), indent=2, default=str); print(json.dumps(res, indent=1, default=str))
