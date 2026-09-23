#!/usr/bin/env bash
# Fetch offline assets: PROJ geoid grids, Copernicus GLO-30 tiles for the demo AOIs, swisstopo LiDAR references.
set -e
cd "$(dirname "$0")/.."
echo "[grids]"; for g in us_nga_egm96_15.tif us_nga_egm08_25.tif ch_swisstopo_chgeo2004_ETRS89_LN02.tif; do [ -f assets/proj/$g ] || curl -sfL "https://cdn.proj.org/$g" -o assets/proj/$g; done
echo "[dem]"
for t in N47_00_E008_00 N46_00_E007_00; do f="Copernicus_DSM_COG_10_${t}_DEM.tif"; [ -f assets/dem/$f ] || curl -sfL "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_${t}_DEM/$f" -o assets/dem/$f; done
echo "[reference: swisstopo swissSURFACE3D Raster 0.5 m DSM + swissALTI3D 0.5 m DTM, OGD]"
q() { curl -sfL "https://data.geo.admin.ch/api/stac/v0.9/collections/$1/items?bbox=$2&limit=5"; }
for spec in "2682-1247 8.535,47.375,8.545,47.380 urban" "2621-1202 7.72,46.97,7.73,46.975 rural"; do
  set -- $spec; tile=$1; bbox=$2; name=$3
  q ch.swisstopo.swisssurface3d-raster "$bbox" | python -c "
import sys,json,subprocess,os
d=json.load(sys.stdin)
for it in d['features']:
    for k,a in it['assets'].items():
        if a['href'].endswith('_0.5_2056_5728.tif') and '$tile' in it['id']:
            out='assets/reference/swisssurface3d_${name}_'+it['id'].split('_')[-1]+'_dsm_0.5m.tif'
            if not os.path.exists(out): subprocess.run(['curl','-sfL',a['href'],'-o',out],check=True); print('downloaded',out)
            raise SystemExit
print('no swissSURFACE3D asset found for $tile')"
  q ch.swisstopo.swissalti3d "$bbox" | python -c "
import sys,json,subprocess,os
d=json.load(sys.stdin)
for it in d['features']:
    for k,a in it['assets'].items():
        if a['href'].endswith('_0.5_2056_5728.tif') and '$tile' in it['id']:
            out='assets/reference/swissalti3d_${name}_'+it['id'].split('_')[-1]+'_dtm_0.5m.tif'
            if not os.path.exists(out): subprocess.run(['curl','-sfL',a['href'],'-o',out],check=True); print('downloaded',out)
            raise SystemExit
print('no swissALTI3D asset found for $tile')"
done
ls -la assets/proj assets/dem assets/reference
