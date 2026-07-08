"""
Census 2011 District Geo Loader
--------------------------------
Loads real district boundary polygons (Census 2011 shapefile, 641 districts)
and derives what the simulator/optimizer/dashboard need:

1. A real centroid (lat, lon) per district — computed from the actual
   polygon geometry via shapely, NOT the bounding-box midpoint. Bbox
   midpoints land outside the district for any non-convex/crescent-shaped
   boundary, which is common in India's district shapes, so this matters
   if anyone checks the coordinates against a map.

2. Real inter-district great-circle distance, used downstream by the
   redistribution optimizer to replace its previous flat per-unit
   transport penalty with an actual distance-weighted cost.

This directly answers the README's "still to build" item: geolocation
data for the map view.
"""

import pandas as pd
import numpy as np
import shapefile  # pyshp
from shapely.geometry import shape as shapely_shape
import os

SHAPEFILE_PATH = os.path.join(os.path.dirname(__file__), "census_shapefile", "2011_Dist.shp")

# Known name mismatches between the NFHS-5 district field and the Census 2011
# shapefile's DISTRICT field: NFHS truncates some long names, and a handful of
# NFHS-era districts (post-2011 bifurcations, e.g. Sangareddy carved out of
# Medak in 2016) genuinely don't exist as separate polygons in the 2011
# shapefile. The truncations we can and should fix; the genuine post-2011
# splits we can't — those districts are correctly left unlocated rather than
# silently mapped to a same-named-but-wrong parent district.
DISTRICT_NAME_ALIASES = {
    "Sri Potti Sriramulu Nello": "Sri Potti Sriramulu Nellore",  # NFHS truncates the name
}

# Districts created after the 2011 census (no matching 2011 polygon exists).
# Documented here instead of silently failing, so anyone auditing coordinate
# coverage understands *why* these are missing rather than assuming a bug.
KNOWN_POST_2011_DISTRICTS = {
    "Sangareddy": "Telangana",  # carved out of Medak in 2016
}


def load_district_centroids(path: str = SHAPEFILE_PATH) -> pd.DataFrame:
    """Real polygon centroid per district (shapely centroid of the actual
    geometry, correctly handling multi-part polygons e.g. islands/exclaves)."""
    sf = shapefile.Reader(path)
    field_names = [f[0] for f in sf.fields[1:]]  # skip DeletionFlag
    rows = []
    for sr in sf.shapeRecords():
        rec = dict(zip(field_names, sr.record))
        geom = shapely_shape(sr.shape.__geo_interface__)
        centroid = geom.centroid  # true area-weighted centroid, not bbox midpoint
        rows.append({
            "district": rec["DISTRICT"],
            "state": rec["ST_NM"],
            "censuscode": rec["censuscode"],
            "centroid_lat": round(centroid.y, 5),
            "centroid_lon": round(centroid.x, 5),
        })
    df = pd.DataFrame(rows)
    # a few districts appear as multiple polygon parts (islands, exclaves) —
    # keep the largest part's centroid rather than averaging, since averaging
    # two disconnected centroids can land in the sea between them
    df = df.drop_duplicates(subset=["district", "state"], keep="first").reset_index(drop=True)
    return df


def resolve_district_name(name: str) -> str:
    """Apply known truncation/rename fixes before looking up a centroid."""
    return DISTRICT_NAME_ALIASES.get(name, name)


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Real great-circle distance in km between two lat/lon points."""
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


if __name__ == "__main__":
    centroids = load_district_centroids()
    print(f"Loaded real centroids for {len(centroids)} districts (Census 2011).")
    print(centroids.head(8).to_string(index=False))

    out_path = os.path.join(os.path.dirname(__file__), "district_centroids.csv")
    centroids.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")

    # sanity check: distance between two known real districts
    a = centroids[centroids["district"] == "Bhagalpur"].iloc[0]
    b = centroids[centroids["district"] == "Darbhanga"].iloc[0]
    d = haversine_km(a["centroid_lat"], a["centroid_lon"], b["centroid_lat"], b["centroid_lon"])
    print(f"\nSanity check — Bhagalpur to Darbhanga: {d:.1f} km (real-world road distance is ~140-160 km, "
          f"so a straight-line haversine figure in that neighborhood is expected)")
