from concurrent.futures import ThreadPoolExecutor
import geopandas as gpd
import pandas as pd
import requests
from collections import defaultdict
import matplotlib.pyplot as plt
import os
from cendat import CenDatHelper
import contextily as ctx

geo_data = defaultdict(gpd.GeoDataFrame)


def get_tiger_polygons(
    layer_id: int,
    where_clause: str,
    fields: str,
    service: str = "TIGERweb/tigerWMS_Current",
) -> gpd.GeoDataFrame:

    API_URL = f"https://tigerweb.geo.census.gov/arcgis/rest/services/{service}/MapServer/{layer_id}/query"

    params = {
        "where": where_clause,
        "outFields": fields,
        "outSR": "4326",
        "f": "geojson",
        "returnGeometry": "false",
        "returnCountOnly": "false",
        "resultOffset": 0,
        "resultRecordCount": 100_000,
        "timeout": 60,
    }

    try:
        response = requests.get(API_URL, params=params)
        response.raise_for_status()
        geo_data[layer_id] = gpd.GeoDataFrame.from_features(response.json()["features"])
        print(f"✅ Successfully fetched {len(geo_data[layer_id])} centroids.")

    except requests.exceptions.RequestException as e:
        print(f"❌ HTTP Request failed: {e}")
    except (KeyError, ValueError) as e:
        print(f"❌ Failed to parse response JSON: {e}")
        print(f"   Server Response: {response.text[:200]}...")


try:
    with ThreadPoolExecutor(max_workers=2) as executor:

        future_places = executor.submit(
            get_tiger_polygons, 28, "1=1", "STATE,NAME,AREALAND,CENTLAT,CENTLON"
        )
        future_countysubs = executor.submit(
            get_tiger_polygons,
            22,
            (
                "NAME LIKE '%township' OR "
                "NAME LIKE '%town' OR "
                "NAME LIKE '%village' OR "
                "NAME LIKE '%borough'"
            ),
            "STATE,NAME,AREALAND,CENTLAT,CENTLON",
        )

        future_places.result()
        future_countysubs.result()

except Exception as exc:
    print(f"❌ A master fetching task failed: {exc}")

stacked = pd.concat(geo_data.values(), ignore_index=True)
stacked["AREALAND"] = stacked["AREALAND"].astype(int)
stacked["CENTLAT"] = stacked["CENTLAT"].astype(float)
stacked["CENTLON"] = stacked["CENTLON"].astype(float)
stacked["DECILE"] = stacked.groupby("STATE")["AREALAND"].transform(
    lambda x: pd.qcut(x, 20, labels=False, duplicates="drop") + 1
)

cdh = CenDatHelper(key=os.getenv("CENSUS_API_KEY"))

cdh.list_products(years=[2023], patterns=r"acs/acs5\)")
cdh.set_products()
cdh.list_groups(patterns=r"^median household income")
cdh.set_groups(["B19013"])
cdh.describe_groups()
cdh.set_geos(["150"])
response = cdh.get_data(
    # in_place=True,
    include_names=True,
    include_geometry=True,
    within={
        "state": [
            "08",
        ],
        "county": ["069", "123", "013"],
    },
)


gdf = response.to_gpd(destring=True, join_strategy="inner")
gdf.loc[gdf["B19013_001E"] == -666666666, "B19013_001E"] = None

fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=300)

# Plot the choropleth map
gdf.plot(
    column="B19013_001E",
    cmap="viridis",
    linewidth=0.2,
    edgecolor="black",
    ax=ax,
    legend=True,
    alpha=0.8,
    legend_kwds={
        "label": "Income",
        "orientation": "horizontal",
        "location": "bottom",
        "shrink": 0.5,
        "fraction": 0.1,
        "format": "{x:,.0f}",
        "alpha": 0.8,
        "pad": 0.1,
    },
    missing_kwds={
        "color": "lightgrey",
        "edgecolor": "grey",
        "hatch": "////",
        "label": "Missing values",
    },
)

xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()

visible_centroids = stacked[
    (stacked["CENTLON"] >= xmin)
    & (stacked["CENTLON"] <= xmax)
    & (stacked["CENTLAT"] >= ymin)
    & (stacked["CENTLAT"] <= ymax)
    & (stacked["DECILE"] >= 19)
]

ax.scatter(
    visible_centroids["CENTLON"],
    visible_centroids["CENTLAT"],
    s=10,  # Marker size
    c="black",  # Marker color
    edgecolor="white",
    zorder=2,  # zorder ensures points are drawn on top of the polygons
    alpha=0.8,
)

y_offset = 0.015

for idx, row in visible_centroids.iterrows():
    ax.text(
        x=row["CENTLON"],
        y=row["CENTLAT"] + y_offset,
        s=row["NAME"],
        fontsize=max(3, 9 * (row["DECILE"] ** 3 / 20**3)),
        fontweight="light",
        ha="center",
        va="bottom",
        zorder=3,
        bbox=dict(
            boxstyle="round,pad=0.1,rounding_size=0.2",
            fc="white",
            ec="none",
            alpha=0.7,
        ),
    )

ctx.add_basemap(
    ax,
    source=ctx.providers.CartoDB.PositronNoLabels,
    attribution=False,
    zoom=10,
    crs=4326,
    alpha=1.0,
)

ax.set_title(
    "Larimer, Weld, and Boulder County Med. HH Income by block group",
    fontdict={"fontsize": "16", "fontweight": "3"},
)
ax.set_axis_off()
plt.show()
