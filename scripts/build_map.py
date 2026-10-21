#!/usr/bin/env python3
"""
build_map.py - build an Equal Earth world map as an inline SVG for a Jekyll site.

Reads the visited / planned countries from a small YAML file, projects Natural
Earth country shapes with the Equal Earth projection, and writes an SVG
fragment that Jekyll can include. Colours come from CSS (see world-map.css),
so the map follows the site's theme, including dark mode.

Usage (from the repo root):
    pip install geopandas pyyaml
    python scripts/build_map.py
    python scripts/build_map.py --preview preview.html    # open this to check it

Then, in the page:
    {% include maps/world.svg %}

Options worth knowing:
    --resolution 50m    more detailed borders (bigger file); default is 110m
    --south -60         where the map is cut off at the bottom (Antarctica is dropped)
    --antarctica        keep Antarctica and draw the full Equal Earth outline
    --graticule 30      grid spacing in degrees; 0 turns the grid off
    --strict            exit with an error if a country in the YAML is not found
"""

from __future__ import annotations

import argparse
import difflib
import html
import sys
import urllib.request
from pathlib import Path

import geopandas as gpd
import numpy as np
import yaml
from pyproj import Transformer
from shapely.affinity import affine_transform

ROOT = Path(__file__).resolve().parent.parent

# Equal Earth on a sphere. Equal-area, so projected areas are true areas in m^2.
CRS = "+proj=eqearth +R=6371008.8 +no_defs"

# Natural Earth's own GitHub mirror of the vector data.
NE_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "master/geojson/ne_{res}_admin_0_countries.geojson"
)

# Visited/planned countries smaller than this also get a dot, so they stay visible.
SMALL_COUNTRY_KM2 = 10_000


# --------------------------------------------------------------------------- data


def load_world(
    resolution: str, data_dir: Path, keep_antarctica: bool
) -> gpd.GeoDataFrame:
    """Load Natural Earth countries, downloading the file once if it is missing."""
    path = data_dir / f"ne_{resolution}_admin_0_countries.geojson"
    if not path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {path.name} ...", file=sys.stderr)
        urllib.request.urlretrieve(NE_URL.format(res=resolution), path)
    world = gpd.read_file(path)
    if not keep_antarctica:
        world = world[world["ADM0_A3"] != "ATA"]
    return world


def build_lookup(world: gpd.GeoDataFrame) -> dict[str, int]:
    """Map every code/name we accept (lower-case) to a row index.

    Note: Natural Earth's ISO_A3 is -99 for France, Norway and a few others,
    so we match on ISO_A3_EH / ADM0_A3 instead.
    """
    lookup: dict[str, int] = {}
    columns = [
        "ISO_A3_EH",
        "ADM0_A3",
        "ISO_A2_EH",
        "NAME",
        "NAME_LONG",
        "ADMIN",
        "BRK_NAME",
    ]
    for idx, row in world.iterrows():
        for col in columns:
            value = row.get(col)
            if isinstance(value, str) and value and value != "-99":
                lookup.setdefault(value.strip().lower(), idx)
    return lookup


def resolve_status(world: gpd.GeoDataFrame, config: dict):
    """Return ({row index: 'visited' | 'planned'}, [unmatched entries])."""
    lookup = build_lookup(world)
    status: dict[int, str] = {}
    unmatched: list[tuple[str, object]] = []
    # Planned first, visited second: a country in both lists counts as visited.
    for kind in ("planned", "visited"):
        for item in config.get(kind) or []:
            idx = lookup.get(str(item).strip().lower())
            if idx is None:
                unmatched.append((kind, item))
            else:
                status[idx] = kind
    return status, unmatched


def suggest(item: object, lookup_keys: list[str]) -> str:
    close = difflib.get_close_matches(str(item).lower(), lookup_keys, n=3, cutoff=0.6)
    return f" (did you mean: {', '.join(close)}?)" if close else ""


# ------------------------------------------------------------------- geometry


def outline_lonlat(south: float, step: float = 1.0):
    """Boundary of the map as (lon, lat) points. In Equal Earth the poles are lines."""
    left = [(-180.0, lat) for lat in np.arange(south, 90 + step, step)]
    top = [(lon, 90.0) for lon in np.arange(-180, 180 + step, step)]
    right = [(180.0, lat) for lat in np.arange(90, south - step, -step)]
    bottom = [(lon, south) for lon in np.arange(180, -180 - step, -step)]
    return left + top + right + bottom


def graticule_lonlat(step: int, south: float):
    """Meridians and parallels as lists of (lon, lat) points, densified for curves."""
    lines = []
    if step <= 0:
        return lines
    lats = np.arange(south, 90 + 1, 1.0)
    for lon in range(-180 + step, 180, step):
        lines.append([(float(lon), float(lat)) for lat in lats])
    for lat in range(-90 + step, 90, step):
        if lat > south:
            lines.append([(float(lon), float(lat)) for lon in (-180, 180)])
    return lines


def points_d(points, closed: bool = False, nd: int = 1) -> str:
    """SVG path data for a list of (x, y), dropping repeated points."""
    cleaned = []
    last = None
    for x, y in points:
        p = (round(x, nd), round(y, nd))
        if p != last:
            cleaned.append(p)
            last = p
    if len(cleaned) < (3 if closed else 2):
        return ""
    d = "M" + "L".join(f"{x:g} {y:g}" for x, y in cleaned)
    return d + ("Z" if closed else "")


def geometry_d(geom) -> str:
    """SVG path data for a (Multi)Polygon."""
    polygons = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
    parts = []
    for poly in polygons:
        for ring in [poly.exterior, *poly.interiors]:
            parts.append(points_d(ring.coords, closed=True))
    return "".join(parts)


# -------------------------------------------------------------------- build SVG


def build_svg(world, status, markers, *, south, graticule_step, width, pad=4):
    to_proj = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)

    # Fit the projected outline into the viewBox.
    lonlat = outline_lonlat(south)
    ox, oy = to_proj.transform([p[0] for p in lonlat], [p[1] for p in lonlat])
    min_x, max_x, min_y, max_y = min(ox), max(ox), min(oy), max(oy)
    scale = (width - 2 * pad) / (max_x - min_x)
    height = (max_y - min_y) * scale + 2 * pad
    x_off = pad - scale * min_x
    y_off = pad + scale * max_y
    matrix = [scale, 0, 0, -scale, x_off, y_off]  # y is flipped for SVG

    def to_px(lon, lat):
        x, y = to_proj.transform(lon, lat)
        return scale * x + x_off, -scale * y + y_off

    sphere_d = points_d([to_px(lon, lat) for lon, lat in lonlat], closed=True)
    grid_d = "".join(
        points_d([to_px(lon, lat) for lon, lat in line])
        for line in graticule_lonlat(graticule_step, south)
    )

    projected = world.to_crs(CRS)
    areas_km2 = projected.geometry.area / 1e6
    px_geoms = projected.geometry.apply(lambda g: affine_transform(g, matrix))

    # Draw order: background countries, then planned, then visited (so their borders sit on top).
    order = {None: 0, "planned": 1, "visited": 2}
    rows = sorted(world.index, key=lambda i: order[status.get(i)])

    svg = [
        f'<svg class="world-map" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height:.0f}" '
        f'role="img" aria-labelledby="world-map-title">',
        '<title id="world-map-title">World map of visited and planned countries</title>',
        f'<path class="map-sphere" d="{sphere_d}"/>',
    ]
    if grid_d:
        svg.append(f'<path class="map-graticule" d="{grid_d}" fill="none"/>')
    svg.append('<g class="map-countries" fill-rule="evenodd">')

    dots = []
    for idx in rows:
        row = world.loc[idx]
        kind = status.get(idx)
        name = html.escape(str(row["NAME"]))
        css = f"country {kind}" if kind else "country"
        label = f"{name} ({kind})" if kind else name
        iso = (
            row.get("ISO_A3_EH")
            if row.get("ISO_A3_EH") != "-99"
            else row.get("ADM0_A3")
        )
        d = geometry_d(px_geoms.loc[idx])
        if not d:
            continue
        svg.append(
            f'<path class="{css}" data-iso="{html.escape(str(iso))}" data-name="{name}" d="{d}"></path>'
        )
        if kind and areas_km2.loc[idx] < SMALL_COUNTRY_KM2:
            pt = px_geoms.loc[idx].representative_point()
            dots.append((kind, name, pt.x, pt.y))
    svg.append("</g>")

    # Dots: tiny countries that are in the data, plus any explicit markers from the YAML.
    for m in markers:
        x, y = to_px(float(m["lon"]), float(m["lat"]))
        dots.append((m.get("status", "visited"), html.escape(str(m["name"])), x, y))
    if dots:
        svg.append('<g class="map-markers">')
        for kind, name, x, y in dots:
            svg.append(
                f'<circle class="marker {kind}" cx="{x:.1f}" cy="{y:.1f}" r="3.5" data-name="{name}"></circle>'
            )
        svg.append("</g>")

    svg.append("</svg>")
    return "\n".join(svg)


PREVIEW_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>World map preview</title>
<style>
body {{ margin: 0; padding: 2rem; font-family: system-ui, sans-serif; background: #fff; color: #222; }}
@media (prefers-color-scheme: dark) {{ body {{ background: #14151a; color: #e6e6e6; }} }}
.world-map {{ max-width: 1000px; margin: 0 auto; display: block; }}
{css}
</style></head><body>
{svg}
<p class="map-legend"><span class="swatch visited"></span> Visited <span class="swatch planned"></span> Planned</p>
</body></html>
"""


# --------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", type=Path, default=ROOT / "_data" / "world_map.yml")
    ap.add_argument(
        "--out", type=Path, default=ROOT / "_includes" / "maps" / "world.svg"
    )
    ap.add_argument(
        "--preview",
        type=Path,
        help="also write a standalone HTML page for checking the result",
    )
    ap.add_argument(
        "--css", type=Path, default=ROOT / "assets" / "css" / "world-map.css"
    )
    ap.add_argument("--data-dir", type=Path, default=ROOT / "scripts" / "data")
    ap.add_argument("--resolution", choices=["110m", "50m"], default="110m")
    ap.add_argument(
        "--south",
        type=float,
        default=-60.0,
        help="latitude where the map is cut off (default -60)",
    )
    ap.add_argument(
        "--graticule", type=int, default=30, help="grid spacing in degrees, 0 = off"
    )
    ap.add_argument("--width", type=int, default=1000)
    ap.add_argument(
        "--antarctica",
        action="store_true",
        help="keep Antarctica (uses the full outline)",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="fail if a country in the YAML is not found",
    )
    args = ap.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    south = -90.0 if args.antarctica else args.south

    world = load_world(args.resolution, args.data_dir, args.antarctica)
    status, unmatched = resolve_status(world, config)

    for kind, item in unmatched:
        hint = suggest(item, list(build_lookup(world)))
        print(
            f"WARNING: {kind}: '{item}' not found in Natural Earth{hint}",
            file=sys.stderr,
        )
        if isinstance(item, bool):
            print(
                "         (YAML reads bare NO/ON/Y as booleans - use 3-letter codes or quote it)",
                file=sys.stderr,
            )
    if unmatched and args.strict:
        return 1

    svg = build_svg(
        world,
        status,
        config.get("markers") or [],
        south=south,
        graticule_step=args.graticule,
        width=args.width,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(svg + "\n", encoding="utf-8")
    counts = {
        k: sum(1 for v in status.values() if v == k) for k in ("visited", "planned")
    }
    print(
        f"Wrote {args.out} ({len(svg) / 1024:.0f} KB): {counts['visited']} visited, {counts['planned']} planned"
    )

    if args.preview:
        css = args.css.read_text(encoding="utf-8") if args.css.exists() else ""
        args.preview.write_text(PREVIEW_HTML.format(css=css, svg=svg), encoding="utf-8")
        print(f"Wrote {args.preview}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
