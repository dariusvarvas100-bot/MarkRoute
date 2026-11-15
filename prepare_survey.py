"""Prepare an experimental MarkRoute survey CSV from flags or a USGS FEV export."""
import argparse
from collections import defaultdict
import csv
import io
import math
from pathlib import Path
import re
import sys
import tempfile
from urllib.request import urlopen

import numpy as np
from PIL import Image


ZOOM = 12
TILE_SIZE = 256
FEET_PER_METRE = 3.280839895
TERRARIUM = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium"
MAX_LATITUDE = 85.0511287798066
MAX_ELEVATION_FT = 60000
MAX_SITES = 800
MAX_OUTPUT_BYTES = 2 * 1024 * 1024
NUMBER = re.compile(r"^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:e[+-]?[0-9]+)?$", re.IGNORECASE)
ENVIRONMENT = {"coastal": "Coastal", "riverine": "Riverine"}
VERTICAL_DATUM = {"navd88", "navd1988", "northamericanverticaldatumof1988", "northamericanverticaldatum1988"}
VERTICAL_UNITS = {"ft", "feet", "foot", "ussurveyfoot", "ussurveyfeet", "internationalfoot", "internationalfeet"}
OUTPUT_COLUMNS = (
    "site_id", "latitude", "longitude", "environment", "waterbody",
    "terrain_ft", "terrain_median_ft", "relief_ft", "elev_ft",
)


def column_name(value):
    return re.sub(r"[^a-z0-9]+", "_", value.strip().casefold()).strip("_")


def token(value):
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def number(value, name, low, high):
    if not NUMBER.fullmatch(value):
        raise ValueError(f"{name} must be a finite decimal number")
    parsed = float(value)
    if not math.isfinite(parsed) or not low <= parsed <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return parsed


def choose_column(columns, aliases, *, required=True):
    match = next((name for name in aliases if name in columns), None)
    if required and match is None:
        raise ValueError(f"Missing column '{aliases[0]}'")
    return match


def read_marks(source, input_format, confirmed_datum):
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        header = [column_name(value) for value in next(reader, [])]
        if not header or any(not name for name in header) or len(set(header)) != len(header):
            raise ValueError("The CSV needs distinct, nonempty column names")
        columns = set(header)
        if input_format == "auto":
            markers = {"hwm_id", "hwm_type_name", "hwm_quality_id", "hwm_quality_name", "latitude_dd"}
            input_format = "fev" if markers & columns else "flags"
        fev = input_format == "fev"
        id_column = choose_column(columns, ("hwm_id", "id", "site_id") if fev else ("site_id",))
        latitude = choose_column(columns, ("latitude", "latitude_dd") if fev else ("latitude",))
        longitude = choose_column(columns, ("longitude", "longitude_dd") if fev else ("longitude",))
        environment = choose_column(columns, ("environment", "hwm_environment") if fev else ("environment",))
        waterbody = choose_column(columns, ("waterbody", "water_body", "waterbody_name"), required=False)
        elevation = choose_column(columns, ("elev_ft", "elevation_ft"), required=False)
        datum_columns = columns & {"vertical_datum_name", "verticaldatumname", "vertical_datum", "vdatum"}
        unit_columns = columns & {"elev_unit", "elevation_unit", "elevation_units", "vertical_unit", "vertical_units"}

        rows, ids, coordinates, measured_types = [], set(), set(), set()
        event_ids, event_names = set(), set()
        for fields in reader:
            line = reader.line_num
            if not fields or (len(fields) == 1 and not fields[0].strip()):
                continue
            if len(rows) == MAX_SITES:
                raise ValueError(f"The browser accepts up to {MAX_SITES} marks per survey")
            if len(fields) != len(header):
                raise ValueError(f"Line {line}: found {len(fields)} fields, expected {len(header)}")
            record = {name: value.strip() for name, value in zip(header, fields, strict=True)}
            try:
                mark_id = record[id_column]
                id_length = len(mark_id.encode("utf-16-le")) // 2
                if not 1 <= id_length <= 64 or any(ord(char) < 32 or ord(char) == 127 for char in mark_id):
                    raise ValueError("mark ID needs 1–64 characters without control characters")
                if mark_id in ids:
                    raise ValueError(f"duplicate mark ID '{mark_id}'")
                lat = number(record[latitude], "latitude", -MAX_LATITUDE, MAX_LATITUDE)
                lon = number(record[longitude], "longitude", -180, 180)
                if abs(lat) >= MAX_LATITUDE or lon == 180:
                    raise ValueError("coordinate is on the Web Mercator tile boundary")
                if (lat, lon) in coordinates:
                    raise ValueError("duplicate mark coordinates")
                group = ENVIRONMENT.get(record[environment].casefold())
                if group is None:
                    raise ValueError("environment must be Coastal or Riverine")
                reading = record.get(elevation, "")
                if reading:
                    datums = {token(record[name]) for name in datum_columns if record[name]}
                    units = {token(record[name]) for name in unit_columns if record[name]}
                    if datums - VERTICAL_DATUM or units - VERTICAL_UNITS:
                        raise ValueError("measured elevation must already be in NAVD88 feet")
                    if not datums and not confirmed_datum:
                        raise ValueError("measured elevation has no datum. Supply NAVD88 metadata or use "
                                         "--confirm-navd88-feet after verifying the survey")
                    reading = number(reading, "elev_ft", -MAX_ELEVATION_FT, MAX_ELEVATION_FT)
                    measured_types.add(group)
            except ValueError as error:
                raise ValueError(f"Line {line}: {error}") from error

            ids.add(mark_id)
            coordinates.add((lat, lon))
            event_id, event_name = record.get("event_id", ""), record.get("event_name", "")
            if event_id:
                event_ids.add(event_id)
            if event_name:
                event_names.add(event_name.casefold())
            rows.append({"site_id": mark_id, "latitude": lat, "longitude": lon,
                         "environment": group, "waterbody": record.get(waterbody, ""), "elev_ft": reading})

    if not rows:
        raise ValueError("The CSV contains no survey marks")
    if len(event_ids) > 1 or (not event_ids and len(event_names) > 1):
        raise ValueError("Export or prepare one flood event per CSV")
    unsupported = sorted({row["environment"] for row in rows} - measured_types)
    if unsupported:
        raise ValueError(f"Include at least one measured mark in each environment. Missing: {', '.join(unsupported)}")
    longitudes = [row["longitude"] for row in rows]
    if max(longitudes) - min(longitudes) > 180:
        raise ValueError("Split surveys that span the antimeridian")
    return rows


def web_coordinate(latitude, longitude):
    dimension = 2 ** ZOOM * TILE_SIZE
    x = (longitude + 180) / 360 * dimension
    y = (1 - np.arcsinh(np.tan(np.deg2rad(latitude))) / np.pi) / 2 * dimension
    maximum = dimension - 1
    return np.clip(x.astype(int), 0, maximum), np.clip(y.astype(int), 0, maximum)


def decode_tile(content):
    with Image.open(io.BytesIO(content)) as image:
        if image.format != "PNG" or image.size != (TILE_SIZE, TILE_SIZE):
            raise ValueError("expected a 256×256 Terrarium PNG")
        red, green, blue = np.asarray(image.convert("RGB"), dtype=np.float32).transpose(2, 0, 1)
    return (red * 256 + green + blue / 256 - 32768) * FEET_PER_METRE


def write_atomically(path, contents):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(contents)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_tile(key, cache, offline):
    x, y = key
    url = f"{TERRARIUM}/{ZOOM}/{x}/{y}.png"
    cached = cache / f"{x}_{y}.png" if cache is not None else None
    try:
        if cached is not None and cached.is_file():
            return decode_tile(cached.read_bytes())
        if offline:
            raise ValueError(f"missing cached tile {cached.name}")
        with urlopen(url, timeout=22) as response:
            contents = response.read(1024 * 1024 + 1)
        if len(contents) > 1024 * 1024:
            raise ValueError("tile download exceeded 1 MiB")
        raster = decode_tile(contents)
        if cached is not None:
            write_atomically(cached, contents)
        return raster
    except (OSError, ValueError) as error:
        raise ValueError(f"Terrain tile {ZOOM}/{x}/{y}: {error}") from error


def attach_terrain(rows, cache, offline):
    x, y = web_coordinate(np.asarray([row["latitude"] for row in rows]),
                          np.asarray([row["longitude"] for row in rows]))
    by_tile = defaultdict(list)
    for row, px, py in zip(rows, x, y, strict=True):
        by_tile[(px // TILE_SIZE, py // TILE_SIZE)].append((row, px % TILE_SIZE, py % TILE_SIZE))
    if cache is not None and not offline:
        cache.mkdir(parents=True, exist_ok=True)
    for key, samples in by_tile.items():
        raster = read_tile(key, cache, offline)
        for row, px, py in samples:
            local = raster[max(0, py - 3):py + 4, max(0, px - 3):px + 4]
            surrounding = raster[max(0, py - 8):py + 9, max(0, px - 8):px + 9]
            terrain = (float(raster[py, px]), float(np.median(local)),
                       float(np.percentile(surrounding, 90) - np.percentile(surrounding, 10)))
            for column, value, low in zip(OUTPUT_COLUMNS[5:8], terrain, (-MAX_ELEVATION_FT, -MAX_ELEVATION_FT, 0), strict=True):
                if not math.isfinite(value) or not low <= value <= MAX_ELEVATION_FT:
                    raise ValueError(f"Mark {row['site_id']}: terrain value '{column}' is outside the browser's bounds")
                row[column] = value
    return len(by_tile)


def prepare(input_path, output_path, input_format, confirmed_datum, cache, offline):
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Choose an output file different from the source CSV")
    if offline and cache is None:
        raise ValueError("--offline requires --tile-cache")
    rows = read_marks(input_path, input_format, confirmed_datum)
    tiles = attach_terrain(rows, cache, offline)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=OUTPUT_COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
    content = buffer.getvalue().encode("utf-8")
    if len(content) > MAX_OUTPUT_BYTES:
        raise ValueError("The result is larger than the browser’s 2 MiB import limit")
    write_atomically(output_path, content)
    return len(rows), tiles


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("input", type=Path, help="Raw flagged-survey CSV or a USGS Flood Event Viewer high-water-mark CSV")
    cli.add_argument("output", type=Path, help="Destination for the MarkRoute import-ready CSV")
    cli.add_argument("--format", choices=("auto", "flags", "fev"), default="auto", help="Source kind (default: auto)")
    cli.add_argument("--confirm-navd88-feet", action="store_true",
                     help="Confirm the datum and units if your verified readings omit datum metadata")
    cli.add_argument("--tile-cache", type=Path, help="Read and retain Mapzen tiles in this directory")
    cli.add_argument("--offline", action="store_true", help="Require all tiles in the local cache and skip network access")
    args = cli.parse_args()
    try:
        sites, tiles = prepare(args.input, args.output, args.format, args.confirm_navd88_feet, args.tile_cache, args.offline)
    except (ValueError, OSError, csv.Error) as error:
        cli.exit(2, f"prepare_survey: {error}\n")
    print(f"Wrote {sites} survey marks using {tiles} terrain tiles to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
