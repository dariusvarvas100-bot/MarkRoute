# Prepare survey flags for MarkRoute

`prepare_survey.py` creates the nine-column CSV accepted by the [MarkRoute demo](https://dariusvarvas100-bot.github.io/MarkRoute/demo.html). Run it with Python 3.10 or newer, NumPy, and Pillow. Both libraries are in the repository requirements.

```sh
python -m pip install numpy Pillow
python prepare_survey.py flagged_marks.csv markroute_survey.csv --tile-cache terrain_cache
```

Use one row per flagged location. A field CSV can use this header and layout. The sample values illustrate the format.

```csv
site_id,latitude,longitude,environment,waterbody,elev_ft,vertical_datum
R01,35.1040,-77.0620,Riverine,Example Creek,16.8,NAVD88
R02,35.1081,-77.0670,Riverine,Example Creek,,
C01,35.0490,-76.9620,Coastal,Example Bay,6.2,NAVD88
C02,35.0537,-76.9685,Coastal,Example Bay,,
```

- Use decimal-degree latitude and longitude. Use `Coastal` or `Riverine`. The waterbody name is optional.
- Enter surveyed peak water-surface elevations in NAVD88 feet. Leave `elev_ft` blank at unmeasured flags and include one measured flag for each environment in the file. If the elevations' datum column is missing, add `--confirm-navd88-feet` after checking the survey metadata. An explicitly different datum or unit is rejected. The script performs no vertical conversion and leaves unmeasured elevations blank.
- Keep each file to one flood event of at most 800 marks. The CLI reports duplicate IDs or coordinates and preserves row order. Use a local survey on one side of the antimeridian and within the Mercator raster footprint.

For an official export from the [USGS Flood Event Viewer](https://apps.usgs.gov/fev/data-download-portal), download **High-Water Marks** as CSV for one event. The CLI recognizes the standard export and uses each high-water-mark ID, since several marks can share a survey site ID. It uses the mark coordinates, `environment`, `waterbody`, `elev_ft`, and `vertical_datum_name`. Keep elevation records whose coordinates, environment, and vertical datum have been confirmed. If necessary, explicitly specify the format.

```sh
python prepare_survey.py fev_export.csv markroute_survey.csv --format fev --tile-cache terrain_cache
python prepare_survey.py flagged_marks.csv markroute_survey.csv --tile-cache terrain_cache --offline
```

The CLI samples the [public Mapzen Terrarium tiles](https://registry.opendata.aws/terrain-tiles/) with the same zoom-12 Web Mercator projection and calculations as `terrain.py`. It decodes the selected pixel, the median of the clipped 7×7 neighborhood, and the 90th-minus-10th percentile spread of the clipped 17×17 neighborhood. It converts raster metres to feet. Mapzen combines [different elevation sources](https://github.com/tilezen/joerd/blob/master/docs/data-sources.md) and the PNGs carry no per-pixel datum metadata. Treat the derived values as the model's terrain features. Surveyed depths or terrain elevations for engineering use require independently verified survey and terrain data.

If a cache is provided, the CLI reuses `x_y.png` files there and saves newly downloaded tiles for repeatability. `--offline` uses only cached files. The network fetches public tile URLs only. Input rows and readings stay local.

Verification used the 294 published Hurricane Ian replay sites. All three terrain features matched at every site and the browser loaded the resulting CSV with 74 measured and 220 unmeasured marks. The CLI also accepted the [official Colorado event 11 CSV](https://api.waterdata.usgs.gov/stnservices/hwms/filteredhwms/?event=11&format=csv) and preserved all 32 source high-water-mark IDs and readings.
