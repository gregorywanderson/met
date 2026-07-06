# Meteorological Data Utilities

This directory contains Python modules and student-facing notebooks for acquiring, archiving, and analyzing meteorological observations. Reusable network and file-handling code belongs in Python modules; notebooks contain documented workflows, site configuration, visualization, and interpretation.

The directory currently includes two independent workflows:

1. Ambient Weather observations from the NEIU integrated weather station.
2. Public precipitation archives that do not depend on CROCUS infrastructure.

CROCUS-specific notebooks and Sage/Waggle functions remain in the separate CROCUS repository.

## Files

| File | Purpose |
|---|---|
| `ambient_weather.py` | Reusable Ambient Weather REST API and annual-archive functions. It contains no NEIU-specific metadata or credentials. |
| `ambient_weather_archive.ipynb` | Operational and student-facing workflow for updating the NEIU Ambient Weather archive and producing a compact diagnostic plot. |
| `precipitation.py` | General-purpose functions for discovering and retrieving public precipitation observations from NCEI, IEM, USGS, and CoCoRaHS. |
| `precipitation.ipynb` | Student-facing comparison of public precipitation sources near a configurable location. |
| `.env.example` | Safe credential template. The real `.env` file is not committed. |
| `README.md` | Directory overview, setup, and data-handling conventions. |

Future notebooks may add detailed Ambient Weather analysis, MWRD rain-gauge investigation, Vaisala WXT workflows, radar, or other meteorological sources.

## Directory structure

```text
met/
├── .env.example
├── .gitignore
├── README.md
├── ambient_weather.py
├── ambient_weather_archive.ipynb
├── precipitation.py
├── precipitation.ipynb
└── data/
    ├── ambient/
    └── precipitation/
```

## Requirements

The current workflows require Python 3 and these packages:

```text
numpy
pandas
requests
matplotlib
python-dotenv
jupyter
```

Install them with conda or mamba:

```bash
conda install numpy pandas requests matplotlib python-dotenv jupyter
```

## Credentials

Create a local `.env` file in this directory:

```bash
AMBIENT_API_KEY=your-ambient-api-key
AMBIENT_APPLICATION_KEY=your-ambient-application-key
```

The committed `.env.example` should contain the same variable names with empty values:

```bash
AMBIENT_API_KEY=
AMBIENT_APPLICATION_KEY=
```

The repository's `.gitignore` should include:

```gitignore
.env
```

Check that Git is ignoring the real credential file:

```bash
git check-ignore -v .env
```

Never print tokens in notebook output or embed them in committed source files.

---

# Ambient Weather archive

## Annual files

The Ambient Weather notebook maintains one CSV file per **station-local calendar year**:

```text
data/ambient/
├── neiu_weather_station_2021.csv
├── neiu_weather_station_2022.csv
├── neiu_weather_station_2023.csv
├── neiu_weather_station_2024.csv
├── neiu_weather_station_2025.csv
└── neiu_weather_station_2026.csv
```

The filenames identify the NEIU integrated weather station without tying the archive to a specific console model. Vaisala WXT files elsewhere include `wxt` in their names, which distinguishes the sources.

The timestamp is stored as timezone-aware UTC in `time_utc`, but file years are assigned using `America/Chicago` local time. This prevents observations from the evening of December 31 from being placed in the next year's file merely because UTC has crossed midnight.

## Typical workflow

1. Load credentials from `.env`.
2. Confirm the account and selected device.
3. Test a small recent-data request.
4. Run the routine incremental update.
5. Enable historical backfilling only when extending the archive into earlier years.
6. Inspect the annual inventory and diagnostic plot.
7. Disable backfilling again after the historical pass.

Historical backfilling is disabled by default because a multi-year record requires many API requests.

## Reusing `ambient_weather.py`

Another notebook can request recent observations without maintaining the annual archive:

```python
from ambient_weather import AmbientWeatherClient, fetch_recent_data

client = AmbientWeatherClient(API_KEY, APPLICATION_KEY)
weather = fetch_recent_data(client, DEVICE_MAC, hours=48)
solar_radiation = weather["solarradiation"]
```

The returned DataFrame uses a UTC `DatetimeIndex`. Convert to local time only when needed for display or alignment:

```python
weather_local = weather.tz_convert("America/Chicago")
```

## Ambient data notes

- Annual files preserve the original Ambient Weather field names and units.
- SI columns are created in memory for plotting rather than replacing source values.
- Duplicate timestamps are removed during merges.
- File writes are atomic.
- Internal `passkey` and `loc` fields are excluded from student archives.
- Relative pressure remains provisional until the console elevation and pressure correction are verified.
- Older manual dashboard exports should be preserved unchanged before conversion to the annual archive schema.

## Ambient Weather references

- [Ambient Weather REST API documentation](https://ambientweather.docs.apiary.io/)
- [Ambient Weather API documentation repository](https://github.com/ambient-weather/api-docs)
- [Device Data Specifications](https://github.com/ambient-weather/api-docs/wiki/Device-Data-Specs)
- [Ambient Weather account page](https://ambientweather.net/account)

---

# Public precipitation workflow

## Scope

`precipitation.py` and `precipitation.ipynb` are explicitly independent of CROCUS. They accept ordinary coordinates and station identifiers rather than CROCUS site objects, Sage node identifiers, or Waggle dependencies.

The first version covers:

| Source | Resolution | Access | Primary use |
|---|---:|---|---|
| NOAA NCEI GHCN-Daily | Daily | Public files; no token required | Long, quality-reviewed climate records |
| IEM ASOS/AWOS/METAR | Hourly reports | Public | Recent and historical airport observations |
| USGS Water Data | Site-dependent continuous data | Public | Local hydrologic precipitation gauges |
| CoCoRaHS through IEM | Daily reports | Public | Spatially dense volunteer observations |
| MWRD Rain Gauge Viewer | Varies | Manual CSV export for now | Local Chicago-area gauges |

The NWS observations API is intentionally omitted because the IEM ASOS archive is more appropriate for the historical hourly workflow developed here.

## Module interface

The public functions include:

```python
find_ghcnd_stations(...)
get_ghcnd_precipitation(...)

find_asos_stations(...)
get_asos_hourly_precipitation(...)
get_asos_precipitation(...)  # raw METAR reports; do not sum

find_usgs_precipitation_sites(...)
get_usgs_precipitation(...)  # query by time_series_id

find_cocorahs_stations(...)
ghcnd_to_cocorahs_id(...)
get_cocorahs_precipitation(...)

aggregate_interval_precipitation_daily(...)
summarize_daily_precipitation_coverage(...)
```

All precipitation values are exposed in millimeters when the source unit can be interpreted safely. Source identifiers, trace flags, quality flags, qualifiers, approval status, and native values are retained where available.

### GHCN-Daily metadata cache

`find_ghcnd_stations()` uses NOAA's public `ghcnd-stations.txt` and `ghcnd-inventory.txt` files. On the first call, it downloads those files and creates a compact precipitation-only cache under `data/precipitation/cache/ghcnd/`. Download and parsing progress are printed. Later station searches use the compact cache and should be fast. Pass `refresh_cache=True` only when fresh station metadata are needed.

`get_ghcnd_precipitation()` downloads NOAA's public compressed `by_station` file for each selected station and caches it under `data/precipitation/cache/ghcnd/by_station/`. Each file contains the station's complete period of record, so a long-running station may require a comparatively large first download and parse even for a short requested date range. Subsequent calls filter the local station file to the requested dates. Neither GHCN-Daily station discovery nor observation retrieval requires an API token.

## Scientific cautions

Converting units does not make all datasets directly comparable.

- GHCN-Daily values follow station observing schedules and may not represent midnight-to-midnight totals.
- ASOS totals use IEM's computed hourly precipitation service. Raw METAR `p01i` reports can overlap within an hour and must not be summed. Trace values are retained separately. ASOS and CoCoRaHS station discovery currently searches one state IEM network at a time, even when the radius crosses a state boundary.
- CoCoRaHS reports are generally morning-to-morning manual accumulations and can be intermittent. Station discovery filters IEM archive metadata to the requested dates before ranking by distance. GHCN-Daily CoCoRaHS identifiers such as `US1ILCK0323` are cross-walked to the native form `IL-CK-323` so that active gauges found through GHCN-Daily can also be queried through IEM.
- USGS parameter `00045` is commonly published as a continuous `Decumulated` interval series. Discovery uses the time-series metadata endpoint, filters each series by period-of-record overlap, and retrieves observations by `time_series_id`.
- A single USGS monitoring location can operate multiple primary precipitation gauges or sublocations. Those series are retained and labeled separately; they must not be summed or averaged merely because they share a location identifier.
- USGS daily totals are formed only from nonoverlapping interval values. The notebook infers each series' nominal interval, calculates daily interval coverage, and masks days below the configured completeness threshold.
- The final comparison reports both each series' available-data total and a common-valid-day total calculated over the intersection of dates valid for every selected series. This prevents an outage during a major storm from appearing to be an instrument disagreement. Dates excluded from the common comparison are displayed explicitly.
- Precipitation can vary sharply over short distances during convective storms.
- Missing intervals should remain missing rather than being filled by interpolation without a defensible scientific reason.

## Notebook workflow

1. Configure a location, date range, state, and search radius.
2. Discover nearby stations independently for each network.
3. Inspect station distance, time-series identity, and period-of-record metadata.
4. Retrieve observations from selected stations or time series.
5. Preserve source-specific flags, timing information, and sublocation identifiers.
6. Calculate GHCN-Daily reporting coverage before choosing a reference station.
7. Aggregate interval data only when its semantics support aggregation and retain coverage diagnostics.
8. Keep colocated USGS gauges as separate series.
9. Compare networks both over all available observations and over common valid dates.
10. Document differences in reporting periods and excluded dates.
11. Save figures under `figures/precipitation/`.
12. Load a manually downloaded MWRD CSV only after inspecting its schema; leaving `MWRD_CSV = None` is an expected no-op.

## Public precipitation references

### NOAA NCEI

- [GHCN-Daily](https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily)
- [GHCN-Daily file documentation](https://www.ncei.noaa.gov/pub/data/ghcn/daily/readme.txt)
- [GHCN-Daily by-station documentation](https://www.ncei.noaa.gov/pub/data/ghcn/daily/readme-by_station.txt)
- [GHCN-Daily public file directory](https://www.ncei.noaa.gov/pub/data/ghcn/daily/)

### Iowa Environmental Mesonet

- [IEM API documentation](https://mesonet.agron.iastate.edu/api/)
- [Station-network GeoJSON service](https://mesonet.agron.iastate.edu/geojson/network.py?help=)
- [ASOS request documentation](https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?help=)
- [Hourly precipitation request documentation](https://mesonet.agron.iastate.edu/cgi-bin/request/hourlyprecip.py?help=)
- [ASOS precipitation notes](https://mesonet.agron.iastate.edu/ASOS/precipnote.phtml)
- [Daily-summary request documentation](https://mesonet.agron.iastate.edu/cgi-bin/request/daily.py?help=)
- [IEM CoCoRaHS archive](https://mesonet.agron.iastate.edu/cocorahs/)

### USGS and regional data

- [USGS Water Data APIs](https://api.waterdata.usgs.gov/)
- [USGS OGC API guide](https://api.waterdata.usgs.gov/docs/ogcapi/)
- [USGS time-series metadata](https://api.waterdata.usgs.gov/ogcapi/v0/collections/time-series-metadata?f=html)
- [USGS time-series metadata schema](https://api.waterdata.usgs.gov/ogcapi/v0/collections/time-series-metadata/schema?f=html)
- [USGS continuous values](https://api.waterdata.usgs.gov/ogcapi/v0/collections/continuous?f=html)
- [CoCoRaHS](https://www.cocorahs.org/)
- [MWRD Rain Gauge Viewer](https://gispub.mwrd.org/raingaugeviewer/)

## Development conventions

- Keep network access and reusable transformations in modules.
- Keep location configuration, explanation, examples, and plots in notebooks.
- Use timezone-aware timestamps for subdaily observations.
- Use descriptive snake-case filenames and function names.
- Preserve raw manual downloads before cleaning.
- Add a documented test or small validation whenever an upstream service changes.
