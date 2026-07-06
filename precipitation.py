"""Public precipitation-data access utilities.

This module provides reusable functions for discovering and retrieving
precipitation observations from public, non-CROCUS data sources:

* NOAA NCEI Global Historical Climatology Network-Daily (GHCN-Daily)
* Iowa Environmental Mesonet (IEM) ASOS/AWOS/METAR archive
* USGS Water Data OGC APIs
* IEM's archive of CoCoRaHS daily reports

The functions return tidy pandas DataFrames, use millimeters for
precipitation, and retain source identifiers and quality metadata where the
source provides them. Network requests are made only when a function is
called; importing the module has no network side effects.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Mapping, Sequence
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

GHCND_METADATA_BASE_URL = (
    "https://www.ncei.noaa.gov/pub/data/ghcn/daily"
)
GHCND_STATIONS_FILENAME = "ghcnd-stations.txt"
GHCND_INVENTORY_FILENAME = "ghcnd-inventory.txt"
GHCND_PRCP_CACHE_FILENAME = "ghcnd-prcp-stations.csv.gz"
GHCND_BY_STATION_BASE_URL = (
    "https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station"
)
IEM_NETWORK_URL = "https://mesonet.agron.iastate.edu/geojson/network.py"
IEM_ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
IEM_HOURLY_PRECIP_URL = (
    "https://mesonet.agron.iastate.edu/cgi-bin/request/hourlyprecip.py"
)
IEM_DAILY_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/daily.py"
USGS_OGC_BASE_URL = "https://api.waterdata.usgs.gov/ogcapi/v0"
USGS_PRECIPITATION_PARAMETER = "00045"

DEFAULT_TIMEOUT = 60
EARTH_RADIUS_KM = 6371.0088
INCH_TO_MM = 25.4
TRACE_INCHES = 0.0001


class PrecipitationDataError(RuntimeError):
    """Raised when a precipitation service returns an unusable response."""


def _as_list(values: str | Iterable[str]) -> list[str]:
    """Return one or more identifiers as a list of nonempty strings."""
    if isinstance(values, str):
        result = [values]
    else:
        result = [str(value) for value in values]

    result = [value.strip() for value in result if value and value.strip()]
    if not result:
        raise ValueError("At least one station identifier is required.")
    return result


def _is_trace_value(values: pd.Series) -> pd.Series:
    """Return a boolean mask for IEM's 0.0001-inch trace sentinel."""
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.Series(
        np.isclose(
            numeric.to_numpy(dtype=float, na_value=np.nan),
            TRACE_INCHES,
            rtol=0.0,
            atol=1e-9,
            equal_nan=False,
        ),
        index=values.index,
    )


def _request(
    url: str,
    *,
    params: Mapping[str, Any] | Sequence[tuple[str, Any]] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: int | float = DEFAULT_TIMEOUT,
    session: requests.Session | None = None,
    attempts: int = 3,
) -> requests.Response:
    """Issue a GET request with modest retry handling.

    Retries are limited to transient connection failures, HTTP 429, and
    server-side 5xx responses. Client-side errors are raised immediately.
    """
    requester = session or requests
    last_error: Exception | None = None

    for attempt in range(attempts):
        try:
            response = requester.get(
                url,
                params=params,
                headers=headers,
                timeout=timeout,
            )
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            status = getattr(getattr(exc, "response", None), "status_code", None)
            transient = status in {429, 500, 502, 503, 504} or status is None
            if not transient or attempt == attempts - 1:
                break
            time.sleep(2**attempt)

    raise PrecipitationDataError(f"Request failed for {url}: {last_error}")


def haversine_distance_km(
    latitude: float,
    longitude: float,
    station_latitude: float,
    station_longitude: float,
) -> float:
    """Calculate great-circle distance between two points in kilometers."""
    lat1, lon1, lat2, lon2 = map(
        math.radians,
        (latitude, longitude, station_latitude, station_longitude),
    )
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def bounding_box(
    latitude: float,
    longitude: float,
    radius_km: float,
) -> tuple[float, float, float, float]:
    """Return a west, south, east, north box around a point.

    The box is an approximation used for service-side station filtering.
    Returned stations should still be filtered by great-circle distance.
    """
    if radius_km <= 0:
        raise ValueError("radius_km must be positive.")

    latitude_delta = radius_km / 111.32
    longitude_scale = max(math.cos(math.radians(latitude)), 0.01)
    longitude_delta = radius_km / (111.32 * longitude_scale)

    return (
        longitude - longitude_delta,
        latitude - latitude_delta,
        longitude + longitude_delta,
        latitude + latitude_delta,
    )


def _download_metadata_file(
    url: str,
    path: Path,
    *,
    timeout: int | float = DEFAULT_TIMEOUT,
    session: requests.Session | None = None,
    verbose: bool = True,
) -> Path:
    """Download a metadata file atomically and report visible progress."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".part")
    requester = session or requests
    last_error: Exception | None = None

    if verbose:
        print(f"Downloading {path.name} from NOAA NCEI...")

    for attempt in range(3):
        try:
            response = requester.get(
                url,
                stream=True,
                timeout=(10, timeout),
            )
            response.raise_for_status()

            total_bytes = int(response.headers.get("content-length", 0) or 0)
            downloaded_bytes = 0
            next_report = 5 * 1024 * 1024

            with temporary_path.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    output.write(chunk)
                    downloaded_bytes += len(chunk)
                    if verbose and downloaded_bytes >= next_report:
                        downloaded_mb = downloaded_bytes / 1024**2
                        if total_bytes:
                            total_mb = total_bytes / 1024**2
                            print(
                                f"  {downloaded_mb:.1f} of {total_mb:.1f} MB",
                                flush=True,
                            )
                        else:
                            print(f"  {downloaded_mb:.1f} MB", flush=True)
                        next_report += 5 * 1024 * 1024

            temporary_path.replace(path)
            if verbose:
                size_mb = path.stat().st_size / 1024**2
                print(f"Cached {path.name} ({size_mb:.1f} MB).")
            return path
        except (OSError, requests.RequestException) as exc:
            last_error = exc
            temporary_path.unlink(missing_ok=True)
            if attempt < 2:
                time.sleep(2**attempt)

    raise PrecipitationDataError(
        f"Could not download GHCN-Daily metadata from {url}: {last_error}"
    )


def _read_ghcnd_station_metadata(path: Path) -> pd.DataFrame:
    """Read NOAA's fixed-width GHCN-Daily station metadata file."""
    columns = [
        "station_id",
        "latitude",
        "longitude",
        "elevation_m",
        "state",
        "station_name",
        "gsn_flag",
        "hcn_crn_flag",
        "wmo_id",
    ]
    stations = pd.read_fwf(
        path,
        colspecs=[
            (0, 11),
            (12, 20),
            (21, 30),
            (31, 37),
            (38, 40),
            (41, 71),
            (72, 75),
            (76, 79),
            (80, 85),
        ],
        names=columns,
        dtype={"station_id": "string", "state": "string"},
    )
    for column in ("station_id", "state", "station_name"):
        stations[column] = stations[column].astype("string").str.strip()
    for column in ("latitude", "longitude", "elevation_m"):
        stations[column] = pd.to_numeric(stations[column], errors="coerce")
    stations.loc[stations["elevation_m"] <= -999, "elevation_m"] = pd.NA
    return stations


def _read_ghcnd_precipitation_inventory(path: Path) -> pd.DataFrame:
    """Read precipitation periods of record from GHCN-Daily inventory."""
    inventory = pd.read_fwf(
        path,
        colspecs=[
            (0, 11),
            (12, 20),
            (21, 30),
            (31, 35),
            (36, 40),
            (41, 45),
        ],
        names=[
            "station_id",
            "latitude_inventory",
            "longitude_inventory",
            "element",
            "first_year",
            "last_year",
        ],
        dtype={"station_id": "string", "element": "string"},
    )
    inventory["station_id"] = inventory["station_id"].str.strip()
    inventory["element"] = inventory["element"].str.strip()
    inventory = inventory.loc[inventory["element"] == "PRCP"].copy()
    for column in ("first_year", "last_year"):
        inventory[column] = pd.to_numeric(
            inventory[column],
            errors="coerce",
        ).astype("Int64")
    return inventory[
        ["station_id", "first_year", "last_year"]
    ].drop_duplicates("station_id")


def _load_ghcnd_precipitation_stations(
    cache_dir: str | Path,
    *,
    refresh_cache: bool = False,
    timeout: int | float = DEFAULT_TIMEOUT,
    session: requests.Session | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load a compact cached table of GHCN-Daily PRCP stations."""
    cache_path = Path(cache_dir)
    stations_path = cache_path / GHCND_STATIONS_FILENAME
    inventory_path = cache_path / GHCND_INVENTORY_FILENAME
    compact_path = cache_path / GHCND_PRCP_CACHE_FILENAME

    if compact_path.exists() and not refresh_cache:
        if verbose:
            print(f"Using cached GHCN-Daily metadata: {compact_path}")
        stations = pd.read_csv(
            compact_path,
            dtype={"station_id": "string", "state": "string"},
        )
        for column in ("first_year", "last_year"):
            stations[column] = pd.to_numeric(
                stations[column],
                errors="coerce",
            ).astype("Int64")
        return stations

    files = [
        (
            stations_path,
            f"{GHCND_METADATA_BASE_URL}/{GHCND_STATIONS_FILENAME}",
        ),
        (
            inventory_path,
            f"{GHCND_METADATA_BASE_URL}/{GHCND_INVENTORY_FILENAME}",
        ),
    ]
    for path, url in files:
        if refresh_cache or not path.exists():
            _download_metadata_file(
                url,
                path,
                timeout=timeout,
                session=session,
                verbose=verbose,
            )

    if verbose:
        print("Reading and combining GHCN-Daily metadata...")
    stations = _read_ghcnd_station_metadata(stations_path)
    inventory = _read_ghcnd_precipitation_inventory(inventory_path)
    stations = stations.merge(inventory, on="station_id", how="inner")

    compact_path.parent.mkdir(parents=True, exist_ok=True)
    stations.to_csv(compact_path, index=False, compression="gzip")
    if verbose:
        print(f"Cached compact precipitation inventory: {compact_path}")
    return stations


def find_ghcnd_stations(
    latitude: float,
    longitude: float,
    *,
    radius_km: float = 50,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    n_stations: int | None = 20,
    cache_dir: str | Path = "data/precipitation/cache/ghcnd",
    refresh_cache: bool = False,
    timeout: int | float = DEFAULT_TIMEOUT,
    session: requests.Session | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Find nearby GHCN-Daily stations that report precipitation.

    Station discovery uses NOAA's public ``ghcnd-stations.txt`` and
    ``ghcnd-inventory.txt`` files rather than the slower token-based CDO
    station endpoint. The source files and a compact precipitation-only table
    are cached locally. Set ``refresh_cache=True`` to download fresh metadata.

    Parameters
    ----------
    latitude, longitude
        Search location in decimal degrees.
    radius_km
        Maximum station distance after exact distance filtering.
    start, end
        Optional requested period. Stations must have an unflagged PRCP period
        of record that overlaps these years.
    n_stations
        Maximum number of nearest stations returned. ``None`` returns all
        matching stations within ``radius_km``.
    cache_dir
        Directory used for NOAA metadata and the compact parsed cache.
    refresh_cache
        Download fresh metadata and rebuild the compact cache.
    timeout
        Read timeout in seconds for each metadata download attempt.
    session
        Optional requests session, primarily useful for testing.
    verbose
        Print download, cache, and parsing progress.

    Returns
    -------
    pandas.DataFrame
        One row per station, sorted by ``distance_km``.
    """
    if n_stations is not None and n_stations <= 0:
        raise ValueError("n_stations must be positive or None.")

    stations = _load_ghcnd_precipitation_stations(
        cache_dir,
        refresh_cache=refresh_cache,
        timeout=timeout,
        session=session,
        verbose=verbose,
    )

    west, south, east, north = bounding_box(latitude, longitude, radius_km)
    stations = stations.loc[
        stations["latitude"].between(south, north)
        & stations["longitude"].between(west, east)
    ].copy()

    if start is not None:
        start_year = pd.Timestamp(start).year
        stations = stations.loc[stations["last_year"] >= start_year].copy()
    if end is not None:
        end_year = pd.Timestamp(end).year
        stations = stations.loc[stations["first_year"] <= end_year].copy()

    if stations.empty:
        return pd.DataFrame(
            columns=[
                "station_id",
                "station_name",
                "state",
                "latitude",
                "longitude",
                "elevation_m",
                "first_year",
                "last_year",
                "distance_km",
            ]
        )

    stations["distance_km"] = stations.apply(
        lambda row: haversine_distance_km(
            latitude,
            longitude,
            row["latitude"],
            row["longitude"],
        ),
        axis=1,
    )
    stations = stations.loc[stations["distance_km"] <= radius_km].copy()
    stations = stations.sort_values("distance_km").reset_index(drop=True)
    if n_stations is not None:
        stations = stations.head(n_stations).copy()

    columns = [
        "station_id",
        "station_name",
        "state",
        "latitude",
        "longitude",
        "elevation_m",
        "first_year",
        "last_year",
        "distance_km",
    ]
    return stations.reindex(columns=columns)


def _normalize_ghcnd_station_id(station_id: str) -> str:
    """Return the bare 11-character GHCN-Daily station identifier."""
    normalized = station_id.strip()
    if normalized.upper().startswith("GHCND:"):
        normalized = normalized.split(":", 1)[1]
    if len(normalized) != 11:
        raise ValueError(
            "GHCN-Daily station IDs must contain 11 characters, such as "
            "USW00094846."
        )
    return normalized


def _download_ghcnd_station_file(
    station_id: str,
    path: Path,
    *,
    timeout: int | float = DEFAULT_TIMEOUT,
    session: requests.Session | None = None,
    verbose: bool = True,
    attempts: int = 3,
) -> Path:
    """Download one compressed GHCN-Daily by-station file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".part")
    url = f"{GHCND_BY_STATION_BASE_URL}/{station_id}.csv.gz"
    requester = session or requests
    last_error: Exception | None = None

    if verbose:
        print(f"Downloading GHCN-Daily observations for {station_id}...")

    for attempt in range(attempts):
        try:
            response = requester.get(
                url,
                stream=True,
                timeout=(10, timeout),
            )
            response.raise_for_status()
            with temporary_path.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output.write(chunk)
            temporary_path.replace(path)
            break
        except (OSError, requests.RequestException) as exc:
            last_error = exc
            temporary_path.unlink(missing_ok=True)
            status = getattr(
                getattr(exc, "response", None),
                "status_code",
                None,
            )
            transient = status in {429, 500, 502, 503, 504} or status is None
            if not transient or attempt == attempts - 1:
                raise PrecipitationDataError(
                    "Could not download GHCN-Daily station file for "
                    f"{station_id}: {last_error}"
                ) from exc
            time.sleep(2**attempt)

    if verbose:
        size_mb = path.stat().st_size / 1024**2
        print(f"Cached {path.name} ({size_mb:.1f} MB).")
    return path


def _read_ghcnd_station_precipitation(
    path: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Read and subset PRCP observations from a by-station CSV file."""
    names = [
        "station_id",
        "date",
        "element",
        "value_tenths_mm",
        "measurement_flag",
        "quality_flag",
        "source_flag",
        "observation_time",
    ]
    data = pd.read_csv(
        path,
        compression="gzip",
        header=None,
        names=names,
        dtype={
            "station_id": "string",
            "date": "string",
            "element": "string",
            "measurement_flag": "string",
            "quality_flag": "string",
            "source_flag": "string",
            "observation_time": "string",
        },
    )
    data = data.loc[data["element"].eq("PRCP")].copy()
    data["date"] = pd.to_datetime(
        data["date"],
        format="%Y%m%d",
        errors="coerce",
    )
    data = data.loc[data["date"].between(start, end, inclusive="both")]
    if data.empty:
        return data

    data["precip_mm"] = pd.to_numeric(
        data["value_tenths_mm"],
        errors="coerce",
    ) / 10.0
    flag_columns = [
        "measurement_flag",
        "quality_flag",
        "source_flag",
        "observation_time",
    ]
    for column in flag_columns:
        data[column] = data[column].replace({"": pd.NA}).astype("string")
    data["attributes"] = data[flag_columns].fillna("").agg(",".join, axis=1)
    data["source"] = "NOAA NCEI GHCN-Daily"
    return data


def get_ghcnd_precipitation(
    station_ids: str | Iterable[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    *,
    cache_dir: str | Path = "data/precipitation/cache/ghcnd/by_station",
    refresh_cache: bool = False,
    session: requests.Session | None = None,
    timeout: int | float = DEFAULT_TIMEOUT,
    verbose: bool = True,
) -> pd.DataFrame:
    """Retrieve GHCN-Daily precipitation from public by-station files.

    NOAA publishes one compressed CSV for the complete period of record of
    each GHCN-Daily station. Files are downloaded once and cached locally;
    subsequent calls filter the cached files to the requested dates. A
    long-running station can therefore require a comparatively large initial
    download and parse even for a short requested date range.

    Daily dates are retained as calendar dates without a timezone. A GHCN
    daily value does not necessarily represent midnight-to-midnight UTC or
    local time; the station's observing schedule determines the reporting
    interval.
    """
    stations = [
        _normalize_ghcnd_station_id(value)
        for value in _as_list(station_ids)
    ]
    start_date = pd.Timestamp(start).normalize().tz_localize(None)
    end_date = pd.Timestamp(end).normalize().tz_localize(None)
    if end_date < start_date:
        raise ValueError("end must be on or after start.")

    cache_path = Path(cache_dir)
    frames: list[pd.DataFrame] = []
    for station_id in stations:
        station_path = cache_path / f"{station_id}.csv.gz"
        if refresh_cache or not station_path.exists():
            _download_ghcnd_station_file(
                station_id,
                station_path,
                timeout=timeout,
                session=session,
                verbose=verbose,
            )
        elif verbose:
            print(f"Using cached GHCN-Daily observations: {station_path}")

        station_data = _read_ghcnd_station_precipitation(
            station_path,
            start_date,
            end_date,
        )
        if not station_data.empty:
            frames.append(station_data)

    columns = [
        "station_id",
        "precip_mm",
        "measurement_flag",
        "quality_flag",
        "source_flag",
        "observation_time",
        "attributes",
        "source",
    ]
    if not frames:
        empty = pd.DataFrame(columns=columns)
        empty.index = pd.DatetimeIndex([], name="date")
        return empty

    data = pd.concat(frames, ignore_index=True)
    data = data.set_index("date").sort_index()
    return data.reindex(columns=columns)


def _geojson_station_table(payload: Mapping[str, Any]) -> pd.DataFrame:
    """Convert an IEM network GeoJSON response to a station table."""
    rows: list[dict[str, Any]] = []
    for feature in payload.get("features", []):
        properties = dict(feature.get("properties") or {})
        coordinates = (feature.get("geometry") or {}).get("coordinates", [])
        longitude = coordinates[0] if len(coordinates) >= 2 else None
        latitude = coordinates[1] if len(coordinates) >= 2 else None
        rows.append(
            {
                "station_id": properties.get("sid")
                or properties.get("station")
                or properties.get("id"),
                "station_name": properties.get("sname")
                or properties.get("name")
                or properties.get("station_name"),
                "latitude": latitude,
                "longitude": longitude,
                "elevation_m": properties.get("elevation"),
                "network": properties.get("network"),
                "time_zone": properties.get("tzname")
                or properties.get("timezone"),
                "archive_begin": properties.get("archive_begin"),
                "archive_end": properties.get("archive_end"),
                "online": properties.get("online"),
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return pd.DataFrame(
            columns=[
                "station_id",
                "station_name",
                "latitude",
                "longitude",
                "elevation_m",
                "network",
                "time_zone",
                "archive_begin",
                "archive_end",
                "online",
            ]
        )
    for column in ("latitude", "longitude", "elevation_m"):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    return table.dropna(subset=["station_id", "latitude", "longitude"])


def get_iem_network_stations(
    network: str,
    *,
    only_online: bool = False,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Retrieve station metadata for an Iowa Mesonet network."""
    params = {"network": network.upper()}
    if only_online:
        params["only_online"] = "1"
    response = _request(IEM_NETWORK_URL, params=params, session=session)
    return _geojson_station_table(response.json())


def _nearest_stations(
    stations: pd.DataFrame,
    latitude: float,
    longitude: float,
    *,
    radius_km: float | None,
    n_stations: int | None,
) -> pd.DataFrame:
    """Add distance and return nearest station rows."""
    if stations.empty:
        result = stations.copy()
        result["distance_km"] = pd.Series(dtype=float)
        return result

    result = stations.copy()
    result["distance_km"] = result.apply(
        lambda row: haversine_distance_km(
            latitude,
            longitude,
            row["latitude"],
            row["longitude"],
        ),
        axis=1,
    )
    if radius_km is not None:
        result = result.loc[result["distance_km"] <= radius_km].copy()
    result = result.sort_values("distance_km").reset_index(drop=True)
    if n_stations is not None:
        result = result.head(n_stations).copy()
    return result


def find_asos_stations(
    latitude: float,
    longitude: float,
    *,
    state: str = "IL",
    radius_km: float | None = 100,
    n_stations: int | None = 10,
    only_online: bool = False,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Find nearby ASOS/AWOS/METAR stations in one state IEM network.

    The search does not automatically include neighboring-state networks,
    even when ``radius_km`` crosses a state boundary.
    """
    network = f"{state.upper()}_ASOS"
    stations = get_iem_network_stations(
        network,
        only_online=only_online,
        session=session,
    )
    return _nearest_stations(
        stations,
        latitude,
        longitude,
        radius_km=radius_km,
        n_stations=n_stations,
    )


def get_asos_hourly_precipitation(
    station_ids: str | Iterable[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    *,
    network: str,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Retrieve IEM-computed, nonoverlapping hourly precipitation totals.

    The IEM hourly-precipitation service derives one total for each station
    hour from processed METAR observations. This avoids double-counting the
    overlapping ``p01i`` values that can appear in multiple routine and
    special METAR reports during the same hour.

    Parameters
    ----------
    station_ids
        One station identifier or an iterable of identifiers.
    start, end
        Requested interval. Naive timestamps are interpreted as UTC.
    network
        IEM network identifier, such as ``"IL_ASOS"``.
    session
        Optional requests-compatible session, useful for testing.

    Returns
    -------
    pandas.DataFrame
        UTC DatetimeIndex named ``time_utc`` and columns ``station_id``,
        ``network``, ``precip_mm``, ``trace``, ``latitude``, ``longitude``,
        ``state``, and ``source``. The timestamp is the ending hour of the
        precipitation interval.

    Notes
    -----
    IEM uses 0.0001 inch as its trace sentinel. The returned numerical
    accumulation is set to 0 mm while ``trace`` remains True.
    """
    stations = _as_list(station_ids)
    network = str(network).strip().upper()
    if not network:
        raise ValueError("network must be a nonempty IEM network identifier.")

    start_time = pd.Timestamp(start)
    end_time = pd.Timestamp(end)
    if start_time.tzinfo is None:
        start_time = start_time.tz_localize("UTC")
    else:
        start_time = start_time.tz_convert("UTC")
    if end_time.tzinfo is None:
        end_time = end_time.tz_localize("UTC")
    else:
        end_time = end_time.tz_convert("UTC")
    if end_time < start_time:
        raise ValueError("end must be on or after start.")

    params: list[tuple[str, Any]] = [
        ("network", network),
        ("sts", start_time.isoformat().replace("+00:00", "Z")),
        ("ets", end_time.isoformat().replace("+00:00", "Z")),
        ("tz", "UTC"),
        ("lalo", "1"),
        ("st", "1"),
    ]
    params.extend(("station", station_id) for station_id in stations)

    response = _request(
        IEM_HOURLY_PRECIP_URL,
        params=params,
        session=session,
    )
    try:
        data = pd.read_csv(StringIO(response.text), comment="#")
    except pd.errors.EmptyDataError:
        data = pd.DataFrame()

    columns = [
        "station_id",
        "network",
        "precip_mm",
        "trace",
        "latitude",
        "longitude",
        "state",
        "source",
    ]
    if data.empty:
        empty = pd.DataFrame(columns=columns)
        empty.index = pd.DatetimeIndex([], tz="UTC", name="time_utc")
        return empty

    data.columns = [str(column).strip().lower() for column in data.columns]
    required = {"station", "valid", "precip_in"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise PrecipitationDataError(
            "The IEM hourly-precipitation response is missing expected "
            f"columns: {', '.join(missing)}."
        )

    data["time_utc"] = pd.to_datetime(
        data["valid"],
        errors="coerce",
        utc=True,
    )
    raw_precip = pd.to_numeric(data["precip_in"], errors="coerce")
    data["trace"] = _is_trace_value(raw_precip)
    data["precip_mm"] = raw_precip.mask(data["trace"], 0.0) * INCH_TO_MM
    data["station_id"] = data["station"].astype("string")
    if "network" not in data:
        data["network"] = network
    data["network"] = data["network"].astype("string")
    data["latitude"] = pd.to_numeric(data.get("lat"), errors="coerce")
    data["longitude"] = pd.to_numeric(data.get("lon"), errors="coerce")
    data["state"] = data.get(
        "st",
        pd.Series(pd.NA, index=data.index, dtype="string"),
    )
    data["source"] = "IEM computed hourly ASOS precipitation"

    data = data.dropna(subset=["time_utc"])
    data = data.loc[
        (data["time_utc"] >= start_time) & (data["time_utc"] <= end_time)
    ]
    data = data.sort_values(["station_id", "time_utc"])

    duplicated = data.duplicated(["station_id", "time_utc"], keep=False)
    if duplicated.any():
        examples = (
            data.loc[duplicated, ["station_id", "time_utc"]]
            .drop_duplicates()
            .head(5)
        )
        example_text = "; ".join(
            f"{row.station_id} at {row.time_utc}"
            for row in examples.itertuples(index=False)
        )
        raise PrecipitationDataError(
            "The IEM computed-hourly response contained duplicate "
            f"station-hours ({example_text})."
        )

    data = data.set_index("time_utc")
    return data[columns]


def get_asos_precipitation(
    station_ids: str | Iterable[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    *,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Retrieve raw precipitation fields from individual METAR reports.

    Warning
    -------
    Multiple reports in one hour can contain overlapping ``p01i``
    accumulations. Do not sum these rows to calculate hourly, daily, or
    cumulative precipitation. Use :func:`get_asos_hourly_precipitation`
    for nonoverlapping hourly totals.

    The raw ``p01i`` value is converted from inches to millimeters. IEM's
    numerical trace sentinel of 0.0001 inch is converted to 0 mm while a
    separate ``trace`` flag is retained.
    """
    stations = _as_list(station_ids)
    start_time = pd.Timestamp(start)
    end_time = pd.Timestamp(end)
    if start_time.tzinfo is None:
        start_time = start_time.tz_localize("UTC")
    else:
        start_time = start_time.tz_convert("UTC")
    if end_time.tzinfo is None:
        end_time = end_time.tz_localize("UTC")
    else:
        end_time = end_time.tz_convert("UTC")
    if end_time < start_time:
        raise ValueError("end must be on or after start.")

    params: list[tuple[str, Any]] = [
        ("data", "p01i"),
        ("data", "wxcodes"),
        ("sts", start_time.isoformat().replace("+00:00", "Z")),
        ("ets", end_time.isoformat().replace("+00:00", "Z")),
        ("tz", "UTC"),
        ("format", "onlycomma"),
        ("missing", "empty"),
        ("latlon", "no"),
        ("elev", "no"),
        ("direct", "no"),
    ]
    params.extend(("station", station_id) for station_id in stations)

    response = _request(IEM_ASOS_URL, params=params, session=session)
    try:
        data = pd.read_csv(StringIO(response.text), comment="#")
    except pd.errors.EmptyDataError:
        data = pd.DataFrame()

    columns = ["station_id", "precip_mm", "trace", "present_weather", "source"]
    if data.empty:
        empty = pd.DataFrame(columns=columns)
        empty.index = pd.DatetimeIndex([], tz="UTC", name="time_utc")
        return empty

    data.columns = [str(column).strip().lower() for column in data.columns]
    if "valid" not in data.columns:
        raise PrecipitationDataError(
            "The IEM ASOS response did not contain the expected 'valid' column."
        )
    if "p01i" not in data.columns:
        data["p01i"] = pd.NA

    data["time_utc"] = pd.to_datetime(data["valid"], errors="coerce", utc=True)
    raw_text = data["p01i"].astype("string").str.strip()
    raw_precip = pd.to_numeric(raw_text, errors="coerce")
    data["trace"] = raw_text.str.upper().eq("T") | _is_trace_value(raw_precip)
    data["precip_mm"] = raw_precip.mask(data["trace"], 0.0) * INCH_TO_MM
    if "station" in data:
        data["station_id"] = data["station"].astype("string")
    else:
        data["station_id"] = pd.Series(pd.NA, index=data.index, dtype="string")
    data["present_weather"] = data.get("wxcodes", pd.NA)
    data["source"] = "Iowa Environmental Mesonet ASOS"

    data = data.dropna(subset=["time_utc"])
    data = data.loc[
        (data["time_utc"] >= start_time) & (data["time_utc"] <= end_time)
    ]
    data = data.sort_values(["station_id", "time_utc"])
    data = data.drop_duplicates(["station_id", "time_utc"], keep="last")
    data = data.set_index("time_utc")
    return data[columns]


def ghcnd_to_cocorahs_id(station_id: str) -> str | None:
    """Convert a GHCN-Daily CoCoRaHS ID to the native CoCoRaHS form.

    GHCN-Daily represents CoCoRaHS stations with 11-character identifiers
    such as ``US1ILCK0323``. The corresponding CoCoRaHS and IEM identifier is
    ``IL-CK-323``. Non-CoCoRaHS GHCN-Daily identifiers return ``None``.
    """
    normalized = str(station_id).strip().upper()
    if normalized.startswith("GHCND:"):
        normalized = normalized.split(":", 1)[1]
    if not normalized.startswith("US1") or len(normalized) != 11:
        return None

    state = normalized[3:5]
    county = normalized[5:7]
    number_text = normalized[7:]
    if not number_text.isdigit():
        return None

    return f"{state}-{county}-{int(number_text)}"


def _normalize_cocorahs_station_id(station_id: str) -> str:
    """Normalize GHCN-Daily or native CoCoRaHS station identifiers."""
    normalized = str(station_id).strip().upper()
    converted = ghcnd_to_cocorahs_id(normalized)
    if converted is not None:
        return converted

    parts = normalized.split("-")
    if len(parts) == 3 and parts[2].isdigit():
        return f"{parts[0]}-{parts[1]}-{int(parts[2])}"
    return normalized


def find_cocorahs_stations(
    latitude: float,
    longitude: float,
    *,
    state: str = "IL",
    radius_km: float | None = 50,
    n_stations: int | None = 25,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Find nearby CoCoRaHS stations in an IEM state network.

    When ``start`` or ``end`` is supplied, station metadata are filtered to
    records whose IEM archive period overlaps the requested analysis period.
    This avoids filling a nearest-station list with gauges that ceased
    reporting before the dates of interest. The search is limited to the
    specified state's IEM network and does not automatically include
    neighboring states.
    """
    network = f"{state.upper()}_COCORAHS"
    stations = get_iem_network_stations(network, session=session)

    if stations.empty:
        return _nearest_stations(
            stations,
            latitude,
            longitude,
            radius_km=radius_km,
            n_stations=n_stations,
        )

    stations = stations.copy()
    stations["station_id"] = stations["station_id"].map(
        _normalize_cocorahs_station_id
    )
    stations["archive_begin"] = pd.to_datetime(
        stations["archive_begin"],
        errors="coerce",
    )
    stations["archive_end"] = pd.to_datetime(
        stations["archive_end"],
        errors="coerce",
    )

    if start is not None:
        start_time = pd.Timestamp(start).tz_localize(None)
        stations = stations.loc[
            stations["archive_end"].isna()
            | (stations["archive_end"] >= start_time)
        ].copy()
    if end is not None:
        end_time = pd.Timestamp(end).tz_localize(None)
        stations = stations.loc[
            stations["archive_begin"].isna()
            | (stations["archive_begin"] <= end_time)
        ].copy()

    return _nearest_stations(
        stations,
        latitude,
        longitude,
        radius_km=radius_km,
        n_stations=n_stations,
    )


def get_cocorahs_precipitation(
    station_ids: str | Iterable[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    *,
    state: str = "IL",
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Retrieve CoCoRaHS daily precipitation totals from IEM.

    CoCoRaHS values are usually manual 24-hour accumulations reported in the
    morning. The returned index is therefore a reporting date, not a precise
    UTC timestamp.
    """
    stations = [
        _normalize_cocorahs_station_id(station_id)
        for station_id in _as_list(station_ids)
    ]
    stations = list(dict.fromkeys(stations))
    start_date = pd.Timestamp(start).date().isoformat()
    end_date = pd.Timestamp(end).date().isoformat()
    network = f"{state.upper()}_COCORAHS"

    params: list[tuple[str, Any]] = [
        ("network", network),
        ("sts", start_date),
        ("ets", end_date),
        ("var", "precip_in"),
        ("format", "csv"),
        ("na", ""),
    ]
    params.extend(("stations", station_id) for station_id in stations)

    response = _request(IEM_DAILY_URL, params=params, session=session)
    try:
        data = pd.read_csv(StringIO(response.text), comment="#")
    except pd.errors.EmptyDataError:
        data = pd.DataFrame()

    columns = [
        "station_id",
        "station_name",
        "precip_mm",
        "trace",
        "source",
    ]
    if data.empty:
        empty = pd.DataFrame(columns=columns)
        empty.index = pd.DatetimeIndex([], name="date")
        return empty

    data.columns = [str(column).strip().lower() for column in data.columns]
    date_column = next(
        (column for column in ("day", "date", "valid") if column in data),
        None,
    )
    if date_column is None:
        raise PrecipitationDataError(
            "The IEM daily response did not contain a recognized date column."
        )

    data["date"] = pd.to_datetime(data[date_column], errors="coerce").dt.tz_localize(
        None
    )
    if "station" in data:
        data["station_id"] = data["station"].map(
            _normalize_cocorahs_station_id
        ).astype("string")
    else:
        data["station_id"] = pd.Series(pd.NA, index=data.index, dtype="string")
    if "station_name" in data:
        data["station_name"] = data["station_name"].astype("string")
    elif "name" in data:
        data["station_name"] = data["name"].astype("string")
    else:
        data["station_name"] = pd.NA

    raw_precip = data.get(
        "precip_in",
        pd.Series(pd.NA, index=data.index, dtype="string"),
    ).astype("string").str.strip()
    numeric_precip = pd.to_numeric(raw_precip, errors="coerce")
    data["trace"] = raw_precip.str.upper().eq("T") | _is_trace_value(numeric_precip)
    data["precip_mm"] = numeric_precip.mask(data["trace"], 0.0) * INCH_TO_MM
    data["source"] = "Iowa Environmental Mesonet CoCoRaHS"

    data = data.dropna(subset=["date"])
    data = data.drop_duplicates(["date", "station_id"], keep="last")
    data = data.set_index("date").sort_index()
    return data[columns]


def _ogc_features(
    collection: str,
    *,
    params: Mapping[str, Any],
    session: requests.Session | None = None,
    max_pages: int = 200,
) -> list[dict[str, Any]]:
    """Retrieve paginated GeoJSON features from a USGS OGC collection."""
    url = f"{USGS_OGC_BASE_URL}/collections/{collection}/items"
    query = dict(params)
    query.setdefault("f", "json")
    query.setdefault("limit", 50000)
    features: list[dict[str, Any]] = []

    for _ in range(max_pages):
        response = _request(url, params=query, session=session)
        payload = response.json()
        page_features = payload.get("features", [])
        features.extend(page_features)
        next_link = next(
            (
                link.get("href")
                for link in payload.get("links", [])
                if link.get("rel") == "next"
            ),
            None,
        )
        if not next_link or not page_features:
            return features
        url = next_link
        query = {}

    raise PrecipitationDataError(
        f"USGS pagination exceeded {max_pages} pages for {collection}."
    )


def _feature_rows(features: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Flatten GeoJSON feature properties and point geometry."""
    rows: list[dict[str, Any]] = []
    for feature in features:
        row = dict(feature.get("properties") or {})
        coordinates = (feature.get("geometry") or {}).get("coordinates", [])
        if len(coordinates) >= 2:
            row["longitude"] = coordinates[0]
            row["latitude"] = coordinates[1]
        row["feature_id"] = feature.get("id")
        rows.append(row)
    return pd.DataFrame(rows)


def _as_utc_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    """Return a timezone-aware UTC timestamp."""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _usgs_monitoring_location_metadata(
    monitoring_location_ids: Iterable[str],
    *,
    session: requests.Session | None = None,
    chunk_size: int = 100,
) -> pd.DataFrame:
    """Retrieve names and site types for USGS monitoring locations."""
    station_ids = _as_list(monitoring_location_ids)
    frames: list[pd.DataFrame] = []

    for start in range(0, len(station_ids), chunk_size):
        chunk = station_ids[start : start + chunk_size]
        features = _ogc_features(
            "monitoring-locations",
            params={
                "id": ",".join(chunk),
                "properties": "monitoring_location_name,site_type",
                "skipGeometry": "true",
            },
            session=session,
        )
        table = _feature_rows(features)
        if table.empty:
            continue
        table = table.rename(
            columns={"feature_id": "monitoring_location_id"}
        )
        frames.append(table)

    columns = [
        "monitoring_location_id",
        "monitoring_location_name",
        "site_type",
    ]
    if not frames:
        return pd.DataFrame(columns=columns)

    data = pd.concat(frames, ignore_index=True)
    return (
        data.reindex(columns=columns)
        .drop_duplicates("monitoring_location_id")
        .reset_index(drop=True)
    )


def find_usgs_precipitation_sites(
    latitude: float,
    longitude: float,
    *,
    radius_km: float = 75,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    n_locations: int | None = 20,
    primary_only: bool = True,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Find nearby USGS continuous precipitation time series.

    The search uses the ``time-series-metadata`` collection rather than the
    combined metadata collection. USGS precipitation parameter ``00045`` is
    commonly published as a ``Decumulated`` continuous series, so the search
    does not require an ``Instantaneous`` computation identifier.

    Date filtering is performed against each time series' ``begin_utc`` and
    ``end_utc`` period of record. The nearest ``n_locations`` monitoring
    locations are retained, but all qualifying primary precipitation time
    series at those locations are returned. Consequently, one monitoring
    location can contribute more than one row when it has multiple physical
    gauges or sublocations.
    """
    if n_locations is not None and n_locations <= 0:
        raise ValueError("n_locations must be positive or None.")

    west, south, east, north = bounding_box(latitude, longitude, radius_km)
    features = _ogc_features(
        "time-series-metadata",
        params={
            "bbox": f"{west},{south},{east},{north}",
            "parameter_code": USGS_PRECIPITATION_PARAMETER,
        },
        session=session,
    )
    data = _feature_rows(features)

    columns = [
        "time_series_id",
        "monitoring_location_id",
        "monitoring_location_name",
        "site_type",
        "latitude",
        "longitude",
        "distance_km",
        "parameter_code",
        "unit_of_measure",
        "computation_identifier",
        "computation_period_identifier",
        "statistic_id",
        "primary",
        "sublocation_identifier",
        "web_description",
        "begin_utc",
        "end_utc",
        "data_gap_interval",
    ]
    if data.empty:
        return pd.DataFrame(columns=columns)

    data = data.rename(columns={"feature_id": "time_series_id"})
    for column in columns:
        if column not in data:
            data[column] = pd.NA

    for column in (
        "statistic_id",
        "parent_time_series_id",
        "primary",
        "sublocation_identifier",
        "web_description",
        "computation_period_identifier",
    ):
        if column in data:
            data[column] = data[column].replace({"": pd.NA})

    # Daily-value time series have a statistic, a parent time series, or a
    # daily computation period. They do not belong in the continuous workflow.
    statistic = data["statistic_id"]
    parent = data.get(
        "parent_time_series_id",
        pd.Series(pd.NA, index=data.index),
    )
    computation_period = (
        data["computation_period_identifier"]
        .astype("string")
        .str.casefold()
    )
    is_daily = (
        statistic.notna()
        | parent.notna()
        | computation_period.eq("daily").fillna(False)
    )
    data = data.loc[~is_daily].copy()

    if primary_only:
        primary = data["primary"].astype("string").str.casefold()
        data = data.loc[primary.eq("primary")].copy()

    for column in ("latitude", "longitude"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["begin_utc"] = pd.to_datetime(
        data["begin_utc"], errors="coerce", utc=True
    )
    data["end_utc"] = pd.to_datetime(
        data["end_utc"], errors="coerce", utc=True
    )
    data = data.dropna(
        subset=[
            "time_series_id",
            "monitoring_location_id",
            "latitude",
            "longitude",
        ]
    )

    if start is not None:
        start_time = _as_utc_timestamp(start)
        data = data.loc[data["end_utc"].ge(start_time)].copy()
    if end is not None:
        end_time = _as_utc_timestamp(end)
        data = data.loc[data["begin_utc"].le(end_time)].copy()

    if data.empty:
        return pd.DataFrame(columns=columns)

    data["distance_km"] = data.apply(
        lambda row: haversine_distance_km(
            latitude,
            longitude,
            row["latitude"],
            row["longitude"],
        ),
        axis=1,
    )
    data = data.loc[data["distance_km"] <= radius_km].copy()
    if data.empty:
        return pd.DataFrame(columns=columns)

    location_metadata = _usgs_monitoring_location_metadata(
        data["monitoring_location_id"].drop_duplicates(),
        session=session,
    )
    data = data.drop(
        columns=["monitoring_location_name", "site_type"],
        errors="ignore",
    ).merge(
        location_metadata,
        on="monitoring_location_id",
        how="left",
    )

    location_order = (
        data.groupby("monitoring_location_id")["distance_km"]
        .min()
        .sort_values()
        .index
    )
    if n_locations is not None:
        location_order = location_order[:n_locations]
    data = data.loc[
        data["monitoring_location_id"].isin(location_order)
    ].copy()

    data["_location_order"] = pd.Categorical(
        data["monitoring_location_id"],
        categories=list(location_order),
        ordered=True,
    )
    data = data.sort_values(
        [
            "_location_order",
            "sublocation_identifier",
            "time_series_id",
        ],
        na_position="first",
    ).drop(columns="_location_order")
    return data.reindex(columns=columns).reset_index(drop=True)


def get_usgs_precipitation(
    time_series_ids: str | Iterable[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    *,
    session: requests.Session | None = None,
    chunk_size: int = 10,
) -> pd.DataFrame:
    """Retrieve USGS continuous precipitation by time-series identifier.

    Querying by ``time_series_id`` prevents multiple gauges at the same
    monitoring location from being mixed together. Values reported in inches
    or millimeters are converted to ``precip_mm`` while the native value,
    units, qualifier, approval status, location, and time-series identifiers
    are preserved.
    """
    series_ids = _as_list(time_series_ids)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    start_time = _as_utc_timestamp(start)
    end_time = _as_utc_timestamp(end)
    if end_time < start_time:
        raise ValueError("end must be on or after start.")

    frames: list[pd.DataFrame] = []
    time_filter = (
        f"{start_time.isoformat().replace('+00:00', 'Z')}/"
        f"{end_time.isoformat().replace('+00:00', 'Z')}"
    )

    for chunk_start in range(0, len(series_ids), chunk_size):
        chunk = series_ids[chunk_start : chunk_start + chunk_size]
        features = _ogc_features(
            "continuous",
            params={
                "time_series_id": ",".join(chunk),
                "parameter_code": USGS_PRECIPITATION_PARAMETER,
                "time": time_filter,
                "skipGeometry": "true",
            },
            session=session,
        )
        table = _feature_rows(features)
        if not table.empty:
            frames.append(table)

    columns = [
        "monitoring_location_id",
        "precip_mm",
        "value",
        "unit_of_measure",
        "qualifier",
        "approval_status",
        "time_series_id",
        "parameter_code",
        "source",
    ]
    if not frames:
        empty = pd.DataFrame(columns=columns)
        empty.index = pd.DatetimeIndex([], tz="UTC", name="time_utc")
        return empty

    data = pd.concat(frames, ignore_index=True)
    data["time_utc"] = pd.to_datetime(
        data.get("time"), errors="coerce", utc=True
    )
    data["value"] = pd.to_numeric(data.get("value"), errors="coerce")

    units = data.get(
        "unit_of_measure",
        pd.Series(pd.NA, index=data.index),
    ).astype("string")
    unit_key = units.str.lower().str.replace(".", "", regex=False).str.strip()
    inch_units = unit_key.isin({"in", "inch", "inches"})
    millimeter_units = unit_key.isin(
        {"mm", "millimeter", "millimeters"}
    )
    data["precip_mm"] = pd.NA
    data.loc[inch_units, "precip_mm"] = (
        data.loc[inch_units, "value"] * INCH_TO_MM
    )
    data.loc[millimeter_units, "precip_mm"] = data.loc[
        millimeter_units, "value"
    ]
    data["precip_mm"] = pd.to_numeric(
        data["precip_mm"], errors="coerce"
    )

    # Normalize field-name variants encountered in OGC responses.
    if "approvals_status" in data:
        data["approval_status"] = data["approvals_status"]
    elif "approval_status" not in data:
        data["approval_status"] = pd.NA

    if "timeseries_id" in data:
        data["time_series_id"] = data["timeseries_id"]
    elif "time_series_id" not in data:
        data["time_series_id"] = pd.NA

    data["source"] = "USGS Water Data continuous values"
    data = data.dropna(subset=["time_utc", "time_series_id"])

    duplicate_count = data.duplicated(
        ["time_series_id", "time_utc"]
    ).sum()
    if duplicate_count:
        raise PrecipitationDataError(
            "USGS continuous data contain duplicate timestamps within a "
            f"time series ({duplicate_count} duplicate rows)."
        )

    data = data.sort_values(["time_utc", "time_series_id"])
    data = data.set_index("time_utc")
    return data.reindex(columns=columns)


def aggregate_interval_precipitation_daily(
    data: pd.DataFrame,
    *,
    timezone: str,
    precipitation_column: str = "precip_mm",
    station_column: str = "station_id",
    trace_column: str | None = "trace",
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    infer_expected_interval: bool = False,
    minimum_coverage: float | None = None,
) -> pd.DataFrame:
    """Aggregate nonoverlapping interval precipitation to local days.

    Parameters
    ----------
    data
        Timezone-aware interval observations indexed by timestamp.
    timezone
        IANA timezone used to assign observations to local calendar dates.
    precipitation_column, station_column, trace_column
        Input column names.
    start_date, end_date
        Optional inclusive local-date bounds. When supplied, completely
        missing days are retained in the result.
    infer_expected_interval
        Infer each station's nominal interval from the median spacing of its
        observations and calculate an expected daily observation count.
    minimum_coverage
        Optional fraction from 0 to 1. Daily totals below this interval
        coverage are masked as missing while ``observed_precip_mm`` preserves
        the partial sum.

    Notes
    -----
    This function is intended for interval totals such as IEM computed hourly
    ASOS precipitation or USGS ``Decumulated`` precipitation. It must not be
    applied to cumulative counters or repeated overlapping METAR ``p01i``
    reports.
    """
    if precipitation_column not in data:
        raise KeyError(f"Missing precipitation column: {precipitation_column}")
    if station_column not in data:
        raise KeyError(f"Missing station column: {station_column}")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("data must use a DatetimeIndex.")
    if data.index.tz is None:
        raise ValueError("data index must be timezone-aware.")
    if minimum_coverage is not None and not 0 <= minimum_coverage <= 1:
        raise ValueError("minimum_coverage must be between 0 and 1.")

    reset = data.reset_index()
    time_column = reset.columns[0]
    duplicate_count = reset.duplicated([station_column, time_column]).sum()
    if duplicate_count:
        raise PrecipitationDataError(
            "Interval data contain duplicate station timestamps "
            f"({duplicate_count} duplicate rows)."
        )

    local = data.copy()
    local.index = local.index.tz_convert(timezone)
    local["date"] = local.index.tz_localize(None).normalize()

    grouped = local.groupby(["date", station_column], dropna=False)
    daily = grouped[precipitation_column].sum(min_count=1).rename(
        "precip_mm"
    )
    result = daily.to_frame()
    result["observation_count"] = grouped[precipitation_column].count()
    if trace_column and trace_column in local:
        result["trace"] = grouped[trace_column].any()

    stations = pd.Index(local[station_column].dropna().unique())
    if start_date is not None or end_date is not None:
        first_date = (
            pd.Timestamp(start_date).normalize().tz_localize(None)
            if start_date is not None
            else local["date"].min()
        )
        last_date = (
            pd.Timestamp(end_date).normalize().tz_localize(None)
            if end_date is not None
            else local["date"].max()
        )
        if last_date < first_date:
            raise ValueError("end_date must be on or after start_date.")
        all_dates = pd.date_range(first_date, last_date, freq="D")
        full_index = pd.MultiIndex.from_product(
            [all_dates, stations],
            names=["date", station_column],
        )
        result = result.reindex(full_index)
        result["observation_count"] = (
            result["observation_count"].fillna(0).astype(int)
        )
        if "trace" in result:
            result["trace"] = result["trace"].fillna(False).astype(bool)

    if infer_expected_interval:
        interval_by_station: dict[Any, pd.Timedelta] = {}
        for station_id, station_data in local.groupby(station_column):
            unique_times = pd.DatetimeIndex(
                station_data.index.unique()
            ).sort_values()
            differences = unique_times.to_series().diff().dropna()
            differences = differences.loc[differences > pd.Timedelta(0)]
            if differences.empty:
                interval_by_station[station_id] = pd.NaT
            else:
                interval_by_station[station_id] = differences.median()

        expected_counts = []
        inferred_intervals = []
        for date, station_id in result.index:
            interval = interval_by_station.get(station_id, pd.NaT)
            inferred_intervals.append(interval)
            if pd.isna(interval) or interval <= pd.Timedelta(0):
                expected_counts.append(pd.NA)
                continue

            local_start = pd.Timestamp(date).tz_localize(timezone)
            local_end = (pd.Timestamp(date) + pd.Timedelta(days=1)).tz_localize(
                timezone
            )
            duration = (
                local_end.tz_convert("UTC")
                - local_start.tz_convert("UTC")
            )
            expected_counts.append(int(round(duration / interval)))

        result["inferred_interval"] = inferred_intervals
        result["expected_observation_count"] = pd.array(
            expected_counts,
            dtype="Int64",
        )
        result["coverage_fraction"] = (
            result["observation_count"]
            / result["expected_observation_count"]
        )
        result["coverage_fraction"] = result[
            "coverage_fraction"
        ].clip(upper=1.0)

        if minimum_coverage is not None:
            result["complete_day"] = (
                result["coverage_fraction"] >= minimum_coverage
            ).fillna(False)
            result["observed_precip_mm"] = result["precip_mm"]
            result["precip_mm"] = result["precip_mm"].where(
                result["complete_day"]
            )

    return result.reset_index().set_index("date").sort_index()


def summarize_daily_precipitation_coverage(
    data: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    *,
    station_column: str = "station_id",
    precipitation_column: str = "precip_mm",
) -> pd.DataFrame:
    """Summarize daily reporting coverage for each station or time series."""
    if station_column not in data:
        raise KeyError(f"Missing station column: {station_column}")
    if precipitation_column not in data:
        raise KeyError(f"Missing precipitation column: {precipitation_column}")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("data must use a DatetimeIndex.")

    start_date = pd.Timestamp(start).normalize().tz_localize(None)
    end_date = pd.Timestamp(end).normalize().tz_localize(None)
    if end_date < start_date:
        raise ValueError("end must be on or after start.")
    expected_days = len(pd.date_range(start_date, end_date, freq="D"))

    rows: list[dict[str, Any]] = []
    for station_id, group in data.groupby(station_column):
        dates = pd.DatetimeIndex(group.index).tz_localize(None).normalize()
        valid = group[precipitation_column].notna().to_numpy()
        reported_days = dates[valid].nunique()
        rows.append(
            {
                station_column: station_id,
                "expected_days": expected_days,
                "reported_days": int(reported_days),
                "missing_days": int(expected_days - reported_days),
                "coverage_fraction": reported_days / expected_days,
                "total_precip_mm": group[precipitation_column].sum(
                    min_count=1
                ),
            }
        )

    return pd.DataFrame(rows)


__all__ = [
    "PrecipitationDataError",
    "aggregate_interval_precipitation_daily",
    "bounding_box",
    "find_asos_stations",
    "find_cocorahs_stations",
    "find_ghcnd_stations",
    "find_usgs_precipitation_sites",
    "get_asos_hourly_precipitation",
    "get_asos_precipitation",
    "get_cocorahs_precipitation",
    "get_ghcnd_precipitation",
    "get_iem_network_stations",
    "get_usgs_precipitation",
    "haversine_distance_km",
    "summarize_daily_precipitation_coverage",
]
