"""Utilities for downloading and archiving Ambient Weather station data.

The module contains no station-specific metadata or credentials. A notebook or
script supplies an API key, application key, device MAC address, station time
zone, and archive filename prefix.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
import requests

API_BASE_URL = "https://api.ambientweather.net/v1"
MAX_BATCH_LIMIT = 288


class AmbientWeatherClient:
    """Small client for the documented Ambient Weather REST API."""

    def __init__(
        self,
        api_key: str,
        application_key: str,
        *,
        base_url: str = API_BASE_URL,
        request_interval_seconds: float = 1.1,
        request_timeout_seconds: float = 30.0,
        max_retries: int = 3,
        user_agent: str = "ambient-weather-python/1.0",
    ) -> None:
        if not api_key or not application_key:
            raise ValueError("Both Ambient Weather keys are required.")
        if request_interval_seconds < 1.0:
            raise ValueError(
                "request_interval_seconds must be at least 1.0 to respect "
                "Ambient Weather's per-user rate limit."
            )
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative.")

        self.api_key = api_key
        self.application_key = application_key
        self.base_url = base_url.rstrip("/")
        self.request_interval_seconds = request_interval_seconds
        self.request_timeout_seconds = request_timeout_seconds
        self.max_retries = max_retries

        self._session = requests.Session()
        self._session.headers.update({"User-Agent": user_agent})
        self._last_request_time = 0.0

    def _respect_rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        wait = self.request_interval_seconds - elapsed
        if wait > 0:
            time.sleep(wait)

    def _get(self, endpoint: str, **parameters: Any) -> Any:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        query = {
            "apiKey": self.api_key,
            "applicationKey": self.application_key,
            **parameters,
        }

        for attempt in range(self.max_retries + 1):
            self._respect_rate_limit()

            try:
                response = self._session.get(
                    url,
                    params=query,
                    timeout=self.request_timeout_seconds,
                )
            except requests.RequestException:
                self._last_request_time = time.monotonic()
                if attempt == self.max_retries:
                    raise RuntimeError(
                        "The Ambient Weather API request failed before a "
                        "response was received. Check the network connection."
                    ) from None
                time.sleep(2**attempt)
                continue

            self._last_request_time = time.monotonic()

            if response.status_code == 429:
                if attempt == self.max_retries:
                    raise RuntimeError(
                        "Ambient Weather rate-limited the request after "
                        "repeated retries."
                    )
                time.sleep(max(2**attempt, self.request_interval_seconds))
                continue

            if not response.ok:
                detail = response.text.strip()[:300]
                raise RuntimeError(
                    f"Ambient Weather returned HTTP {response.status_code}. "
                    f"Response: {detail or 'No response body.'}"
                )

            try:
                return response.json()
            except ValueError:
                raise RuntimeError(
                    "Ambient Weather returned a response that was not valid "
                    "JSON."
                ) from None

        raise RuntimeError("Ambient Weather request failed unexpectedly.")

    def list_devices(self) -> list[dict[str, Any]]:
        """Return devices associated with the API key's user account."""
        devices = self._get("devices")
        if not isinstance(devices, list):
            raise RuntimeError("The device-list response was not a JSON list.")
        return devices

    def get_device_data(
        self,
        mac_address: str,
        *,
        end_date: Any | None = None,
        limit: int = MAX_BATCH_LIMIT,
    ) -> list[dict[str, Any]]:
        """Download one descending batch of observations for a device."""
        if not mac_address:
            raise ValueError("mac_address is required.")
        if not 1 <= limit <= MAX_BATCH_LIMIT:
            raise ValueError(
                f"limit must be between 1 and {MAX_BATCH_LIMIT}."
            )

        parameters: dict[str, Any] = {"limit": limit}
        if end_date is not None:
            timestamp = as_utc_timestamp(end_date)
            parameters["endDate"] = int(timestamp.timestamp() * 1000)

        records = self._get(
            f"devices/{mac_address.upper()}",
            **parameters,
        )
        if not isinstance(records, list):
            raise RuntimeError(
                "The historical-data response was not a JSON list."
            )
        return records


def as_utc_timestamp(
    value: Any,
    *,
    assume_timezone: str = "UTC",
) -> pd.Timestamp:
    """Convert a date-like value to a timezone-aware UTC timestamp."""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(assume_timezone)
    return timestamp.tz_convert("UTC")


def records_to_dataframe(
    records: list[dict[str, Any]],
) -> pd.DataFrame:
    """Convert API records to a sorted DataFrame indexed by UTC time."""
    if not records:
        return _empty_frame()

    frame = pd.DataFrame.from_records(records)
    if "dateutc" not in frame.columns:
        raise KeyError(
            "The API response does not contain the required dateutc field."
        )

    frame["time_utc"] = pd.to_datetime(
        frame["dateutc"],
        unit="ms",
        utc=True,
        errors="raise",
    )
    frame = frame.sort_values("time_utc")
    frame = frame.drop_duplicates(subset="time_utc", keep="last")
    return frame.set_index("time_utc")


def fetch_recent_data(
    client: AmbientWeatherClient,
    mac_address: str,
    *,
    hours: float = 24.0,
    now: Any | None = None,
    max_batches: int | None = None,
) -> pd.DataFrame:
    """Fetch recent observations, paging backward until ``hours`` is covered.

    This is convenient for notebooks that need recent station data but do not
    need to maintain a local archive. The returned index is UTC.
    """
    if hours <= 0:
        raise ValueError("hours must be positive.")

    end_time = as_utc_timestamp(now) if now is not None else pd.Timestamp.now(
        tz="UTC"
    )
    start_time = end_time - pd.Timedelta(hours=hours)

    cursor: pd.Timestamp | None = end_time
    batches: list[pd.DataFrame] = []
    batch_count = 0

    while max_batches is None or batch_count < max_batches:
        batch = records_to_dataframe(
            client.get_device_data(
                mac_address,
                end_date=cursor,
                limit=MAX_BATCH_LIMIT,
            )
        )
        batch_count += 1

        if batch.empty:
            break

        batches.append(batch)
        oldest_time = batch.index.min()
        if oldest_time <= start_time:
            break

        next_cursor = oldest_time - pd.Timedelta(milliseconds=1)
        if cursor is not None and next_cursor >= cursor:
            raise RuntimeError("API pagination did not move backward in time.")
        cursor = next_cursor

    combined = merge_frames(*batches)
    if combined.empty:
        return combined
    return combined.loc[(combined.index >= start_time) & (combined.index <= end_time)]


def merge_frames(*frames: pd.DataFrame) -> pd.DataFrame:
    """Combine observations, sort by time, and remove duplicate timestamps."""
    nonempty = [frame for frame in frames if frame is not None and not frame.empty]
    if not nonempty:
        return _empty_frame()

    combined = pd.concat(nonempty, axis=0, sort=False)
    combined = combined.sort_index()
    return combined[~combined.index.duplicated(keep="last")]


def annual_archive_path(
    data_dir: Path | str,
    prefix: str,
    year: int,
) -> Path:
    """Return the path for one local-calendar-year CSV archive."""
    return Path(data_dir) / f"{prefix}_{int(year):04d}.csv"


def list_archive_years(
    data_dir: Path | str,
    prefix: str,
) -> list[int]:
    """Return sorted years represented by matching annual CSV files."""
    directory = Path(data_dir)
    years: list[int] = []
    for path in directory.glob(f"{prefix}_????.csv"):
        suffix = path.stem.removeprefix(f"{prefix}_")
        if len(suffix) == 4 and suffix.isdigit():
            years.append(int(suffix))
    return sorted(set(years))


def read_csv_archive(path: Path | str) -> pd.DataFrame:
    """Read one archive CSV with a UTC DatetimeIndex."""
    path = Path(path)
    if not path.exists():
        return _empty_frame()

    frame = pd.read_csv(path)
    if "time_utc" not in frame.columns:
        raise ValueError(f"{path} does not contain a time_utc column.")

    frame["time_utc"] = pd.to_datetime(
        frame["time_utc"],
        utc=True,
        errors="raise",
    )
    frame = frame.set_index("time_utc").sort_index()
    return frame[~frame.index.duplicated(keep="last")]


def read_annual_archive(
    data_dir: Path | str,
    prefix: str,
    *,
    years: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Read and combine selected annual CSV files."""
    selected_years = (
        list_archive_years(data_dir, prefix)
        if years is None
        else sorted(set(int(year) for year in years))
    )
    frames = [
        read_csv_archive(annual_archive_path(data_dir, prefix, year))
        for year in selected_years
    ]
    return merge_frames(*frames)


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = frame.sort_index()
    clean = clean[~clean.index.duplicated(keep="last")]

    temporary_path = path.with_suffix(path.suffix + ".tmp")
    clean.to_csv(temporary_path, index_label="time_utc")
    temporary_path.replace(path)


def write_annual_archive(
    frame: pd.DataFrame,
    data_dir: Path | str,
    prefix: str,
    station_timezone: str,
    *,
    drop_columns: Sequence[str] = (),
) -> list[Path]:
    """Merge observations into CSV files partitioned by local calendar year.

    File years are based on the station's local time, while ``time_utc`` remains
    the stored index. This avoids assigning late-evening December observations
    to the following year merely because UTC has crossed midnight.
    """
    if frame.empty:
        return []
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("frame must have a DatetimeIndex.")

    clean = frame.copy()
    if clean.index.tz is None:
        clean.index = clean.index.tz_localize("UTC")
    else:
        clean.index = clean.index.tz_convert("UTC")

    existing_drop_columns = [
        column for column in drop_columns if column in clean.columns
    ]
    if existing_drop_columns:
        clean = clean.drop(columns=existing_drop_columns)

    local_years = clean.index.tz_convert(station_timezone).year
    written: list[Path] = []

    for year in sorted(set(int(year) for year in local_years)):
        selected = clean.loc[local_years == year]
        path = annual_archive_path(data_dir, prefix, year)
        existing = read_csv_archive(path)
        merged = merge_frames(existing, selected)
        _write_csv_atomic(merged, path)
        written.append(path)

    return written


def archive_bounds(
    data_dir: Path | str,
    prefix: str,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None, int]:
    """Return earliest time, latest time, and row count across annual files."""
    earliest: pd.Timestamp | None = None
    latest: pd.Timestamp | None = None
    total_rows = 0

    for year in list_archive_years(data_dir, prefix):
        path = annual_archive_path(data_dir, prefix, year)
        times = pd.read_csv(path, usecols=["time_utc"])
        if times.empty:
            continue
        parsed = pd.to_datetime(times["time_utc"], utc=True, errors="raise")
        year_min = parsed.min()
        year_max = parsed.max()
        earliest = year_min if earliest is None else min(earliest, year_min)
        latest = year_max if latest is None else max(latest, year_max)
        total_rows += len(parsed)

    return earliest, latest, total_rows


def annual_archive_summary(
    data_dir: Path | str,
    prefix: str,
    station_timezone: str,
) -> pd.Series:
    """Return a compact summary of annual archive coverage."""
    years = list_archive_years(data_dir, prefix)
    earliest, latest, observations = archive_bounds(data_dir, prefix)

    return pd.Series(
        {
            "observations": observations,
            "first_time_utc": earliest,
            "last_time_utc": latest,
            "first_time_local": (
                earliest.tz_convert(station_timezone)
                if earliest is not None
                else pd.NaT
            ),
            "last_time_local": (
                latest.tz_convert(station_timezone)
                if latest is not None
                else pd.NaT
            ),
            "archive_years": ", ".join(str(year) for year in years),
            "number_of_files": len(years),
        },
        name="archive",
    )


def update_annual_archive(
    client: AmbientWeatherClient,
    mac_address: str,
    data_dir: Path | str,
    prefix: str,
    station_timezone: str,
    *,
    drop_columns: Sequence[str] = (),
    max_batches: int | None = None,
) -> int:
    """Add all observations newer than the latest archived timestamp."""
    _, cutoff, _ = archive_bounds(data_dir, prefix)
    cursor: pd.Timestamp | None = None
    downloaded: list[pd.DataFrame] = []
    batch_count = 0

    while max_batches is None or batch_count < max_batches:
        batch = records_to_dataframe(
            client.get_device_data(mac_address, end_date=cursor)
        )
        batch_count += 1

        if batch.empty:
            break

        if cutoff is None:
            downloaded.append(batch)
            break

        newer = batch.loc[batch.index > cutoff]
        if not newer.empty:
            downloaded.append(newer)

        oldest_time = batch.index.min()
        if oldest_time <= cutoff:
            break

        next_cursor = oldest_time - pd.Timedelta(milliseconds=1)
        if cursor is not None and next_cursor >= cursor:
            raise RuntimeError("API pagination did not move backward in time.")
        cursor = next_cursor

    new_data = merge_frames(*downloaded)
    write_annual_archive(
        new_data,
        data_dir,
        prefix,
        station_timezone,
        drop_columns=drop_columns,
    )
    return len(new_data)


def backfill_annual_archive(
    client: AmbientWeatherClient,
    mac_address: str,
    data_dir: Path | str,
    prefix: str,
    station_timezone: str,
    *,
    start_date: Any | None = None,
    drop_columns: Sequence[str] = (),
    max_batches: int | None = 5,
    checkpoint_every: int = 10,
) -> int:
    """Download observations older than the earliest annual archive record."""
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be at least 1.")

    earliest, _, _ = archive_bounds(data_dir, prefix)
    start_timestamp = (
        as_utc_timestamp(start_date, assume_timezone=station_timezone)
        if start_date is not None
        else None
    )
    cursor = (
        earliest - pd.Timedelta(milliseconds=1)
        if earliest is not None
        else None
    )

    buffer: list[pd.DataFrame] = []
    batch_count = 0
    added = 0

    while max_batches is None or batch_count < max_batches:
        batch = records_to_dataframe(
            client.get_device_data(mac_address, end_date=cursor)
        )
        batch_count += 1

        if batch.empty:
            break

        oldest_time = batch.index.min()
        selected = batch
        reached_start = False

        if start_timestamp is not None:
            selected = batch.loc[batch.index >= start_timestamp]
            reached_start = oldest_time <= start_timestamp

        if not selected.empty:
            buffer.append(selected)
            added += len(selected)

        should_checkpoint = (
            batch_count % checkpoint_every == 0
            or reached_start
            or (max_batches is not None and batch_count == max_batches)
        )
        if should_checkpoint and buffer:
            write_annual_archive(
                merge_frames(*buffer),
                data_dir,
                prefix,
                station_timezone,
                drop_columns=drop_columns,
            )
            buffer = []

        if reached_start:
            break

        next_cursor = oldest_time - pd.Timedelta(milliseconds=1)
        if cursor is not None and next_cursor >= cursor:
            raise RuntimeError("API pagination did not move backward in time.")
        cursor = next_cursor

    if buffer:
        write_annual_archive(
            merge_frames(*buffer),
            data_dir,
            prefix,
            station_timezone,
            drop_columns=drop_columns,
        )

    return added


def migrate_single_csv_to_annual(
    source_path: Path | str,
    data_dir: Path | str,
    prefix: str,
    station_timezone: str,
    *,
    drop_columns: Sequence[str] = (),
) -> list[Path]:
    """Split a legacy single-file archive into local-calendar-year files."""
    source = read_csv_archive(source_path)
    return write_annual_archive(
        source,
        data_dir,
        prefix,
        station_timezone,
        drop_columns=drop_columns,
    )


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(
        index=pd.DatetimeIndex([], name="time_utc", tz="UTC")
    )
