#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-shot 32-beam trace-mode A/B/omega parameter generator.

This program computes one set of trace-mode A/B/omega parameters at a user-
provided UTC reference time ``t0`` and a user-provided center RA/Dec. It
writes exactly two output files:

- ``trace_mode_phase_coeff.txt``
- ``trace_mode_phase_coeff.md5``
"""

from __future__ import division, print_function

import argparse
import hashlib
import io
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import ephem
import numpy as np

from beam32_from_azel_functions import (
    build_epoch_time_context,
    build_katpoint_antennas_from_records,
    build_katpoint_target_from_radec,
    compute_source_state,
    deg_to_dms_str,
    ensure_utc_datetime,
    format_dual_timestamp,
    parse_dec_to_rad,
    parse_ra_to_rad,
    signed_angle_delta_rad,
    trace_build_source_aligned_horizontal_frame as build_source_aligned_horizontal_frame,
    trace_collect_antenna_beam_results as collect_antenna_beam_results,
    trace_compute_local_phase_angle_rad as compute_local_phase_angle_rad,
    wrap_deg_360,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# =========================
# User-editable configuration
# =========================
ANTS_TXT = None
BEAM_OFFSET_TABLE_FILE = None
MIN_ELEVATION_DEG = 32.0
SIMULATION_IGNORE_VISIBILITY = True
CENTER_FREQ_HZ = 1.25e9
OMEGA_DELTA_SECONDS = 1.0
PRINT_PIPELINE_STEPS = True


# =========================
# Fixed hardware contract
# =========================
SIGNALS_PER_ANTENNA = 2
TOTAL_SIGNAL_INPUTS = 20
TOTAL_BEAMS = 32
TRACE_STREAM_FLOAT_COUNT = 3 * TOTAL_SIGNAL_INPUTS * TOTAL_BEAMS
LIGHT_SPEED_M_PER_S = 299792458.0
WGS84_A_M = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


# =========================
# Fixed output paths
# =========================
OUTPUT_DIR = SCRIPT_DIR
TRACE_MODE_PHASE_TXT = os.path.join(OUTPUT_DIR, "trace_mode_phase_coeff.txt")
TRACE_MODE_PHASE_MD5 = os.path.splitext(TRACE_MODE_PHASE_TXT)[0] + ".md5"


@dataclass
class AntennaRecord:
    """Parsed antenna row plus derived coordinates."""

    name: str
    lat_deg: float
    lon_deg: float
    height_m: float
    diameter_m: float
    lat_rad: float
    lon_rad: float
    ecef_m: np.ndarray
    enu_m: np.ndarray


@dataclass
class VisibilityWindow:
    """Visibility state for one epoch at the reference antenna."""

    epoch_time_utc: datetime
    min_elevation_deg: float
    visible_now: bool
    rise_utc: Optional[datetime]
    set_utc: Optional[datetime]
    transit_utc: Optional[datetime]
    next_rise_utc: Optional[datetime]
    circumpolar: bool
    never_up: bool
    window_contains_epoch: bool


def get_active_signal_inputs(antenna_count):
    """Return the active hardware input count for the current antenna count."""

    active_signal_inputs = int(antenna_count) * SIGNALS_PER_ANTENNA
    if active_signal_inputs > TOTAL_SIGNAL_INPUTS:
        raise ValueError(
            "Antenna count {} requires {} signal inputs, which exceeds TOTAL_SIGNAL_INPUTS {}".format(
                antenna_count,
                active_signal_inputs,
                TOTAL_SIGNAL_INPUTS,
            )
        )
    return active_signal_inputs


def parse_args():
    """Parse command-line arguments for the one-shot trace-mode generator."""

    parser = argparse.ArgumentParser(
        description="Compute one-shot trace-mode A/B/omega parameters for 32 beams.",
    )
    parser.add_argument(
        "-d",
        "--direction",
        required=True,
        help='Center RA/Dec as "RA DEC", for example "19:35:00 21:54:00".',
    )
    parser.add_argument(
        "-t",
        "--time-utc",
        required=True,
        help='Reference UTC datetime t0 as "YYYY-MM-DD HH:MM", for example "2026-06-04 20:20". Seconds default to 00.',
    )
    parser.add_argument(
        "-f",
        "--ants-file",
        "--ants-txt",
        dest="ants_txt",
        required=True,
        help="Path to the antenna TXT file, for example ants.txt.",
    )
    parser.add_argument(
        "-b",
        "--beam-offsets-file",
        "--beam-offset-table",
        dest="beam_offset_table",
        required=True,
        help="Path to the 32-beam offset table TXT file.",
    )
    parser.add_argument(
        "--target-name",
        default="TARGET",
        help="Target name printed in terminal output.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        default=OUTPUT_DIR,
        help=(
            "Directory for generated trace_mode_phase_coeff.txt/.md5 files. "
            "Default: script directory."
        ),
    )
    visibility_group = parser.add_mutually_exclusive_group()
    visibility_group.add_argument(
        "--ignore-visibility",
        dest="ignore_visibility",
        action="store_true",
        help="Allow output even when the target is below the minimum elevation.",
    )
    visibility_group.add_argument(
        "--require-visibility",
        dest="ignore_visibility",
        action="store_false",
        help="Fail instead of outputting parameters when the target is below the minimum elevation.",
    )
    parser.set_defaults(ignore_visibility=SIMULATION_IGNORE_VISIBILITY)

    args = parser.parse_args()
    try:
        args.target_ra, args.target_dec = parse_direction_arg(args.direction)
        args.t0_utc = parse_time_utc_arg(args.time_utc)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def configure_ants_file(ants_txt):
    """Update the antenna TXT path from the command line."""

    global ANTS_TXT

    if not ants_txt:
        raise ValueError("ants_txt must not be empty")

    ANTS_TXT = os.path.abspath(os.path.expanduser(ants_txt))
    return ANTS_TXT


def configure_beam_offset_table_file(beam_offset_table):
    """Update the beam-offset TXT path from the command line."""

    global BEAM_OFFSET_TABLE_FILE

    if not beam_offset_table:
        raise ValueError("beam_offset_table must not be empty")

    BEAM_OFFSET_TABLE_FILE = os.path.abspath(os.path.expanduser(beam_offset_table))
    return BEAM_OFFSET_TABLE_FILE


def configure_output_dir(output_dir):
    """Update the generated trace output paths to use one requested directory."""

    global OUTPUT_DIR
    global TRACE_MODE_PHASE_TXT
    global TRACE_MODE_PHASE_MD5

    if output_dir is None:
        raise ValueError("output_dir must not be None")

    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    if os.path.exists(output_dir) and not os.path.isdir(output_dir):
        raise ValueError("Output path exists but is not a directory: {}".format(output_dir))

    os.makedirs(output_dir, exist_ok=True)

    OUTPUT_DIR = output_dir
    TRACE_MODE_PHASE_TXT = os.path.join(OUTPUT_DIR, "trace_mode_phase_coeff.txt")
    TRACE_MODE_PHASE_MD5 = os.path.splitext(TRACE_MODE_PHASE_TXT)[0] + ".md5"
    return OUTPUT_DIR


def parse_direction_arg(direction_text):
    """Parse one command-line direction string into RA and Dec fields."""

    parts = str(direction_text).strip().split()
    if len(parts) != 2:
        raise ValueError(
            'Invalid -d/--direction. Expected "RA DEC", for example "19:35:00 21:54:00".'
        )
    return parts[0], parts[1]


def parse_time_utc_arg(time_text):
    """Parse a minute-precision UTC datetime and normalize seconds to zero."""

    time_text = str(time_text).strip()
    try:
        return datetime.strptime(time_text, "%Y-%m-%d %H:%M").replace(second=0, tzinfo=timezone.utc)
    except ValueError:
        raise ValueError(
            'Invalid -t/--time-utc. Expected UTC datetime to minute precision '
            '"YYYY-MM-DD HH:MM", for example "2026-06-04 20:20". Seconds are '
            'set to 00 automatically.'
        )


def _parse_optional_int_token(token):
    token = str(token).strip()
    if token.lower() in ("n/a", "na", "none", "-"):
        return 0
    return int(token)


def build_ephem_observer(ref_ant, when_utc, horizon_deg):
    """Build a fresh ephem observer for one reference antenna and epoch."""

    observer = ephem.Observer()
    observer.lat = deg_to_dms_str(ref_ant.lat_deg)
    observer.lon = deg_to_dms_str(ref_ant.lon_deg)
    observer.elevation = float(ref_ant.height_m)
    observer.horizon = str(float(horizon_deg))
    observer.date = ephem.Date(ensure_utc_datetime(when_utc))
    return observer


def build_ephem_fixed_body(target_ra_text, target_dec_text):
    """Build one fixed RA/Dec source for visibility window calculations."""

    body = ephem.FixedBody()
    body._ra = str(target_ra_text).strip()
    body._dec = str(target_dec_text).strip()
    body._epoch = ephem.J2000
    return body


def to_utc_aware_datetime(when_dt):
    """Convert an ephem-returned datetime into aware UTC."""

    if when_dt is None:
        return None
    if when_dt.tzinfo is None:
        return when_dt.replace(tzinfo=timezone.utc)
    return when_dt.astimezone(timezone.utc)


def compute_visibility_window(ref_ant, epoch_time_utc, target_ra_text, target_dec_text, min_elevation_deg):
    """Describe whether the source is visible now and the relevant rise/set window."""

    observer = build_ephem_observer(ref_ant, epoch_time_utc, min_elevation_deg)
    body = build_ephem_fixed_body(target_ra_text, target_dec_text)
    body.compute(observer)

    alt_deg = np.rad2deg(float(body.alt))
    visible_now = bool(alt_deg >= float(min_elevation_deg))
    transit_utc = None
    try:
        transit_utc = to_utc_aware_datetime(observer.next_transit(body, start=observer.date).datetime())
    except Exception:
        transit_utc = None

    status = VisibilityWindow(
        epoch_time_utc=ensure_utc_datetime(epoch_time_utc),
        min_elevation_deg=float(min_elevation_deg),
        visible_now=visible_now,
        rise_utc=None,
        set_utc=None,
        transit_utc=transit_utc,
        next_rise_utc=None,
        circumpolar=False,
        never_up=False,
        window_contains_epoch=False,
    )

    try:
        if visible_now:
            rise = observer.previous_rising(body, start=observer.date)
            set_ = observer.next_setting(body, start=observer.date)
            status.window_contains_epoch = True
        else:
            rise = observer.next_rising(body, start=observer.date)
            set_ = observer.next_setting(body, start=rise)
            status.next_rise_utc = to_utc_aware_datetime(rise.datetime())

        status.rise_utc = to_utc_aware_datetime(rise.datetime())
        status.set_utc = to_utc_aware_datetime(set_.datetime())
    except ephem.AlwaysUpError:
        status.visible_now = True
        status.circumpolar = True
        status.window_contains_epoch = True
    except ephem.NeverUpError:
        status.visible_now = False
        status.never_up = True

    return status


def build_visibility_status_lines(visibility_window):
    """Return human-readable terminal lines for one visibility window state."""

    lines = []
    if visibility_window.never_up:
        lines.append("  visibility status      : NOT VISIBLE")
        lines.append(
            "  next visible start     : None (target never rises above {:.3f} deg)".format(
                visibility_window.min_elevation_deg
            )
        )
    elif visibility_window.circumpolar:
        lines.append("  visibility status      : VISIBLE")
        lines.append(
            "  visibility window      : always above {:.3f} deg".format(visibility_window.min_elevation_deg)
        )
        if visibility_window.transit_utc is not None:
            lines.append("  next transit           : {}".format(format_dual_timestamp(visibility_window.transit_utc)))
    elif visibility_window.visible_now:
        lines.append("  visibility status      : VISIBLE")
        lines.append("  current window start   : {}".format(format_dual_timestamp(visibility_window.rise_utc)))
        lines.append("  current window end     : {}".format(format_dual_timestamp(visibility_window.set_utc)))
        if visibility_window.transit_utc is not None:
            lines.append("  next transit           : {}".format(format_dual_timestamp(visibility_window.transit_utc)))
    else:
        lines.append("  visibility status      : NOT VISIBLE")
        lines.append("  next visible start     : {}".format(format_dual_timestamp(visibility_window.next_rise_utc)))
        lines.append("  next visible end       : {}".format(format_dual_timestamp(visibility_window.set_utc)))
        if visibility_window.transit_utc is not None:
            lines.append("  next transit           : {}".format(format_dual_timestamp(visibility_window.transit_utc)))
    return lines


def atomic_replace(src_path, dst_path):
    """Atomically replace a file on both Windows and POSIX."""

    if hasattr(os, "replace"):
        os.replace(src_path, dst_path)
        return

    if os.name == "nt" and os.path.exists(dst_path):
        os.remove(dst_path)
    os.rename(src_path, dst_path)


def write_text_temp_file(final_path, text):
    """Write UTF-8 text to a temporary file and fsync it."""

    tmp_path = final_path + ".tmp"
    with io.open(tmp_path, "w", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    return tmp_path


def build_md5_output_path(path):
    """Replace the source file's final suffix with .md5."""

    root, ext = os.path.splitext(path)
    if ext:
        return root + ".md5"
    return path + ".md5"


def compute_file_md5(path, chunk_size=1024 * 1024):
    """Return the hex MD5 digest for one local file."""

    digest = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_md5_file_for_output(path, md5_path=None):
    """Write one UTF-8 .md5 sidecar for the requested output file."""

    if md5_path is None:
        md5_path = build_md5_output_path(path)
    checksum = compute_file_md5(path)
    md5_tmp = write_text_temp_file(md5_path, "{}  {}\n".format(checksum, os.path.basename(path)))
    atomic_replace(md5_tmp, md5_path)
    return md5_path, checksum


def build_primary_beam_validation_for_rows(beam_offset_rows, dish_diameter_m, f0_hz):
    """Check whether all beam centres stay within the single-dish primary-beam FWHM radius."""

    if dish_diameter_m <= 0.0:
        raise ValueError("dish_diameter_m must be > 0, got {}".format(dish_diameter_m))
    if f0_hz <= 0.0:
        raise ValueError("f0_hz must be > 0, got {}".format(f0_hz))

    lambda_m = LIGHT_SPEED_M_PER_S / float(f0_hz)
    theta_pb_fwhm_rad = 1.02 * lambda_m / float(dish_diameter_m)
    theta_pb_fwhm_deg = float(np.rad2deg(theta_pb_fwhm_rad))
    allowed_radius_rad = theta_pb_fwhm_rad / 2.0
    allowed_radius_deg = theta_pb_fwhm_deg / 2.0

    max_offset_rad = max(float(row["offset_rad"]) for row in beam_offset_rows)
    max_offset_deg = max(float(row["offset_deg"]) for row in beam_offset_rows)
    offending_rows = [row for row in beam_offset_rows if float(row["offset_rad"]) > allowed_radius_rad]
    offending_ids = [int(row["beam_id"]) for row in offending_rows]
    within_primary_beam_fwhm = not offending_rows

    if within_primary_beam_fwhm:
        validation_message = (
            "PASS: all 32 beam centres stay within the single-dish primary-beam "
            "FWHM radius (max offset {:.8f} deg <= {:.8f} deg)."
        ).format(max_offset_deg, allowed_radius_deg)
    else:
        offending_text = ", ".join(
            "Beam {beam_id} ({offset_deg:.8f} deg)".format(
                beam_id=int(row["beam_id"]),
                offset_deg=float(row["offset_deg"]),
            )
            for row in offending_rows
        )
        validation_message = (
            "Beam offsets exceed the single-dish primary-beam FWHM radius. "
            "Allowed <= {:.8f} deg (theta_pb_fwhm / 2), got max {:.8f} deg. "
            "Offending beams: {}"
        ).format(allowed_radius_deg, max_offset_deg, offending_text)

    return {
        "theta_pb_fwhm_rad": theta_pb_fwhm_rad,
        "theta_pb_fwhm_deg": theta_pb_fwhm_deg,
        "primary_beam_fwhm_radius_rad": allowed_radius_rad,
        "primary_beam_fwhm_radius_deg": allowed_radius_deg,
        "max_offset_rad": max_offset_rad,
        "max_offset_deg": max_offset_deg,
        "within_primary_beam_fwhm": within_primary_beam_fwhm,
        "primary_beam_check_status": "PASS" if within_primary_beam_fwhm else "FAIL",
        "primary_beam_offending_beam_ids": offending_ids,
        "primary_beam_validation_message": validation_message,
    }


def geodetic_to_ecef_m(lat_rad, lon_rad, height_m):
    """Convert WGS84 geodetic coordinates to ECEF meters."""

    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    sin_lon = math.sin(lon_rad)
    cos_lon = math.cos(lon_rad)
    prime_vertical_radius = WGS84_A_M / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)

    x_m = (prime_vertical_radius + height_m) * cos_lat * cos_lon
    y_m = (prime_vertical_radius + height_m) * cos_lat * sin_lon
    z_m = (prime_vertical_radius * (1.0 - WGS84_E2) + height_m) * sin_lat
    return np.asarray([x_m, y_m, z_m], dtype=np.float64)


def ecef_to_local_enu_m(xyz_ecef_m, ref_xyz_ecef_m, ref_lat_rad, ref_lon_rad):
    """Convert an ECEF point to ENU meters around the reference site."""

    dx, dy, dz = np.asarray(xyz_ecef_m, dtype=np.float64) - np.asarray(ref_xyz_ecef_m, dtype=np.float64)
    sin_lat = math.sin(ref_lat_rad)
    cos_lat = math.cos(ref_lat_rad)
    sin_lon = math.sin(ref_lon_rad)
    cos_lon = math.cos(ref_lon_rad)

    east_m = -sin_lon * dx + cos_lon * dy
    north_m = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    up_m = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz
    return np.asarray([east_m, north_m, up_m], dtype=np.float64)


def read_antenna_file(txt_path):
    """Read antenna rows from the configured text file."""

    rows = []
    with io.open(txt_path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if (not line) or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(
                    "Line {} in {} must contain exactly 5 columns: "
                    "name lat lon height diameter".format(lineno, txt_path)
                )

            name = parts[0]
            lat_deg = float(parts[1])
            lon_deg = float(parts[2])
            height_m = float(parts[3])
            diameter_m = float(parts[4])

            lat_rad = np.deg2rad(lat_deg)
            lon_rad = np.deg2rad(lon_deg)
            ecef_m = geodetic_to_ecef_m(lat_rad, lon_rad, height_m)
            rows.append(
                {
                    "name": name,
                    "lat_deg": lat_deg,
                    "lon_deg": lon_deg,
                    "height_m": height_m,
                    "diameter_m": diameter_m,
                    "lat_rad": lat_rad,
                    "lon_rad": lon_rad,
                    "ecef_m": ecef_m,
                }
            )

    if not rows:
        raise ValueError("No antenna rows found in {}".format(txt_path))
    get_active_signal_inputs(len(rows))

    ref_row = rows[0]
    ref_ecef_m = ref_row["ecef_m"]
    ref_lat_rad = ref_row["lat_rad"]
    ref_lon_rad = ref_row["lon_rad"]

    antennas = []
    for row in rows:
        enu_m = ecef_to_local_enu_m(
            row["ecef_m"],
            ref_ecef_m,
            ref_lat_rad,
            ref_lon_rad,
        )
        antennas.append(
            AntennaRecord(
                name=row["name"],
                lat_deg=row["lat_deg"],
                lon_deg=row["lon_deg"],
                height_m=row["height_m"],
                diameter_m=row["diameter_m"],
                lat_rad=row["lat_rad"],
                lon_rad=row["lon_rad"],
                ecef_m=row["ecef_m"],
                enu_m=enu_m,
            )
        )
    return antennas


def normalize_beam_offset_row(index, row):
    """Normalize one beam offset row into a dict form."""

    if isinstance(row, dict):
        beam_id = int(row.get("beam_id", index))
        d_east_deg = float(row["dEast_deg"])
        d_north_deg = float(row["dNorth_deg"])
        q = int(row["q"]) if "q" in row else 0
        r = int(row["r"]) if "r" in row else 0
    else:
        arr = np.asarray(row, dtype=np.float64).reshape(-1)
        if arr.size < 2:
            raise ValueError("Beam offset row {} must contain at least dEast_deg and dNorth_deg".format(index))
        beam_id = index
        d_east_deg = float(arr[0])
        d_north_deg = float(arr[1])
        q = 0
        r = 0

    offset_deg = float(np.hypot(d_east_deg, d_north_deg))
    d_east_rad = float(np.deg2rad(d_east_deg))
    d_north_rad = float(np.deg2rad(d_north_deg))
    offset_rad = float(np.hypot(d_east_rad, d_north_rad))
    position_angle_deg = wrap_deg_360(np.rad2deg(math.atan2(d_north_deg, d_east_deg)))
    return {
        "beam_id": beam_id,
        "q": q,
        "r": r,
        "dEast_deg": d_east_deg,
        "dEast_rad": d_east_rad,
        "dNorth_deg": d_north_deg,
        "dNorth_rad": d_north_rad,
        "offset_deg": offset_deg,
        "offset_rad": offset_rad,
        "position_angle_deg": position_angle_deg,
    }


def load_beam_offset_table_from_file(path):
    """Load a 32-beam offset table from the external beam-offset report."""

    rows = []
    with io.open(path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if (not line) or line.startswith("#"):
                continue
            parts = line.split()
            try:
                file_beam_id = int(parts[0])
            except (ValueError, IndexError):
                continue

            if len(parts) < 5:
                raise ValueError(
                    "Line {} in {} must contain at least: BeamID q r dEast_deg dNorth_deg".format(
                        lineno,
                        path,
                    )
                )

            row = normalize_beam_offset_row(
                len(rows),
                {
                    "beam_id": file_beam_id,
                    "q": _parse_optional_int_token(parts[1]),
                    "r": _parse_optional_int_token(parts[2]),
                    "dEast_deg": float(parts[3]),
                    "dNorth_deg": float(parts[4]),
                },
            )
            row["file_beam_id"] = file_beam_id
            rows.append(row)

    meta = {
        "source": "BEAM_OFFSET_TABLE_FILE: {}".format(path),
        "source_mode": "file",
        "beam_definition_file": path,
        "beam_definition_module": None,
    }
    return rows, meta


def resolve_beam_offset_table(antennas, beam_offset_table_file=None):
    """Load the required external 32-beam offset table."""

    path = beam_offset_table_file or BEAM_OFFSET_TABLE_FILE
    if not path:
        raise ValueError(
            "BEAM_OFFSET_TABLE_FILE must be set; automatic beam offset generation is disabled."
        )
    if antennas is None or len(antennas) == 0:
        raise ValueError("resolve_beam_offset_table requires at least one antenna record for validation.")

    rows, meta = load_beam_offset_table_from_file(path)
    if len(rows) != TOTAL_BEAMS:
        raise ValueError("Beam offset table must contain exactly {} rows, got {}".format(TOTAL_BEAMS, len(rows)))

    file_beam_ids = [row.get("file_beam_id", row["beam_id"]) for row in rows]
    if sorted(file_beam_ids) == list(range(1, TOTAL_BEAMS + 1)):
        for row in rows:
            row["beam_id"] = row["file_beam_id"] - 1
    elif sorted(file_beam_ids) == list(range(TOTAL_BEAMS)):
        for row in rows:
            row["beam_id"] = row["file_beam_id"]
    else:
        raise ValueError(
            "Beam offset table beam_id values must be either 1..{} or 0..{}, got {}".format(
                TOTAL_BEAMS,
                TOTAL_BEAMS - 1,
                sorted(file_beam_ids),
            )
        )

    rows = sorted(rows, key=lambda row: row["beam_id"])
    actual_ids = [row["beam_id"] for row in rows]
    expected_ids = list(range(TOTAL_BEAMS))
    if actual_ids != expected_ids:
        raise ValueError("Beam offset table beam_id values must be 0..31 in order, got {}".format(actual_ids))

    for row in rows:
        row["beam_index"] = row["beam_id"]

    meta = dict(meta)
    meta["primary_beam_validation"] = build_primary_beam_validation_for_rows(
        rows,
        dish_diameter_m=float(antennas[0].diameter_m),
        f0_hz=CENTER_FREQ_HZ,
    )
    return rows, meta


def compute_beam_model_state(
    ref_ant,
    katpoint_target,
    katpoint_antennas,
    antenna_names,
    epoch_time_utc,
    ra_rad,
    dec_rad,
    beam_offset_rows,
):
    """Compute current main-beam and 32-beam directions for one epoch."""

    time_context = build_epoch_time_context(epoch_time_utc)
    source_state = compute_source_state(
        ref_ant,
        epoch_time_utc,
        ra_rad,
        dec_rad,
        gmst_rad=time_context["gmst_rad"],
    )
    antenna_beam_results = collect_antenna_beam_results(
        katpoint_target,
        katpoint_antennas,
        antenna_names,
        epoch_time_utc,
        beam_offset_rows,
    )
    reference_result = antenna_beam_results[0]
    return {
        "time_context": time_context,
        "source_state": source_state,
        "beam_dirs_enu": np.asarray(reference_result["beam_vectors_enu"], dtype=np.float64),
        "beam_el_rad": np.asarray(reference_result["beam_el_rad"], dtype=np.float64),
        "beam_el_deg": np.asarray(reference_result["beam_el_deg"], dtype=np.float64),
        "beam_az_deg": np.asarray(reference_result["beam_az_deg"], dtype=np.float64),
        "antenna_beam_results": antenna_beam_results,
    }


def compute_abomega(
    ref_ant,
    antennas,
    katpoint_target,
    katpoint_antennas,
    antenna_names,
    epoch_time_utc,
    ra_rad,
    dec_rad,
    beam_offset_rows,
    state_tn=None,
):
    """Compute per-antenna A[N_ant, 32], B[N_ant, 32], and omega[N_ant, 32]."""

    minus_dt = epoch_time_utc - timedelta(seconds=OMEGA_DELTA_SECONDS)
    plus_dt = epoch_time_utc + timedelta(seconds=OMEGA_DELTA_SECONDS)

    state_minus = compute_beam_model_state(
        ref_ant,
        katpoint_target,
        katpoint_antennas,
        antenna_names,
        minus_dt,
        ra_rad,
        dec_rad,
        beam_offset_rows,
    )
    if state_tn is None:
        state_tn = compute_beam_model_state(
            ref_ant,
            katpoint_target,
            katpoint_antennas,
            antenna_names,
            epoch_time_utc,
            ra_rad,
            dec_rad,
            beam_offset_rows,
        )
    state_plus = compute_beam_model_state(
        ref_ant,
        katpoint_target,
        katpoint_antennas,
        antenna_names,
        plus_dt,
        ra_rad,
        dec_rad,
        beam_offset_rows,
    )

    antenna_count = len(antennas)
    ab_scale = 2.0 * np.pi / LIGHT_SPEED_M_PER_S
    a_ant = np.zeros((antenna_count, TOTAL_BEAMS), dtype=np.float32)
    b_ant = np.zeros((antenna_count, TOTAL_BEAMS), dtype=np.float32)
    omega_ant = np.zeros((antenna_count, TOTAL_BEAMS), dtype=np.float32)

    for ant_index, ant in enumerate(antennas):
        antenna_state_tn = state_tn["antenna_beam_results"][ant_index]
        antenna_state_minus = state_minus["antenna_beam_results"][ant_index]
        antenna_state_plus = state_plus["antenna_beam_results"][ant_index]
        baseline_enu_m = np.asarray(ant.enu_m, dtype=np.float64)

        for beam_index in range(TOTAL_BEAMS):
            beam_dir_tn = antenna_state_tn["beam_vectors_enu"][beam_index]
            theta_rad = float(antenna_state_tn["beam_el_rad"][beam_index])
            cos_theta = float(math.cos(theta_rad))
            x_hat, y_hat, _z_hat = build_source_aligned_horizontal_frame(beam_dir_tn)

            phi_minus = compute_local_phase_angle_rad(
                antenna_state_minus["beam_vectors_enu"][beam_index],
                x_hat,
                y_hat,
            )
            phi_plus = compute_local_phase_angle_rad(
                antenna_state_plus["beam_vectors_enu"][beam_index],
                x_hat,
                y_hat,
            )
            omega_ant[ant_index, beam_index] = np.float32(
                signed_angle_delta_rad(phi_plus, phi_minus) / (2.0 * OMEGA_DELTA_SECONDS)
            )

            bx_m = float(np.dot(baseline_enu_m, x_hat))
            by_m = float(np.dot(baseline_enu_m, y_hat))
            a_ant[ant_index, beam_index] = np.float32(ab_scale * bx_m * cos_theta)
            b_ant[ant_index, beam_index] = np.float32(ab_scale * by_m * cos_theta)

    return {
        "a_ant": a_ant,
        "b_ant": b_ant,
        "omega_ant": omega_ant,
    }


def expand_antennas_to_twenty_rows(ant_beam_matrix):
    """Duplicate each antenna row into two inputs and zero-pad the missing inputs."""

    ant_beam_matrix = np.asarray(ant_beam_matrix, dtype=np.float32)
    if ant_beam_matrix.ndim != 2 or ant_beam_matrix.shape[1] != TOTAL_BEAMS:
        raise ValueError(
            "Expected an N x {} antenna/beam matrix, got {}".format(
                TOTAL_BEAMS,
                ant_beam_matrix.shape,
            )
        )

    antenna_count = int(ant_beam_matrix.shape[0])
    get_active_signal_inputs(antenna_count)

    out = np.zeros((TOTAL_SIGNAL_INPUTS, TOTAL_BEAMS), dtype=np.float32)
    for ant_index in range(antenna_count):
        signal_start = ant_index * SIGNALS_PER_ANTENNA
        signal_stop = signal_start + SIGNALS_PER_ANTENNA
        out[signal_start:signal_stop, :] = ant_beam_matrix[ant_index][None, :]
    return out


def write_trace_mode_phase_file(a_mat, b_mat, w_mat, out_path=None):
    """Write one text trace-mode parameter file and its MD5 sidecar."""

    expected_shape = (TOTAL_SIGNAL_INPUTS, TOTAL_BEAMS)
    matrices = {
        "A": np.asarray(a_mat, dtype=np.float32),
        "B": np.asarray(b_mat, dtype=np.float32),
        "omega": np.asarray(w_mat, dtype=np.float32),
    }
    for name, mat in matrices.items():
        if mat.shape != expected_shape:
            raise ValueError(
                "{} matrix shape mismatch: got {}, expected {}".format(
                    name,
                    mat.shape,
                    expected_shape,
                )
            )

    rows = [
        matrices["A"].reshape(-1),
        matrices["B"].reshape(-1),
        matrices["omega"].reshape(-1),
    ]
    for name, row in zip(["A", "B", "omega"], rows):
        if row.size != TOTAL_SIGNAL_INPUTS * TOTAL_BEAMS:
            raise ValueError(
                "{} row value count mismatch: got {}, expected {}".format(
                    name,
                    row.size,
                    TOTAL_SIGNAL_INPUTS * TOTAL_BEAMS,
                )
            )

    if out_path is None:
        out_path = TRACE_MODE_PHASE_TXT

    text = "\n".join(" ".join("{:.12f}".format(float(value)) for value in row) for row in rows)
    tmp_path = write_text_temp_file(out_path, text)
    atomic_replace(tmp_path, out_path)
    write_md5_file_for_output(out_path)
    return np.vstack(rows).astype(np.float32, copy=False)


def print_trace_step(title):
    """Print one visible pipeline step header."""

    if PRINT_PIPELINE_STEPS:
        print("")
        print("=== {} ===".format(title))


def print_one_shot_configuration(args, target_ra, target_dec, t0_utc):
    """Print the one-shot input configuration before computation starts."""

    print_trace_step("Input Configuration")
    print("target_ra                : {}".format(target_ra))
    print("target_dec               : {}".format(target_dec))
    print("t0_utc                   : {}".format(t0_utc.strftime("%Y-%m-%d %H:%M:%S")))
    print("target_name              : {}".format(args.target_name))
    print("ants_txt                 : {}".format(ANTS_TXT))
    print("beam_offset_table        : {}".format(BEAM_OFFSET_TABLE_FILE))
    print("min_elevation_deg        : {:.3f}".format(MIN_ELEVATION_DEG))
    print("ignore_visibility        : {}".format(args.ignore_visibility))
    print("output_directory         : {}".format(OUTPUT_DIR))
    print("center_freq_hz           : {:.6f}".format(CENTER_FREQ_HZ))
    print("omega_delta_seconds      : {:.6f}".format(OMEGA_DELTA_SECONDS))
    print("trace_output_file        : {}".format(TRACE_MODE_PHASE_TXT))
    print("trace_md5_file           : {}".format(TRACE_MODE_PHASE_MD5))
    print("other output files       : disabled")
    print("trace_layout             : 3 text lines: A, B, omega")
    print("trace_values_per_line    : {}".format(TOTAL_SIGNAL_INPUTS * TOTAL_BEAMS))
    print("trace_line_count         : 3")
    print("trace_total_values       : {}".format(TRACE_STREAM_FLOAT_COUNT))
    print("trace_value_format       : text plain decimal notation")


def run_one_shot_trace_phase_coeff(args):
    """Compute one A/B/omega solution at the requested RA/Dec and UTC time, then exit."""

    target_ra, target_dec = args.target_ra, args.target_dec
    t0_utc = args.t0_utc
    print_one_shot_configuration(args, target_ra, target_dec, t0_utc)

    print_trace_step("Step 1: Load Antennas")
    antennas = read_antenna_file(ANTS_TXT)
    active_signal_inputs = get_active_signal_inputs(len(antennas))
    padded_signal_inputs = TOTAL_SIGNAL_INPUTS - active_signal_inputs
    print("loaded antennas          : {}".format(len(antennas)))
    print("active signal inputs     : {}".format(active_signal_inputs))
    print("padded signal inputs     : {}".format(padded_signal_inputs))

    print_trace_step("Step 2: Resolve Beam Offset Table")
    beam_offset_rows, beam_offset_meta = resolve_beam_offset_table(
        antennas=antennas,
        beam_offset_table_file=BEAM_OFFSET_TABLE_FILE,
    )
    print("beam_offset_source       : {}".format(beam_offset_meta.get("source")))
    print("beam_offset_rows         : {}".format(len(beam_offset_rows)))
    print("beam_offset_table_file   : {}".format(BEAM_OFFSET_TABLE_FILE))
    primary_beam_validation = beam_offset_meta.get("primary_beam_validation")
    if primary_beam_validation is not None:
        print("primary_beam_status      : {}".format(primary_beam_validation["primary_beam_check_status"]))

    print_trace_step("Step 3: Build Target And Time Context")
    ra_rad = parse_ra_to_rad(target_ra)
    dec_rad = parse_dec_to_rad(target_dec)
    katpoint_antennas, antenna_names = build_katpoint_antennas_from_records(antennas)
    katpoint_target = build_katpoint_target_from_radec(target_ra, target_dec)
    print("target_name              : {}".format(args.target_name))
    print("target_ra                : {}".format(target_ra))
    print("target_dec               : {}".format(target_dec))
    print("t0_utc                   : {}".format(t0_utc.strftime("%Y-%m-%d %H:%M:%S")))

    print_trace_step("Step 4: Compute Source State At t0")
    state_t0 = compute_beam_model_state(
        antennas[0],
        katpoint_target,
        katpoint_antennas,
        antenna_names,
        t0_utc,
        ra_rad,
        dec_rad,
        beam_offset_rows,
    )
    visibility_window = compute_visibility_window(
        antennas[0],
        t0_utc,
        target_ra,
        target_dec,
        MIN_ELEVATION_DEG,
    )
    visible_now = bool(visibility_window.visible_now)
    simulation_override = bool((not visible_now) and args.ignore_visibility)
    if (not visible_now) and (not args.ignore_visibility):
        raise RuntimeError(
            "Target RA/Dec {} {} is below the minimum elevation {:.3f} deg at UTC {}. "
            "Re-run with --ignore-visibility if you still want one-shot output.".format(
                target_ra,
                target_dec,
                MIN_ELEVATION_DEG,
                t0_utc.strftime("%Y-%m-%d %H:%M:%S"),
            )
        )

    print("visible_now              : {}".format(visible_now))
    print("simulation_override      : {}".format(simulation_override))
    print("main_az_deg              : {:.8f}".format(state_t0["source_state"]["azimuth_deg"]))
    print("main_el_deg              : {:.8f}".format(state_t0["source_state"]["elevation_deg"]))
    print("lst_hms                  : {}".format(state_t0["source_state"]["lst_hms"]))
    for line in build_visibility_status_lines(visibility_window):
        print(line)

    print_trace_step("Step 5: Compute A/B/Omega At t0")
    abomega = compute_abomega(
        antennas[0],
        antennas,
        katpoint_target,
        katpoint_antennas,
        antenna_names,
        t0_utc,
        ra_rad,
        dec_rad,
        beam_offset_rows,
        state_tn=state_t0,
    )
    a_mat = expand_antennas_to_twenty_rows(abomega["a_ant"])
    b_mat = expand_antennas_to_twenty_rows(abomega["b_ant"])
    w_mat = expand_antennas_to_twenty_rows(abomega["omega_ant"])
    print("A/B/omega matrix shape   : {} / {} / {}".format(a_mat.shape, b_mat.shape, w_mat.shape))
    print("omega_method             : central difference using t0-dt and t0+dt")

    print_trace_step("Step 6: Write Trace Output")
    write_trace_mode_phase_file(a_mat, b_mat, w_mat, out_path=TRACE_MODE_PHASE_TXT)
    print("trace_mode_phase_file    : {}".format(TRACE_MODE_PHASE_TXT))
    print("trace_mode_phase_md5     : {}".format(TRACE_MODE_PHASE_MD5))
    print("other output files       : disabled")
    print("one-shot computation complete")


def main():
    """Compute one trace-mode parameter file and exit."""

    args = parse_args()
    configure_ants_file(args.ants_txt)
    configure_beam_offset_table_file(args.beam_offset_table)
    configure_output_dir(args.output_dir)
    run_one_shot_trace_phase_coeff(args)


if __name__ == "__main__":
    main()
