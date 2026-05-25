#!/usr/bin/env python3
"""Reusable helpers for 32-beam products derived from a fixed az/el pointing.

Recommended entry points:
- compute_beam_products_from_fixed_azel: run the full computation pipeline
- print_beam_products_summary: print the same short summary used by the script
"""

from __future__ import division, print_function

import io
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import ephem
import katpoint
import numpy as np


DEFAULT_INTERNAL_ANCHOR_INDEX = 0
DEFAULT_COEFF_TOTAL_FREQ_CHANNELS = 2052


@dataclass(frozen=True)
class PointingConfig:
    """Fixed az/el pointing used to derive the frozen sky target."""

    az_deg: float = 0.0
    el_deg: float = 45.0
    reference_antenna_index: int = 0


@dataclass(frozen=True)
class OutputPaths:
    """File paths for all direction-vector and beam-coefficient outputs."""

    direction_txt: str
    direction_npz: str
    beam_coeff_txt: str
    beam_coeff_npz: str


@dataclass(frozen=True)
class RuntimeConfig:
    """User-adjustable runtime parameters for the fixed az/el pipeline."""

    ants_txt: str
    first_solution_time_utc: Optional[str]
    n_freq_channels: int
    freq_start_hz: float
    freq_stop_hz: float
    total_signal_inputs: int
    signals_per_antenna: int
    default_alt_m: float
    default_diam_m: float
    light_speed_m_per_s: float


def build_default_output_paths(base_dir):
    """Return the default output filenames used by the original script."""

    return OutputPaths(
        direction_txt=os.path.join(base_dir, "beam32_direction_vectors_from_azel.txt"),
        direction_npz=os.path.join(base_dir, "beam32_direction_vectors_from_azel.npz"),
        beam_coeff_txt=os.path.join(base_dir, "beam_coeff_from_azel.txt"),
        beam_coeff_npz=os.path.join(base_dir, "beam_coeff_from_azel.npz"),
    )


# Geometry and coordinate helpers.

def utc_now():
    """Return the current timezone-aware UTC time."""

    return datetime.now(timezone.utc)


def format_utc_timestamp(when_utc):
    """Format a timezone-aware UTC datetime for logs and terminal output."""

    if when_utc is None:
        return "None"
    return ensure_utc_datetime(when_utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def format_local_timestamp(when_local):
    """Format a local timezone-aware datetime."""

    if when_local is None:
        return "None"
    return when_local.strftime("%Y-%m-%d %H:%M:%S %Z")


def ensure_utc_datetime(when_utc):
    """Normalize a datetime to timezone-aware UTC."""

    if when_utc is None:
        return None
    if when_utc.tzinfo is None:
        return when_utc.replace(tzinfo=timezone.utc)
    return when_utc.astimezone(timezone.utc)


def format_dual_timestamp(when_utc):
    """Format one UTC instant in both UTC and local time for operators."""

    when_utc = ensure_utc_datetime(when_utc)
    if when_utc is None:
        return "None"
    local_dt = when_utc.astimezone()
    return "{} UTC | {}".format(
        when_utc.strftime("%Y-%m-%d %H:%M:%S"),
        local_dt.strftime("%Y-%m-%d %H:%M:%S %Z"),
    )


def ceil_utc_to_next_slot_boundary(when_utc, slot_seconds):
    """Round a UTC time upward to the next slot boundary."""

    unix_seconds = ensure_utc_datetime(when_utc).timestamp()
    aligned_seconds = int(math.ceil(unix_seconds / float(slot_seconds)) * slot_seconds)
    return datetime.fromtimestamp(aligned_seconds, tz=timezone.utc)

def deg_to_dms_str(val):
    """Convert signed decimal degrees to the D:M:S string format used by katpoint."""

    sign = "-" if val < 0 else ""
    v = abs(val)
    d = int(v)
    m_float = (v - d) * 60.0
    m = int(m_float)
    s = round((m_float - m) * 60.0, 4)
    if s >= 60.0:
        s -= 60.0
        m += 1
    if m >= 60:
        m -= 60
        d += 1
    return "{}{}:{:02d}:{:07.4f}".format(sign, d, m, s)


def load_antennas(txt_path, default_alt_m, default_diam_m):
    """Load antenna definitions from a text file."""

    ants = []
    names = []
    with io.open(txt_path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if (not line) or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                raise ValueError("Line {}: need at least name lat lon".format(lineno))

            name = parts[0]
            lat_deg = float(parts[1])
            lon_deg = float(parts[2])
            alt_m = float(parts[3]) if len(parts) >= 4 else default_alt_m
            diam_m = float(parts[4]) if len(parts) >= 5 else default_diam_m

            desc = "{}, {}, {}, {}, {}".format(
                name,
                deg_to_dms_str(lat_deg),
                deg_to_dms_str(lon_deg),
                alt_m,
                diam_m,
            )
            ants.append(katpoint.Antenna(desc))
            names.append(name)

    if not ants:
        raise RuntimeError("No antennas found in {}".format(txt_path))

    return ants, names


def build_frequency_axis_hz(runtime_config):
    """Build the active frequency axis used for beam coefficients."""

    return np.linspace(
        runtime_config.freq_start_hz,
        runtime_config.freq_stop_hz,
        runtime_config.n_freq_channels,
        dtype=np.float64,
    )


def resolve_first_solution_time(first_solution_time_utc):
    """Resolve the requested solution time or fall back to current UTC."""

    if first_solution_time_utc is None:
        return datetime.utcnow().replace(microsecond=0)
    return datetime.strptime(first_solution_time_utc, "%Y-%m-%d %H:%M:%S")


def get_target_horizontal_coords(target, ant, when_utc):
    """Get target altitude and azimuth for one antenna at one UTC time."""

    observer = ant.ref_observer
    observer.date = ephem.Date(ensure_utc_datetime(when_utc))
    target.body.compute(observer)
    alt_deg = float(target.body.alt) * 180.0 / np.pi
    az_deg = float(target.body.az) * 180.0 / np.pi
    return alt_deg, az_deg


def to_katpoint_timestamp(when_utc):
    """Convert a UTC datetime into the katpoint timestamp format used downstream."""

    when_utc = ensure_utc_datetime(when_utc)
    return katpoint.Timestamp(when_utc.strftime("%Y-%m-%d %H:%M:%S"))


def build_katpoint_antenna_from_record(ant):
    """Build one katpoint.Antenna from an antenna-like record."""

    desc = "{}, {}, {}, {}, {}".format(
        ant.name,
        deg_to_dms_str(ant.lat_deg),
        deg_to_dms_str(ant.lon_deg),
        ant.height_m,
        ant.diameter_m,
    )
    return katpoint.Antenna(desc)


def build_katpoint_antennas_from_records(antennas):
    """Build katpoint antenna objects matching parsed antenna records."""

    katpoint_antennas = [build_katpoint_antenna_from_record(ant) for ant in antennas]
    antenna_names = [ant.name for ant in antennas]
    return katpoint_antennas, antenna_names


def build_katpoint_target_from_radec(target_ra_text, target_dec_text):
    """Build one frozen katpoint target from RA/Dec text."""

    return katpoint.Target("Beam32TrackingTarget, radec, {}, {}".format(target_ra_text, target_dec_text))


def normalize(vec):
    """Return a unit-length copy of the input vector."""

    vec = np.asarray(vec, dtype=np.float64)
    norm = np.linalg.norm(vec)
    if (not np.isfinite(norm)) or norm == 0.0:
        raise ValueError("Cannot normalize vector {}".format(vec))
    return vec / norm


def validate_pointing_inputs(az_deg, el_deg, antenna_index, n_antennas):
    """Validate the fixed az/el pointing configuration before solving RA/Dec."""

    if n_antennas <= 0:
        raise ValueError("At least one antenna is required")
    if antenna_index < 0 or antenna_index >= n_antennas:
        raise ValueError(
            "reference_antenna_index {} out of range [0, {})".format(
                antenna_index,
                n_antennas,
            )
        )
    if not np.isfinite(az_deg):
        raise ValueError("az_deg must be finite")
    if not np.isfinite(el_deg):
        raise ValueError("el_deg must be finite")
    if el_deg < 0.0 or el_deg > 90.0:
        raise ValueError("el_deg must be within [0, 90]")


def wrap_az_deg(az_deg):
    """Wrap azimuth to the [0, 360) degree range."""

    wrapped = float(az_deg) % 360.0
    if wrapped < 0.0:
        wrapped += 360.0
    return wrapped


def wrap_deg_360(angle_deg):
    """Wrap an angle in degrees to [0, 360)."""

    return wrap_az_deg(angle_deg)


def wrap_rad_2pi(angle_rad):
    """Wrap an angle in radians to [0, 2*pi)."""

    return float(angle_rad) % (2.0 * np.pi)


def signed_angle_delta_rad(current_rad, previous_rad):
    """Return the signed shortest-path difference between two angles."""

    delta = float(current_rad) - float(previous_rad)
    while delta <= -np.pi:
        delta += 2.0 * np.pi
    while delta > np.pi:
        delta -= 2.0 * np.pi
    return delta


def angular_separation_deg(direction_a, direction_b):
    """Return the angular separation between two ENU direction vectors in degrees."""

    vec_a = normalize(direction_a)
    vec_b = normalize(direction_b)
    cosine = float(np.clip(np.dot(vec_a, vec_b), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def parse_sexagesimal(text, allow_sign=False):
    """Parse HH:MM:SS or DD:MM:SS text into signed decimal hours/degrees."""

    raw = str(text).strip()
    if not raw:
        raise ValueError("Empty sexagesimal string")

    sign = 1.0
    if raw[0] in "+-":
        if not allow_sign:
            raise ValueError("Unexpected sign in {}".format(text))
        sign = -1.0 if raw[0] == "-" else 1.0
        raw = raw[1:].strip()

    parts = raw.split(":")
    if len(parts) != 3:
        raise ValueError("Expected HH:MM:SS or DD:MM:SS, got {}".format(text))

    hours_or_deg = float(parts[0])
    minutes = float(parts[1])
    seconds = float(parts[2])

    if minutes < 0.0 or minutes >= 60.0:
        raise ValueError("Minutes out of range in {}".format(text))
    if seconds < 0.0 or seconds >= 60.0:
        raise ValueError("Seconds out of range in {}".format(text))

    return sign * (hours_or_deg + minutes / 60.0 + seconds / 3600.0)


def parse_ra_to_rad(ra_text):
    """Parse RA text into radians."""

    return np.deg2rad(parse_sexagesimal(ra_text, allow_sign=False) * 15.0)


def parse_dec_to_rad(dec_text):
    """Parse declination text into radians."""

    return np.deg2rad(parse_sexagesimal(dec_text, allow_sign=True))


def rad_to_hms_string(angle_rad):
    """Format an angle in radians as HH:MM:SS.SS."""

    total_hours = wrap_rad_2pi(angle_rad) * 12.0 / np.pi
    hours = int(total_hours)
    minutes_float = (total_hours - hours) * 60.0
    minutes = int(minutes_float)
    seconds = (minutes_float - minutes) * 60.0
    return "{:02d}:{:02d}:{:05.2f}".format(hours, minutes, seconds)


def enu_vector_to_altaz_deg(vec_enu):
    """Convert a local ENU unit vector into altitude and azimuth in degrees."""

    east, north, up = normalize(vec_enu)
    alt_deg = np.degrees(np.arcsin(np.clip(up, -1.0, 1.0)))
    az_deg = np.degrees(np.arctan2(east, north))
    if az_deg < 0.0:
        az_deg += 360.0
    return alt_deg, az_deg


def get_uvw_basis_enu(target, antenna, when_utc):
    """Return normalized uvw basis vectors in ENU coordinates for one antenna."""

    basis = np.asarray(
        target.uvw_basis(timestamp=to_katpoint_timestamp(when_utc), antenna=antenna),
        dtype=np.float64,
    )
    if basis.shape != (3, 3):
        raise ValueError("Unexpected uvw basis shape {}".format(basis.shape))
    u_hat = normalize(basis[0])
    v_hat = normalize(basis[1])
    w_hat = normalize(basis[2])
    return u_hat, v_hat, w_hat


def build_epoch_time_context(utc_dt):
    """Build JD and GMST for one logical slot epoch."""

    utc_dt = ensure_utc_datetime(utc_dt)
    unix_seconds = utc_dt.timestamp()
    julian_date = unix_seconds / 86400.0 + 2440587.5
    centuries = (julian_date - 2451545.0) / 36525.0
    gmst_deg = (
        280.46061837
        + 360.98564736629 * (julian_date - 2451545.0)
        + 0.000387933 * centuries * centuries
        - (centuries ** 3) / 38710000.0
    )
    return {
        "julian_date": julian_date,
        "gmst_rad": wrap_rad_2pi(np.deg2rad(gmst_deg)),
    }


def compute_source_state(ref_ant, epoch_time_utc, ra_rad, dec_rad, gmst_rad=None):
    """Compute hour angle, az/el and ENU direction for one reference antenna."""

    if gmst_rad is None:
        gmst_rad = build_epoch_time_context(epoch_time_utc)["gmst_rad"]
    lst_rad = wrap_rad_2pi(ref_ant.lon_rad + gmst_rad)
    hour_angle_rad = lst_rad - ra_rad
    while hour_angle_rad <= -np.pi:
        hour_angle_rad += 2.0 * np.pi
    while hour_angle_rad > np.pi:
        hour_angle_rad -= 2.0 * np.pi

    sin_lat = math.sin(ref_ant.lat_rad)
    cos_lat = math.cos(ref_ant.lat_rad)
    sin_dec = math.sin(dec_rad)
    cos_dec = math.cos(dec_rad)
    sin_ha = math.sin(hour_angle_rad)
    cos_ha = math.cos(hour_angle_rad)

    east = -cos_dec * sin_ha
    north = sin_dec * cos_lat - cos_dec * cos_ha * sin_lat
    up = sin_dec * sin_lat + cos_dec * cos_ha * cos_lat
    direction_enu = normalize([east, north, up])

    elevation_rad = math.asin(np.clip(direction_enu[2], -1.0, 1.0))
    azimuth_rad = wrap_rad_2pi(math.atan2(direction_enu[0], direction_enu[1]))

    return {
        "epoch_time_utc": ensure_utc_datetime(epoch_time_utc),
        "lst_rad": lst_rad,
        "lst_hms": rad_to_hms_string(lst_rad),
        "hour_angle_rad": hour_angle_rad,
        "hour_angle_deg": np.rad2deg(hour_angle_rad),
        "azimuth_rad": azimuth_rad,
        "elevation_rad": elevation_rad,
        "azimuth_deg": wrap_deg_360(np.rad2deg(azimuth_rad)),
        "elevation_deg": np.rad2deg(elevation_rad),
        "direction_enu": direction_enu,
    }


def compute_projected_max_baseline_m(target, ants, ant_names, when_utc):
    """Find the antenna pair with the largest projected baseline for this target."""

    if len(ants) < 2:
        raise ValueError("At least two antennas are required to compute a projected baseline")

    timestamp_s = np.asarray([to_katpoint_timestamp(when_utc).secs], dtype=np.float64)
    max_info = None

    for ant1_index, ant1 in enumerate(ants[:-1]):
        for ant2_index in range(ant1_index + 1, len(ants)):
            ant2 = ants[ant2_index]
            u_m, v_m, w_m = target.uvw(ant1, timestamp_s, ant2)
            u_m = float(np.asarray(u_m, dtype=np.float64)[0])
            v_m = float(np.asarray(v_m, dtype=np.float64)[0])
            w_m = float(np.asarray(w_m, dtype=np.float64)[0])
            projected_baseline_m = float(np.hypot(u_m, v_m))
            total_baseline_m = float(np.sqrt(projected_baseline_m ** 2 + w_m ** 2))

            if max_info is None or projected_baseline_m > max_info["projected_baseline_m"]:
                max_info = {
                    "ant1_index": ant1_index,
                    "ant2_index": ant2_index,
                    "ant1_name": ant_names[ant1_index],
                    "ant2_name": ant_names[ant2_index],
                    "u_m": u_m,
                    "v_m": v_m,
                    "w_m": w_m,
                    "projected_baseline_m": projected_baseline_m,
                    "total_baseline_m": total_baseline_m,
                }

    if max_info is None or max_info["projected_baseline_m"] <= 0.0:
        raise ValueError("Failed to compute a positive projected baseline for the current target")

    return max_info


def offset_direction_vector(u_hat, v_hat, w_hat, d_east_rad, d_north_rad):
    """Offset Beam 0 by small east/north angular steps and return a new ENU vector."""

    radius_rad = np.hypot(d_east_rad, d_north_rad)
    if radius_rad == 0.0:
        return w_hat.copy()

    tangent_offset = d_east_rad * u_hat + d_north_rad * v_hat
    beam_hat = np.cos(radius_rad) * w_hat + (np.sin(radius_rad) / radius_rad) * tangent_offset
    return normalize(beam_hat)


def trace_compute_beam_vectors_for_antenna(target, antenna, antenna_name, when_utc, beam_offset_rows):
    """Compute all 32 trace-mode beam ENU vectors and horizontal coordinates for one antenna."""

    beam0_az_rad, beam0_el_rad = target.azel(to_katpoint_timestamp(when_utc), antenna)
    beam0_az_deg = wrap_az_deg(np.degrees(float(beam0_az_rad)))
    beam0_el_deg = float(np.degrees(float(beam0_el_rad)))
    u_hat, v_hat, w_hat = get_uvw_basis_enu(target, antenna, when_utc)

    beam_vectors = []
    beam_el_deg = []
    beam_az_deg = []
    per_beam_rows = []

    for row in beam_offset_rows:
        beam_hat = offset_direction_vector(u_hat, v_hat, w_hat, row["dEast_rad"], row["dNorth_rad"])
        el_deg, az_deg = enu_vector_to_altaz_deg(beam_hat)
        separation_from_beam0_deg = angular_separation_deg(beam_hat, w_hat)

        beam_vectors.append(beam_hat)
        beam_el_deg.append(el_deg)
        beam_az_deg.append(az_deg)
        per_beam_rows.append(
            {
                "beam_id": row["beam_id"],
                "q": row["q"],
                "r": row["r"],
                "dEast_deg": row["dEast_deg"],
                "dNorth_deg": row["dNorth_deg"],
                "offset_deg": row["offset_deg"],
                "position_angle_deg": row["position_angle_deg"],
                "separation_from_beam0_deg": separation_from_beam0_deg,
                "az_deg": az_deg,
                "el_deg": el_deg,
                "enu": beam_hat,
            }
        )

    return {
        "antenna_name": antenna_name,
        "beam0_az_deg": beam0_az_deg,
        "beam0_el_deg": beam0_el_deg,
        "u_hat": u_hat,
        "v_hat": v_hat,
        "w_hat": w_hat,
        "beam_vectors_enu": np.asarray(beam_vectors, dtype=np.float64),
        "beam_el_deg": np.asarray(beam_el_deg, dtype=np.float64),
        "beam_el_rad": np.deg2rad(np.asarray(beam_el_deg, dtype=np.float64)),
        "beam_az_deg": np.asarray(beam_az_deg, dtype=np.float64),
        "rows": per_beam_rows,
    }


def trace_collect_antenna_beam_results(target, katpoint_antennas, antenna_names, when_utc, beam_offset_rows):
    """Run the per-antenna 32-beam direction-vector calculation for all antennas."""

    return [
        trace_compute_beam_vectors_for_antenna(target, antenna, antenna_name, when_utc, beam_offset_rows)
        for antenna, antenna_name in zip(katpoint_antennas, antenna_names)
    ]


def trace_build_source_aligned_horizontal_frame(beam_direction_enu, angle_eps=1.0e-12):
    """Build the x/y/z basis used by the trace-mode A/B/omega model."""

    z_hat = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    horizontal = np.asarray([beam_direction_enu[0], beam_direction_enu[1], 0.0], dtype=np.float64)
    if np.linalg.norm(horizontal) < angle_eps:
        x_hat = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        y_hat = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        x_hat = normalize(horizontal)
        y_hat = normalize(np.asarray([-x_hat[1], x_hat[0], 0.0], dtype=np.float64))
    return x_hat, y_hat, z_hat


def trace_compute_local_phase_angle_rad(direction_enu, x_hat, y_hat):
    """Compute delta-phi in the local beam model around one slot epoch."""

    x_comp = float(np.dot(direction_enu, x_hat))
    y_comp = float(np.dot(direction_enu, y_hat))
    return math.atan2(y_comp, x_comp)


def solve_radec_from_fixed_azel(ants, ant_names, when_utc, config):
    """Convert the configured fixed az/el into a frozen katpoint RA/Dec target."""

    validate_pointing_inputs(
        config.az_deg,
        config.el_deg,
        config.reference_antenna_index,
        len(ants),
    )
    pointing_ref_ant = ants[config.reference_antenna_index]
    pointing_ref_name = ant_names[config.reference_antenna_index]

    observer = pointing_ref_ant.ref_observer
    observer.date = ephem.Date(when_utc)
    az_rad = np.deg2rad(wrap_az_deg(config.az_deg))
    el_rad = np.deg2rad(config.el_deg)
    ra_angle, dec_angle = observer.radec_of(az_rad, el_rad)

    ra_str = str(ra_angle)
    dec_str = str(dec_angle)
    target = katpoint.Target("Beam32PointingTarget, radec, {}, {}".format(ra_str, dec_str))

    back_az, back_el = target.azel(to_katpoint_timestamp(when_utc), pointing_ref_ant)
    back_az_deg = wrap_az_deg(np.degrees(float(back_az)))
    back_el_deg = np.degrees(float(back_el))

    return target, {
        "pointing_reference_antenna": pointing_ref_name,
        "pointing_reference_index": config.reference_antenna_index,
        "input_az_deg": wrap_az_deg(config.az_deg),
        "input_el_deg": float(config.el_deg),
        "derived_ra": ra_str,
        "derived_dec": dec_str,
        "roundtrip_az_deg": back_az_deg,
        "roundtrip_el_deg": back_el_deg,
    }


# Beam-vector construction.

def compute_beam_vectors_for_antenna(target, antenna, antenna_name, when_utc, beam_rows):
    """Compute all 32 beam ENU vectors and horizontal coordinates for one antenna."""

    beam0_alt_deg, beam0_az_deg = get_target_horizontal_coords(target, antenna, when_utc)
    u_hat, v_hat, w_hat = get_uvw_basis_enu(target, antenna, when_utc)

    beam_vectors = []
    beam_alt_deg = []
    beam_az_deg = []
    per_beam_rows = []

    for row in beam_rows:
        beam_hat = offset_direction_vector(u_hat, v_hat, w_hat, row["dEast_rad"], row["dNorth_rad"])
        alt_deg, az_deg = enu_vector_to_altaz_deg(beam_hat)

        beam_vectors.append(beam_hat)
        beam_alt_deg.append(alt_deg)
        beam_az_deg.append(az_deg)
        per_beam_rows.append(
            {
                "beam_id": row["beam_id"],
                "q": row["q"],
                "r": row["r"],
                "dEast_deg": row["dEast_deg"],
                "dNorth_deg": row["dNorth_deg"],
                "offset_deg": row["offset_deg"],
                "position_angle_deg": row["position_angle_deg"],
                "az_deg": az_deg,
                "alt_deg": alt_deg,
                "enu": beam_hat,
            }
        )

    return {
        "antenna_name": antenna_name,
        "beam0_alt_deg": beam0_alt_deg,
        "beam0_az_deg": beam0_az_deg,
        "u_hat": u_hat,
        "v_hat": v_hat,
        "w_hat": w_hat,
        "beam_vectors_enu": np.asarray(beam_vectors, dtype=np.float64),
        "beam_alt_deg": np.asarray(beam_alt_deg, dtype=np.float64),
        "beam_az_deg": np.asarray(beam_az_deg, dtype=np.float64),
        "rows": per_beam_rows,
    }


def collect_antenna_results(target, ants, ant_names, when_utc, beam_rows):
    """Run the 32-beam direction-vector calculation for every antenna."""

    return [
        compute_beam_vectors_for_antenna(target, ant, name, when_utc, beam_rows)
        for ant, name in zip(ants, ant_names)
    ]


# Delay and phase-coefficient helpers.

def build_signal_labels(antenna_names, total_signal_inputs, signals_per_antenna):
    """Build hardware input labels for active antenna/polarization signals plus padding."""

    n_active_signal_inputs = len(antenna_names) * signals_per_antenna
    if n_active_signal_inputs > total_signal_inputs:
        raise ValueError(
            "Antenna signal count {} exceeds available hardware inputs {}".format(
                n_active_signal_inputs,
                total_signal_inputs,
            )
        )

    labels = []
    for name in antenna_names:
        for pol_index in range(signals_per_antenna):
            labels.append("{}_pol{}".format(name, pol_index))
    for signal_index in range(n_active_signal_inputs, total_signal_inputs):
        labels.append("unused_{:02d}".format(signal_index))
    return labels, n_active_signal_inputs


def compute_delay_series_to_internal_anchor_for_beam(
    anchor_ant,
    ants,
    beam_vector_enu,
    anchor_index,
    light_speed_m_per_s,
):
    """Compute geometric delays from the chosen anchor antenna toward every antenna."""

    delay_series_s = np.zeros(len(ants), dtype=np.float64)
    beam_vector_enu = normalize(beam_vector_enu)
    for ant_index, ant in enumerate(ants):
        if ant_index == anchor_index:
            continue
        baseline_enu_m = np.asarray(anchor_ant.baseline_toward(ant), dtype=np.float64)
        delay_series_s[ant_index] = np.dot(baseline_enu_m, beam_vector_enu) / light_speed_m_per_s
    return delay_series_s


def choose_reference_index(delay_series_s):
    """Choose the antenna with the largest delay as the per-beam phase reference."""

    return int(np.argmax(delay_series_s))


def compute_relative_delay_s(delay_series_s, ref_index):
    """Re-express all delays relative to the chosen per-beam reference antenna."""

    return delay_series_s[ref_index] - delay_series_s


def compute_beam_phase_data(
    ants,
    ant_names,
    antenna_results,
    frequencies_hz,
    runtime_config,
    internal_anchor_index=DEFAULT_INTERNAL_ANCHOR_INDEX,
    coeff_total_freq_channels=DEFAULT_COEFF_TOTAL_FREQ_CHANNELS,
):
    """Build the padded phase-coefficient cube and related delay metadata."""

    n_beams = antenna_results[internal_anchor_index]["beam_vectors_enu"].shape[0]
    signal_labels, n_active_signal_inputs = build_signal_labels(
        ant_names,
        runtime_config.total_signal_inputs,
        runtime_config.signals_per_antenna,
    )
    padded_frequency_hz = np.zeros(coeff_total_freq_channels, dtype=np.float64)
    padded_frequency_hz[: frequencies_hz.shape[0]] = frequencies_hz

    coeff_cube = np.zeros(
        (coeff_total_freq_channels, runtime_config.total_signal_inputs, n_beams),
        dtype=np.float32,
    )
    delay_series_s = np.zeros((n_beams, len(ants)), dtype=np.float64)
    relative_delay_s = np.zeros((n_beams, len(ants)), dtype=np.float64)
    ref_indices = np.zeros(n_beams, dtype=np.int32)

    canonical_beam_vectors_enu = np.asarray(
        antenna_results[internal_anchor_index]["beam_vectors_enu"],
        dtype=np.float64,
    )

    for beam_index in range(n_beams):
        beam_vector_enu = normalize(canonical_beam_vectors_enu[beam_index])
        beam_delay_series_s = compute_delay_series_to_internal_anchor_for_beam(
            ants[internal_anchor_index],
            ants,
            beam_vector_enu,
            internal_anchor_index,
            runtime_config.light_speed_m_per_s,
        )
        ref_index = choose_reference_index(beam_delay_series_s)
        beam_relative_delay_s = compute_relative_delay_s(beam_delay_series_s, ref_index)
        beam_phase = 2.0 * np.pi * frequencies_hz[:, None] * beam_relative_delay_s[None, :]

        delay_series_s[beam_index, :] = beam_delay_series_s
        relative_delay_s[beam_index, :] = beam_relative_delay_s
        ref_indices[beam_index] = ref_index

        for ant_index in range(len(ants)):
            signal_start = ant_index * runtime_config.signals_per_antenna
            signal_stop = signal_start + runtime_config.signals_per_antenna
            coeff_cube[: frequencies_hz.shape[0], signal_start:signal_stop, beam_index] = beam_phase[:, ant_index][:, None]

    reference_names = np.asarray([ant_names[index] for index in ref_indices])
    return {
        "coeff_cube": coeff_cube,
        "delay_series_s": delay_series_s,
        "relative_delay_s": relative_delay_s,
        "reference_indices": ref_indices,
        "reference_names": reference_names,
        "signal_labels": np.asarray(signal_labels),
        "active_signal_inputs": n_active_signal_inputs,
        "padded_frequency_hz": padded_frequency_hz,
        "canonical_beam_vectors_enu": canonical_beam_vectors_enu,
        "order_description": "frequency-major -> beam-major -> signal-major",
    }


# Output writers.

def write_beam_coeff_txt(path, coeff_cube):
    """Write the flattened coefficient cube to the legacy text output format."""

    coeff_stream = np.transpose(coeff_cube, (0, 2, 1)).reshape(-1).astype(np.float32)
    with io.open(path, "w", encoding="utf-8") as f:
        f.write(" ".join("{:.8e}".format(value) for value in coeff_stream))
        f.write("\n")


def write_beam_coeff_npz(path, when_utc, metrics, beam_rows, beam_phase_data, target_info):
    """Write the padded beam-coefficient cube and its metadata to an NPZ file."""

    np.savez(
        path,
        solution_time_utc=np.asarray([when_utc.strftime("%Y-%m-%d %H:%M:%S")]),
        pointing_reference_antenna=np.asarray([target_info["pointing_reference_antenna"]]),
        pointing_reference_index=np.asarray([target_info["pointing_reference_index"]], dtype=np.int32),
        input_az_deg=np.asarray([target_info["input_az_deg"]], dtype=np.float64),
        input_el_deg=np.asarray([target_info["input_el_deg"]], dtype=np.float64),
        source_ra=np.asarray([target_info["derived_ra"]]),
        source_dec=np.asarray([target_info["derived_dec"]]),
        beam_ids=np.asarray([row["beam_id"] for row in beam_rows], dtype=np.int32),
        q=np.asarray([row["q"] for row in beam_rows], dtype=np.int32),
        r=np.asarray([row["r"] for row in beam_rows], dtype=np.int32),
        spacing_deg=np.asarray([metrics["spacing_deg"]], dtype=np.float64),
        spacing_factor=np.asarray([metrics["spacing_factor"]], dtype=np.float64),
        bmax_m=np.asarray([metrics["bmax_m"]], dtype=np.float64),
        bmax_source=np.asarray([metrics["bmax_source"]]),
        projected_bmax_ant1=np.asarray([metrics["projected_bmax_ant1"]]),
        projected_bmax_ant2=np.asarray([metrics["projected_bmax_ant2"]]),
        projected_bmax_u_m=np.asarray([metrics["projected_bmax_u_m"]], dtype=np.float64),
        projected_bmax_v_m=np.asarray([metrics["projected_bmax_v_m"]], dtype=np.float64),
        projected_bmax_w_m=np.asarray([metrics["projected_bmax_w_m"]], dtype=np.float64),
        projected_bmax_total_m=np.asarray([metrics["projected_bmax_total_m"]], dtype=np.float64),
        coeff_frequency_hz_padded=beam_phase_data["padded_frequency_hz"],
        coeff=beam_phase_data["coeff_cube"],
        delay_series_s=beam_phase_data["delay_series_s"],
        relative_delay_s=beam_phase_data["relative_delay_s"],
        reference_indices=beam_phase_data["reference_indices"],
        reference_names=beam_phase_data["reference_names"],
        signal_labels=beam_phase_data["signal_labels"],
        active_signal_inputs=np.asarray([beam_phase_data["active_signal_inputs"]], dtype=np.int32),
        order_description=np.asarray([beam_phase_data["order_description"]]),
        canonical_beam_vectors_enu=beam_phase_data["canonical_beam_vectors_enu"],
    )


def write_text_report(
    path,
    when_utc,
    metrics,
    antenna_results,
    beam_phase_data,
    target_info,
    output_paths,
    runtime_config,
):
    """Write a human-readable report with geometry, vectors, and coefficient metadata."""

    lines = []
    lines.append("Beam 0..31 ENU direction vectors from fixed az/el pointing")
    lines.append("")
    lines.append("Configuration:")
    lines.append("  solution_time_utc        = {}".format(when_utc.strftime("%Y-%m-%d %H:%M:%S")))
    lines.append("  pointing_reference_ant   = {}".format(target_info["pointing_reference_antenna"]))
    lines.append("  input_az_deg             = {:.6f}".format(target_info["input_az_deg"]))
    lines.append("  input_el_deg             = {:.6f}".format(target_info["input_el_deg"]))
    lines.append("  derived_source_ra        = {}".format(target_info["derived_ra"]))
    lines.append("  derived_source_dec       = {}".format(target_info["derived_dec"]))
    lines.append("  roundtrip_ref_az_deg     = {:.6f}".format(target_info["roundtrip_az_deg"]))
    lines.append("  roundtrip_ref_el_deg     = {:.6f}".format(target_info["roundtrip_el_deg"]))
    lines.append("  spacing_factor           = {}".format(metrics["spacing_factor"]))
    lines.append("  spacing_deg              = {:.10f}".format(metrics["spacing_deg"]))
    lines.append("  projected_bmax_m         = {:.10f}".format(metrics["bmax_m"]))
    lines.append("  projected_bmax_pair      = {} <-> {}".format(metrics["projected_bmax_ant1"], metrics["projected_bmax_ant2"]))
    lines.append("  projected_bmax_uvw_m     = [{:+.6f}, {:+.6f}, {:+.6f}]".format(metrics["projected_bmax_u_m"], metrics["projected_bmax_v_m"], metrics["projected_bmax_w_m"]))
    lines.append("  projected_bmax_total_m   = {:.6f}".format(metrics["projected_bmax_total_m"]))
    lines.append("  bmax_source              = {}".format(metrics["bmax_source"]))
    lines.append("  primary_beam_check       = {}".format(metrics["primary_beam_check_status"]))
    lines.append("  allowed_pb_radius_deg    = {:.10f}".format(metrics["primary_beam_fwhm_radius_deg"]))
    lines.append("  max_offset_deg           = {:.10f}".format(metrics["max_offset_deg"]))
    lines.append("  primary_beam_result      = {}".format(metrics["primary_beam_validation_message"]))
    lines.append("  vector_frame             = local ENU unit vectors [east, north, up]")
    lines.append("  beam0_definition         = w-axis from katpoint uvw_basis(frozen target, antenna, time)")
    lines.append("  note                     = after az/el -> RA/Dec inversion, the target is frozen and later logic does not track visibility")
    lines.append("")

    for result in antenna_results:
        lines.append("Antenna: {}".format(result["antenna_name"]))
        lines.append(
            "  beam0 alt/az from frozen target = {:.6f} deg / {:.6f} deg".format(
                result["beam0_alt_deg"],
                result["beam0_az_deg"],
            )
        )
        lines.append(
            "  beam0 enu    = [{:+.12f}, {:+.12f}, {:+.12f}]".format(
                result["w_hat"][0],
                result["w_hat"][1],
                result["w_hat"][2],
            )
        )
        lines.append(
            "  u_hat        = [{:+.12f}, {:+.12f}, {:+.12f}]".format(
                result["u_hat"][0],
                result["u_hat"][1],
                result["u_hat"][2],
            )
        )
        lines.append(
            "  v_hat        = [{:+.12f}, {:+.12f}, {:+.12f}]".format(
                result["v_hat"][0],
                result["v_hat"][1],
                result["v_hat"][2],
            )
        )
        lines.append(
            "  {0:>6s} {1:>4s} {2:>4s} {3:>12s} {4:>12s} {5:>12s} {6:>12s} {7:>14s} {8:>14s} {9:>14s}".format(
                "BeamID", "q", "r", "dEast_deg", "dNorth_deg", "alt_deg", "az_deg", "east", "north", "up"
            )
        )
        for row in result["rows"]:
            enu = row["enu"]
            lines.append(
                "  {beam_id:6d} {q:4d} {r:4d} {dEast_deg:12.8f} {dNorth_deg:12.8f} {alt_deg:12.8f} {az_deg:12.8f} {east:14.10f} {north:14.10f} {up:14.10f}".format(
                    beam_id=row["beam_id"],
                    q=row["q"],
                    r=row["r"],
                    dEast_deg=row["dEast_deg"],
                    dNorth_deg=row["dNorth_deg"],
                    alt_deg=row["alt_deg"],
                    az_deg=row["az_deg"],
                    east=enu[0],
                    north=enu[1],
                    up=enu[2],
                )
            )
        lines.append("")

    lines.append("Beam coefficient generation:")
    lines.append("  output_txt             = {}".format(output_paths.beam_coeff_txt))
    lines.append("  output_npz             = {}".format(output_paths.beam_coeff_npz))
    lines.append("  active_freq_points     = {}".format(runtime_config.n_freq_channels))
    lines.append("  padded_freq_points     = {}".format(beam_phase_data["coeff_cube"].shape[0]))
    lines.append("  active_signal_inputs   = {}".format(beam_phase_data["active_signal_inputs"]))
    lines.append("  total_signal_inputs    = {}".format(runtime_config.total_signal_inputs))
    lines.append("  signals_per_antenna    = {}".format(runtime_config.signals_per_antenna))
    lines.append("  coeff_definition       = phi = 2*pi*f*(tau_ref - tau_ant)")
    lines.append("  storage_order          = {}".format(beam_phase_data["order_description"]))
    lines.append("  flattened_stream       = freq0 beam0 signal0..19, beam1 signal0..19, ..., beam31 signal0..19, then freq1, ..., freq2051")
    lines.append("")
    lines.append("  Signal labels:")
    for signal_index, label in enumerate(beam_phase_data["signal_labels"]):
        lines.append("    {:02d}: {}".format(signal_index, label))
    lines.append("")
    lines.append("  Beam reference antenna per beam:")
    lines.append("    BeamID  RefAnt   DelaySeries_s(anchor0->all)               RelativeDelay_s(ref->all)")
    for beam_index, ref_name in enumerate(beam_phase_data["reference_names"]):
        delay_text = ", ".join("{:+.6e}".format(val) for val in beam_phase_data["delay_series_s"][beam_index])
        rel_text = ", ".join("{:+.6e}".format(val) for val in beam_phase_data["relative_delay_s"][beam_index])
        lines.append("    {:6d}  {:<7s} {} | {}".format(beam_index, ref_name, delay_text, rel_text))

    with io.open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_direction_npz(path, when_utc, metrics, beam_rows, antenna_results, beam_phase_data, target_info):
    """Write all beam direction vectors, basis vectors, and metadata to an NPZ file."""

    antenna_names = np.asarray([result["antenna_name"] for result in antenna_results])
    beam_ids = np.asarray([row["beam_id"] for row in beam_rows], dtype=np.int32)
    q = np.asarray([row["q"] for row in beam_rows], dtype=np.int32)
    r = np.asarray([row["r"] for row in beam_rows], dtype=np.int32)
    d_east_deg = np.asarray([row["dEast_deg"] for row in beam_rows], dtype=np.float64)
    d_north_deg = np.asarray([row["dNorth_deg"] for row in beam_rows], dtype=np.float64)
    offset_deg = np.asarray([row["offset_deg"] for row in beam_rows], dtype=np.float64)
    position_angle_deg = np.asarray([row["position_angle_deg"] for row in beam_rows], dtype=np.float64)

    beam_vectors_enu = np.asarray([result["beam_vectors_enu"] for result in antenna_results], dtype=np.float64)
    beam_alt_deg = np.asarray([result["beam_alt_deg"] for result in antenna_results], dtype=np.float64)
    beam_az_deg = np.asarray([result["beam_az_deg"] for result in antenna_results], dtype=np.float64)
    u_hat = np.asarray([result["u_hat"] for result in antenna_results], dtype=np.float64)
    v_hat = np.asarray([result["v_hat"] for result in antenna_results], dtype=np.float64)
    w_hat = np.asarray([result["w_hat"] for result in antenna_results], dtype=np.float64)

    np.savez(
        path,
        solution_time_utc=np.asarray([when_utc.strftime("%Y-%m-%d %H:%M:%S")]),
        pointing_reference_antenna=np.asarray([target_info["pointing_reference_antenna"]]),
        pointing_reference_index=np.asarray([target_info["pointing_reference_index"]], dtype=np.int32),
        input_az_deg=np.asarray([target_info["input_az_deg"]], dtype=np.float64),
        input_el_deg=np.asarray([target_info["input_el_deg"]], dtype=np.float64),
        source_ra=np.asarray([target_info["derived_ra"]]),
        source_dec=np.asarray([target_info["derived_dec"]]),
        antenna_names=antenna_names,
        beam_ids=beam_ids,
        q=q,
        r=r,
        d_east_deg=d_east_deg,
        d_north_deg=d_north_deg,
        offset_deg=offset_deg,
        position_angle_deg=position_angle_deg,
        spacing_deg=np.asarray([metrics["spacing_deg"]], dtype=np.float64),
        spacing_factor=np.asarray([metrics["spacing_factor"]], dtype=np.float64),
        bmax_m=np.asarray([metrics["bmax_m"]], dtype=np.float64),
        bmax_source=np.asarray([metrics["bmax_source"]]),
        projected_bmax_ant1=np.asarray([metrics["projected_bmax_ant1"]]),
        projected_bmax_ant2=np.asarray([metrics["projected_bmax_ant2"]]),
        projected_bmax_u_m=np.asarray([metrics["projected_bmax_u_m"]], dtype=np.float64),
        projected_bmax_v_m=np.asarray([metrics["projected_bmax_v_m"]], dtype=np.float64),
        projected_bmax_w_m=np.asarray([metrics["projected_bmax_w_m"]], dtype=np.float64),
        projected_bmax_total_m=np.asarray([metrics["projected_bmax_total_m"]], dtype=np.float64),
        beam_vectors_enu=beam_vectors_enu,
        beam_alt_deg=beam_alt_deg,
        beam_az_deg=beam_az_deg,
        u_hat=u_hat,
        v_hat=v_hat,
        w_hat=w_hat,
        beam_coeff_reference_indices=beam_phase_data["reference_indices"],
        beam_coeff_reference_names=beam_phase_data["reference_names"],
        beam_coeff_relative_delay_s=beam_phase_data["relative_delay_s"],
        beam_coeff_delay_series_s=beam_phase_data["delay_series_s"],
        beam_coeff_signal_labels=beam_phase_data["signal_labels"],
    )


def write_beam_products(results, output_paths=None):
    """Write all beam products produced by compute_beam_products_from_fixed_azel."""

    output_paths = output_paths or results["output_paths"]
    if output_paths is None:
        raise ValueError("output_paths must be provided when writing output products")

    write_beam_coeff_txt(output_paths.beam_coeff_txt, results["beam_phase_data"]["coeff_cube"])
    write_beam_coeff_npz(
        output_paths.beam_coeff_npz,
        results["when_utc"],
        results["metrics"],
        results["beam_rows"],
        results["beam_phase_data"],
        results["target_info"],
    )
    write_text_report(
        output_paths.direction_txt,
        results["when_utc"],
        results["metrics"],
        results["antenna_results"],
        results["beam_phase_data"],
        results["target_info"],
        output_paths,
        results["runtime_config"],
    )
    write_direction_npz(
        output_paths.direction_npz,
        results["when_utc"],
        results["metrics"],
        results["beam_rows"],
        results["antenna_results"],
        results["beam_phase_data"],
        results["target_info"],
    )


# High-level pipeline entry points.

def compute_beam_products_from_fixed_azel(
    config=None,
    output_paths=None,
    runtime_config=None,
    internal_anchor_index=DEFAULT_INTERNAL_ANCHOR_INDEX,
    coeff_total_freq_channels=DEFAULT_COEFF_TOTAL_FREQ_CHANNELS,
    write_outputs=True,
):
    """Run the full fixed-az/el pipeline and optionally write all outputs."""

    if config is None:
        config = PointingConfig()
    if output_paths is None:
        output_paths = build_default_output_paths(os.path.dirname(os.path.abspath(__file__)))
    if runtime_config is None:
        raise ValueError("runtime_config must be provided")

    ants, ant_names = load_antennas(
        runtime_config.ants_txt,
        runtime_config.default_alt_m,
        runtime_config.default_diam_m,
    )
    when_utc = resolve_first_solution_time(runtime_config.first_solution_time_utc)
    target, target_info = solve_radec_from_fixed_azel(ants, ant_names, when_utc, config)
    projected_bmax = compute_projected_max_baseline_m(target, ants, ant_names, when_utc)
    metrics, beam_rows = build_beam_rows(bmax_m=projected_bmax["projected_baseline_m"])
    metrics = dict(metrics)
    metrics.update(
        {
            "bmax_source": "max pairwise projected baseline from target.uvw() at solution time",
            "projected_bmax_ant1": projected_bmax["ant1_name"],
            "projected_bmax_ant2": projected_bmax["ant2_name"],
            "projected_bmax_u_m": projected_bmax["u_m"],
            "projected_bmax_v_m": projected_bmax["v_m"],
            "projected_bmax_w_m": projected_bmax["w_m"],
            "projected_bmax_total_m": projected_bmax["total_baseline_m"],
        }
    )
    frequencies_hz = build_frequency_axis_hz(runtime_config)
    antenna_results = collect_antenna_results(target, ants, ant_names, when_utc, beam_rows)
    beam_phase_data = compute_beam_phase_data(
        ants,
        ant_names,
        antenna_results,
        frequencies_hz,
        runtime_config,
        internal_anchor_index=internal_anchor_index,
        coeff_total_freq_channels=coeff_total_freq_channels,
    )

    results = {
        "ants": ants,
        "antenna_names": ant_names,
        "when_utc": when_utc,
        "target": target,
        "target_info": target_info,
        "metrics": metrics,
        "beam_rows": beam_rows,
        "frequencies_hz": frequencies_hz,
        "antenna_results": antenna_results,
        "beam_phase_data": beam_phase_data,
        "config": config,
        "output_paths": output_paths,
        "runtime_config": runtime_config,
        "internal_anchor_index": internal_anchor_index,
        "coeff_total_freq_channels": coeff_total_freq_channels,
    }
    if write_outputs:
        write_beam_products(results, output_paths=output_paths)
    return results


def format_beam_products_summary_lines(results):
    """Build the console summary shown after a successful pipeline run."""

    lines = []
    lines.append("Solution time (UTC)     : {}".format(results["when_utc"].strftime("%Y-%m-%d %H:%M:%S")))
    lines.append(
        "Pointing az/el (deg)    : {:.6f} / {:.6f}".format(
            results["target_info"]["input_az_deg"],
            results["target_info"]["input_el_deg"],
        )
    )
    lines.append("Pointing reference ant  : {}".format(results["target_info"]["pointing_reference_antenna"]))
    lines.append(
        "Derived target RA/Dec   : {} {}".format(
            results["target_info"]["derived_ra"],
            results["target_info"]["derived_dec"],
        )
    )
    lines.append("Antenna count           : {}".format(len(results["antenna_names"])))
    lines.append("Beam count              : {}".format(len(results["beam_rows"])))
    lines.append("Projected bmax (m)      : {:.10f}".format(results["metrics"]["bmax_m"]))
    lines.append(
        "Projected bmax pair     : {} <-> {}".format(
            results["metrics"]["projected_bmax_ant1"],
            results["metrics"]["projected_bmax_ant2"],
        )
    )
    lines.append("Spacing (deg)           : {:.10f}".format(results["metrics"]["spacing_deg"]))
    lines.append("Primary-beam check      : {}".format(results["metrics"]["primary_beam_check_status"]))
    lines.append("Allowed PB radius (deg) : {:.10f}".format(results["metrics"]["primary_beam_fwhm_radius_deg"]))
    lines.append("Max beam offset (deg)   : {:.10f}".format(results["metrics"]["max_offset_deg"]))
    lines.append("Primary-beam result     : {}".format(results["metrics"]["primary_beam_validation_message"]))
    lines.append("TXT output              : {}".format(results["output_paths"].direction_txt))
    lines.append("NPZ output              : {}".format(results["output_paths"].direction_npz))
    lines.append("Beam coeff TXT          : {}".format(results["output_paths"].beam_coeff_txt))
    lines.append("Beam coeff NPZ          : {}".format(results["output_paths"].beam_coeff_npz))
    lines.append("")
    for result in results["antenna_results"]:
        lines.append(
            "{:<10s} beam0 alt {:9.3f} deg | az {:9.3f} deg | enu [{:+.6f}, {:+.6f}, {:+.6f}]".format(
                result["antenna_name"],
                result["beam0_alt_deg"],
                result["beam0_az_deg"],
                result["w_hat"][0],
                result["w_hat"][1],
                result["w_hat"][2],
            )
        )
    lines.append("")
    for beam_index, ref_name in enumerate(results["beam_phase_data"]["reference_names"][:8]):
        lines.append("Beam {:02d} reference antenna: {}".format(beam_index, ref_name))
    if results["beam_rows"]:
        lines.append("...")
        lines.append(
            "Beam {:02d} reference antenna: {}".format(
                len(results["beam_rows"]) - 1,
                results["beam_phase_data"]["reference_names"][-1],
            )
        )
    return lines


def print_beam_products_summary(results):
    """Print the short console summary for a completed fixed-az/el run."""

    for line in format_beam_products_summary_lines(results):
        print(line)


__all__ = [
    "DEFAULT_COEFF_TOTAL_FREQ_CHANNELS",
    "DEFAULT_INTERNAL_ANCHOR_INDEX",
    "OutputPaths",
    "PointingConfig",
    "RuntimeConfig",
    "angular_separation_deg",
    "build_default_output_paths",
    "build_epoch_time_context",
    "build_frequency_axis_hz",
    "build_katpoint_antenna_from_record",
    "build_katpoint_antennas_from_records",
    "build_katpoint_target_from_radec",
    "build_signal_labels",
    "ceil_utc_to_next_slot_boundary",
    "choose_reference_index",
    "collect_antenna_results",
    "compute_beam_phase_data",
    "compute_beam_products_from_fixed_azel",
    "compute_beam_vectors_for_antenna",
    "compute_delay_series_to_internal_anchor_for_beam",
    "compute_projected_max_baseline_m",
    "compute_relative_delay_s",
    "compute_source_state",
    "deg_to_dms_str",
    "enu_vector_to_altaz_deg",
    "format_beam_products_summary_lines",
    "format_dual_timestamp",
    "format_local_timestamp",
    "format_utc_timestamp",
    "get_target_horizontal_coords",
    "get_uvw_basis_enu",
    "load_antennas",
    "normalize",
    "offset_direction_vector",
    "parse_dec_to_rad",
    "parse_ra_to_rad",
    "parse_sexagesimal",
    "print_beam_products_summary",
    "rad_to_hms_string",
    "resolve_first_solution_time",
    "signed_angle_delta_rad",
    "solve_radec_from_fixed_azel",
    "trace_build_source_aligned_horizontal_frame",
    "trace_collect_antenna_beam_results",
    "trace_compute_beam_vectors_for_antenna",
    "trace_compute_local_phase_angle_rad",
    "utc_now",
    "to_katpoint_timestamp",
    "validate_pointing_inputs",
    "wrap_az_deg",
    "wrap_deg_360",
    "wrap_rad_2pi",
    "write_beam_coeff_npz",
    "write_beam_coeff_txt",
    "write_beam_products",
    "write_direction_npz",
    "write_text_report",
]
