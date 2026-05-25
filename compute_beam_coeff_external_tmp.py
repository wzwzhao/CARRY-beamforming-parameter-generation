#!/usr/bin/env python3
"""Compute Beam 0..31 direction vectors and Phi2 phase factors from a user-provided RA/Dec center.

Pipeline:
1. Load antennas
2. Resolve calculation time
3. Build the center target directly from input RA/Dec
4. Load the 32 beam offsets from an external text file
5. Build the valid frequency axis
6. Compute 32 beam directions per antenna
7. Compute beam phase coefficients
8. Convert coefficients to Phi2 float stream
9. Write hardware-order beam_coeff.txt / beam_coeff.bin and diagnostic outputs

This script is intentionally self-contained so the full workflow can be inspected
and modified in one file.
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

OUT_TXT = os.path.join(SCRIPT_DIR, "beam32_report.txt")
OUT_NPZ = os.path.join(SCRIPT_DIR, "beam32_direction_vectors.npz")
OUT_BEAM_COEFF_TXT = os.path.join(SCRIPT_DIR, "beam_coeff.txt")
OUT_BEAM_COEFF_NPZ = os.path.join(SCRIPT_DIR, "beam_coeff.npz")
OUT_BEAM_COEFF_BIN = os.path.join(SCRIPT_DIR, "beam_coeff.bin")
OUT_BEAM_LAYOUT_PNG = os.path.join(SCRIPT_DIR, "beam32_layout_from_offsets.png")
BEAM_OFFSETS_TXT = os.path.join(SCRIPT_DIR, "config_32beam_hex37_drop5_beam_offsets.txt")
EXPECTED_BEAM_COUNT = 32
HARDWARE_COEFF_ORDER = "frequency -> beam -> input"
HARDWARE_COEFF_INDEX_FORMULA = "index = ((v * 32) + j) * 20 + i"
BEAM_OFFSET_CONVENTION = "dEast/dNorth offsets from center target"
CENTER_BEAM_DESCRIPTION = "BeamID 1 / beam_index 0 / zero offset"
MISSING_INPUT_FILL_VALUE = 0.0

# User-adjustable runtime parameters.
ANTS_TXT = r"d:\总\博\imaging\uvcovplot-master\ants.txt"
FIRST_SOLUTION_TIME_UTC = None
N_FREQ_CHANNELS = 2048
FREQ_START_HZ = 1.0e9
FREQ_STOP_HZ = 1.5e9
TOTAL_SIGNAL_INPUTS = 20
SIGNALS_PER_ANTENNA = 2
DEFAULT_ALT_M = 1588.0
DEFAULT_DIAM_M = 7.5
LIGHT_SPEED_M_PER_S = 299792458.0
INTERNAL_ANCHOR_INDEX = 0
COEFF_TOTAL_FREQ_CHANNELS = 2052
WRITE_OUTPUTS = True
WRITE_LAYOUT_PLOT = True
WRITE_BEAM_COEFF_BIN = True

# Console debug switches. Keep these in this file so it is easy to inspect steps.
PRINT_PIPELINE_STEPS = True
PRINT_RUNTIME_CONFIG = True
PRINT_INTERMEDIATE_SAMPLES = True

# User target configuration.
TARGET_NAME = "TARGET"
TARGET_RA = "19:35:00.00"
TARGET_DEC = "21:54:00.00"
IGNORE_VISIBILITY = True


@dataclass(frozen=True)
class OutputPaths:
    """File paths for all direction-vector and beam-coefficient outputs."""

    direction_txt: str
    direction_npz: str
    beam_coeff_txt: str
    beam_coeff_npz: str


@dataclass(frozen=True)
class RuntimeConfig:
    """User-adjustable runtime parameters for the beam pipeline."""

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


def ensure_utc_datetime(when_utc):
    """Normalize a datetime to timezone-aware UTC."""

    if when_utc is None:
        return None
    if when_utc.tzinfo is None:
        return when_utc.replace(tzinfo=timezone.utc)
    return when_utc.astimezone(timezone.utc)


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


def normalize(vec):
    """Return a unit-length copy of the input vector."""

    vec = np.asarray(vec, dtype=np.float64)
    norm = np.linalg.norm(vec)
    if (not np.isfinite(norm)) or norm == 0.0:
        raise ValueError("Cannot normalize vector {}".format(vec))
    return vec / norm


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


def offset_direction_vector(u_hat, v_hat, w_hat, d_east_rad, d_north_rad):
    """Offset Beam 0 by small east/north angular steps and return a new ENU vector."""

    radius_rad = np.hypot(d_east_rad, d_north_rad)
    if radius_rad == 0.0:
        return w_hat.copy()

    tangent_offset = d_east_rad * u_hat + d_north_rad * v_hat
    beam_hat = np.cos(radius_rad) * w_hat + (np.sin(radius_rad) / radius_rad) * tangent_offset
    return normalize(beam_hat)


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
    internal_anchor_index=0,
    coeff_total_freq_channels=2052,
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
        "order_description": "coeff_cube axes = frequency -> input -> beam",
    }


def write_beam_coeff_npz(path, when_utc, metrics, beam_rows, beam_phase_data, target_info):
    """Write the padded beam-coefficient cube and its metadata to an NPZ file."""

    input_mode = target_info.get("input_mode", "azel")
    target_name = target_info.get("target_name", "Beam32PointingTarget")
    input_ra = target_info.get("input_ra", target_info["derived_ra"])
    input_dec = target_info.get("input_dec", target_info["derived_dec"])
    visibility_check = target_info.get("visibility_check", "not recorded")

    np.savez(
        path,
        solution_time_utc=np.asarray([when_utc.strftime("%Y-%m-%d %H:%M:%S")]),
        input_mode=np.asarray([input_mode]),
        target_name=np.asarray([target_name]),
        pointing_reference_antenna=np.asarray([target_info["pointing_reference_antenna"]]),
        pointing_reference_index=np.asarray([target_info["pointing_reference_index"]], dtype=np.int32),
        input_az_deg=np.asarray([target_info["input_az_deg"]], dtype=np.float64),
        input_el_deg=np.asarray([target_info["input_el_deg"]], dtype=np.float64),
        input_ra=np.asarray([input_ra]),
        input_dec=np.asarray([input_dec]),
        visibility_check=np.asarray([visibility_check]),
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

    input_mode = target_info.get("input_mode", "azel")
    target_name = target_info.get("target_name", "Beam32PointingTarget")
    input_ra = target_info.get("input_ra", target_info["derived_ra"])
    input_dec = target_info.get("input_dec", target_info["derived_dec"])
    visibility_check = target_info.get("visibility_check", "not recorded")

    lines = []
    if input_mode == "radec":
        lines.append("Beam 0..31 ENU direction vectors from a user-provided RA/Dec center")
    else:
        lines.append("Beam 0..31 ENU direction vectors from fixed az/el pointing")
    lines.append("")
    lines.append("Configuration:")
    lines.append("  solution_time_utc        = {}".format(when_utc.strftime("%Y-%m-%d %H:%M:%S")))
    lines.append("  target_input_mode        = {}".format(input_mode))
    lines.append("  target_name              = {}".format(target_name))
    lines.append("  pointing_reference_ant   = {}".format(target_info["pointing_reference_antenna"]))
    if input_mode == "radec":
        lines.append("  input_source_ra          = {}".format(input_ra))
        lines.append("  input_source_dec         = {}".format(input_dec))
        lines.append("  visibility_check         = {}".format(visibility_check))
    else:
        lines.append("  input_az_deg             = {:.6f}".format(target_info["input_az_deg"]))
        lines.append("  input_el_deg             = {:.6f}".format(target_info["input_el_deg"]))
    lines.append("  derived_source_ra        = {}".format(target_info["derived_ra"]))
    lines.append("  derived_source_dec       = {}".format(target_info["derived_dec"]))
    if input_mode != "radec":
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
    if input_mode == "radec":
        lines.append("  note                     = the center target is built directly from user-provided RA/Dec and visibility is not enforced")
    else:
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

    input_mode = target_info.get("input_mode", "azel")
    target_name = target_info.get("target_name", "Beam32PointingTarget")
    input_ra = target_info.get("input_ra", target_info["derived_ra"])
    input_dec = target_info.get("input_dec", target_info["derived_dec"])
    visibility_check = target_info.get("visibility_check", "not recorded")

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
        input_mode=np.asarray([input_mode]),
        target_name=np.asarray([target_name]),
        pointing_reference_antenna=np.asarray([target_info["pointing_reference_antenna"]]),
        pointing_reference_index=np.asarray([target_info["pointing_reference_index"]], dtype=np.int32),
        input_az_deg=np.asarray([target_info["input_az_deg"]], dtype=np.float64),
        input_el_deg=np.asarray([target_info["input_el_deg"]], dtype=np.float64),
        input_ra=np.asarray([input_ra]),
        input_dec=np.asarray([input_dec]),
        visibility_check=np.asarray([visibility_check]),
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


def _parse_optional_int_token(token):
    token = token.strip()
    if token.lower() in ("n/a", "na", "none", "-"):
        return None
    return int(token)


def format_layout_source_label(source_value):
    """Convert internal source tags into clearer terminal text."""

    if source_value is None:
        return "unknown"
    return str(source_value).replace("_", " ")


def load_beam_rows_from_offsets_txt(txt_path, expected_count=32):
    """Load beam offsets from the external 32-beam layout report."""

    beam_rows = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue

            parts = line.split()
            try:
                beam_id = int(parts[0])
            except (ValueError, IndexError):
                continue

            if len(parts) < 7:
                raise ValueError(
                    "Line {}: expected 7 columns: BeamID q r dEast_deg dNorth_deg offset_deg PA_deg".format(lineno)
                )

            try:
                q = _parse_optional_int_token(parts[1])
                r = _parse_optional_int_token(parts[2])
                d_east_deg = float(parts[3])
                d_north_deg = float(parts[4])
                offset_deg = float(parts[5])
                pa_deg = float(parts[6])
            except Exception as exc:
                raise ValueError(
                    "Line {}: failed to parse beam offset row: {}".format(lineno, line)
                ) from exc

            calculated_offset = math.hypot(d_east_deg, d_north_deg)
            if abs(calculated_offset - offset_deg) > 2.0e-5:
                raise ValueError(
                    "Line {}: offset_deg mismatch for BeamID {}: file {:.8f}, calculated {:.8f}".format(
                        lineno,
                        beam_id,
                        offset_deg,
                        calculated_offset,
                    )
                )

            beam_rows.append(
                {
                    "beam_id": beam_id,
                    "beam_index": beam_id - 1,
                    "q": q,
                    "r": r,
                    "dEast_deg": d_east_deg,
                    "dNorth_deg": d_north_deg,
                    "dEast_rad": math.radians(d_east_deg),
                    "dNorth_rad": math.radians(d_north_deg),
                    "offset_deg": offset_deg,
                    "PA_deg": pa_deg,
                    "position_angle_deg": pa_deg,
                    "source": "external_offset_file",
                }
            )

    if len(beam_rows) != expected_count:
        raise ValueError(
            "Expected {} beam rows from {}, got {}".format(expected_count, txt_path, len(beam_rows))
        )

    seen = set()
    for row in beam_rows:
        beam_id = row["beam_id"]
        if beam_id in seen:
            raise ValueError("Duplicate BeamID {} in {}".format(beam_id, txt_path))
        seen.add(beam_id)

    expected_ids = set(range(1, expected_count + 1))
    if seen != expected_ids:
        raise ValueError(
            "BeamID set mismatch in {}: expected 1..{}, got {}".format(
                txt_path,
                expected_count,
                sorted(seen),
            )
        )

    beam_rows = sorted(beam_rows, key=lambda row: row["beam_id"])
    first = beam_rows[0]
    if abs(first["dEast_deg"]) > 1.0e-9 or abs(first["dNorth_deg"]) > 1.0e-9:
        raise ValueError(
            "BeamID 1 should be the center beam with zero offset, got dEast={} dNorth={}".format(
                first["dEast_deg"],
                first["dNorth_deg"],
            )
        )

    return beam_rows


def build_external_layout_metrics(runtime_config, beam_rows, beam_offsets_txt):
    """Build a downstream-compatible metrics dictionary for external beam offsets."""

    max_offset_deg = max(row["offset_deg"] for row in beam_rows)
    nonzero_offsets = [row["offset_deg"] for row in beam_rows if row["offset_deg"] > 0.0]
    min_nonzero_offset_deg = min(nonzero_offsets) if nonzero_offsets else 0.0

    layout_reference_freq_hz = 0.5 * (runtime_config.freq_start_hz + runtime_config.freq_stop_hz)
    lambda_m = runtime_config.light_speed_m_per_s / layout_reference_freq_hz
    theta_pb_fwhm_rad = 1.02 * lambda_m / runtime_config.default_diam_m
    primary_beam_fwhm_radius_deg = math.degrees(theta_pb_fwhm_rad) / 2.0
    within_primary_beam = max_offset_deg <= primary_beam_fwhm_radius_deg + 1.0e-12

    if within_primary_beam:
        validation_message = (
            "PASS: all external beam centres stay within the single-dish primary-beam "
            "FWHM radius (max offset {:.8f} deg <= {:.8f} deg)."
        ).format(max_offset_deg, primary_beam_fwhm_radius_deg)
    else:
        validation_message = (
            "FAIL: external beam offsets exceed the single-dish primary-beam "
            "FWHM radius (max offset {:.8f} deg > {:.8f} deg)."
        ).format(max_offset_deg, primary_beam_fwhm_radius_deg)

    return {
        "beam_count": len(beam_rows),
        "layout": "external_offset_file",
        "beam_offsets_file": beam_offsets_txt,
        "spacing_deg": min_nonzero_offset_deg,
        "coverage_radius_deg": max_offset_deg,
        "spacing_factor": float("nan"),
        "bmax_source": "not used; beam offsets loaded from external file",
        "bmax_m": float("nan"),
        "projected_bmax_ant1": "n/a",
        "projected_bmax_ant2": "n/a",
        "projected_bmax_u_m": float("nan"),
        "projected_bmax_v_m": float("nan"),
        "projected_bmax_w_m": float("nan"),
        "projected_bmax_total_m": float("nan"),
        "primary_beam_check_status": "PASS" if within_primary_beam else "FAIL",
        "primary_beam_fwhm_radius_deg": primary_beam_fwhm_radius_deg,
        "max_offset_deg": max_offset_deg,
        "primary_beam_validation_message": validation_message,
    }


def build_runtime_config():
    """Build the runtime configuration formerly pulled from another script."""

    return RuntimeConfig(
        ants_txt=ANTS_TXT,
        first_solution_time_utc=FIRST_SOLUTION_TIME_UTC,
        n_freq_channels=N_FREQ_CHANNELS,
        freq_start_hz=FREQ_START_HZ,
        freq_stop_hz=FREQ_STOP_HZ,
        total_signal_inputs=TOTAL_SIGNAL_INPUTS,
        signals_per_antenna=SIGNALS_PER_ANTENNA,
        default_alt_m=DEFAULT_ALT_M,
        default_diam_m=DEFAULT_DIAM_M,
        light_speed_m_per_s=LIGHT_SPEED_M_PER_S,
    )


def build_target_from_radec(target_name, target_ra, target_dec):
    """Build the fixed sky target directly from user-provided RA/Dec."""

    description = "{}, radec, {}, {}".format(
        target_name,
        target_ra,
        target_dec,
    )
    target = katpoint.Target(description)
    target_info = {
        "target_name": target_name,
        "input_mode": "radec",
        "input_ra": target_ra,
        "input_dec": target_dec,
        "derived_ra": target_ra,
        "derived_dec": target_dec,
        "pointing_reference_antenna": "n/a",
        "pointing_reference_index": -1,
        "visibility_check": "ignored",
        "input_az_deg": float("nan"),
        "input_el_deg": float("nan"),
        "roundtrip_az_deg": float("nan"),
        "roundtrip_el_deg": float("nan"),
    }

    return target, target_info


def build_output_paths():
    """Build the output file locations used by the pipeline."""

    return OutputPaths(
        direction_txt=OUT_TXT,
        direction_npz=OUT_NPZ,
        beam_coeff_txt=OUT_BEAM_COEFF_TXT,
        beam_coeff_npz=OUT_BEAM_COEFF_NPZ,
    )


def normalize_phi2_cube_to_input_beam_freq(phi2_cube, n_inputs, n_beams):
    """Normalize a Phi2 cube to the common axis order [input, beam, frequency]."""

    arr = np.asarray(phi2_cube)

    if np.iscomplexobj(arr):
        arr = np.angle(arr).astype(np.float32)
    else:
        arr = arr.astype(np.float32, copy=False)

    if arr.ndim != 3:
        raise ValueError("Phi2 cube must be 3D, got shape {}".format(arr.shape))

    if arr.shape[0] == n_beams and arr.shape[1] <= n_inputs:
        arr = arr.transpose(1, 0, 2)
    elif arr.shape[0] <= n_inputs and arr.shape[1] == n_beams:
        pass
    elif arr.shape[0] <= n_inputs and arr.shape[2] == n_beams:
        arr = arr.transpose(0, 2, 1)
    elif arr.shape[0] == n_beams and arr.shape[2] <= n_inputs:
        arr = arr.transpose(2, 0, 1)
    elif arr.shape[1] <= n_inputs and arr.shape[2] == n_beams:
        arr = arr.transpose(1, 2, 0)
    elif arr.shape[1] == n_beams and arr.shape[2] <= n_inputs:
        arr = arr.transpose(2, 1, 0)
    else:
        raise ValueError(
            "Unsupported Phi2 cube shape {}. Expected axes compatible with "
            "(beam,input,freq), (input,beam,freq), (freq,input,beam), or (freq,beam,input).".format(
                arr.shape
            )
        )

    if arr.shape[0] > n_inputs or arr.shape[1] != n_beams:
        raise ValueError(
            "Unexpected normalized Phi2 shape {}, expected (<= {}, {}, n_freq)".format(
                arr.shape,
                n_inputs,
                n_beams,
            )
        )

    return arr


def pad_missing_signal_inputs(phi2_input_beam_freq, n_total_inputs):
    """Pad missing signal-input channels with zeros up to TOTAL_SIGNAL_INPUTS."""

    n_active_inputs, n_beams, n_freq = phi2_input_beam_freq.shape

    if n_active_inputs > n_total_inputs:
        raise ValueError(
            "Active signal inputs {} exceed TOTAL_SIGNAL_INPUTS {}".format(
                n_active_inputs,
                n_total_inputs,
            )
        )

    if n_active_inputs == n_total_inputs:
        return phi2_input_beam_freq, 0

    padded = np.zeros(
        (n_total_inputs, n_beams, n_freq),
        dtype=phi2_input_beam_freq.dtype,
    )
    padded[:n_active_inputs, :, :] = phi2_input_beam_freq

    n_missing_inputs = n_total_inputs - n_active_inputs
    return padded, n_missing_inputs


def format_input_index_list(indices):
    """Format a list of input indices for compact terminal output."""

    if not indices:
        return "none"
    return ", ".join(str(index) for index in indices)


def pad_phi2_freq_axis(phi2_input_beam_freq, n_total_freq):
    """Pad the frequency axis with zeros up to the hardware output length."""

    n_inputs, n_beams, n_freq = phi2_input_beam_freq.shape

    if n_freq > n_total_freq:
        raise ValueError(
            "Phi2 frequency channels {} exceed output total {}".format(n_freq, n_total_freq)
        )

    if n_freq == n_total_freq:
        return phi2_input_beam_freq

    padded = np.zeros((n_inputs, n_beams, n_total_freq), dtype=np.float32)
    padded[:, :, :n_freq] = phi2_input_beam_freq
    return padded


def prepare_phi2_hardware_stream(
    phi2_cube,
    n_inputs=20,
    n_beams=32,
    n_total_freq=2052,
    n_valid_freq=None,
    n_active_inputs=None,
):
    """Return the flattened hardware-order Phi2 stream and bookkeeping metadata."""

    phi2 = normalize_phi2_cube_to_input_beam_freq(
        phi2_cube,
        n_inputs=n_inputs,
        n_beams=n_beams,
    )

    normalized_active_inputs = phi2.shape[0]
    if n_active_inputs is None:
        n_active_inputs = normalized_active_inputs
    else:
        n_active_inputs = int(n_active_inputs)
        if n_active_inputs < 0 or n_active_inputs > n_inputs:
            raise ValueError(
                "Invalid n_active_inputs {} for TOTAL_SIGNAL_INPUTS {}".format(
                    n_active_inputs,
                    n_inputs,
                )
            )
        if normalized_active_inputs < n_active_inputs:
            raise ValueError(
                "Normalized Phi2 only has {} input channels, but n_active_inputs is {}".format(
                    normalized_active_inputs,
                    n_active_inputs,
                )
            )
        phi2 = phi2[:n_active_inputs, :, :]

    phi2, n_missing_inputs = pad_missing_signal_inputs(phi2, n_total_inputs=n_inputs)
    missing_input_indices = list(range(n_active_inputs, n_inputs))

    raw_n_freq = phi2.shape[2]
    if n_valid_freq is None:
        n_valid_freq = raw_n_freq

    if n_valid_freq < 0 or n_valid_freq > raw_n_freq:
        raise ValueError(
            "Invalid n_valid_freq {} for normalized Phi2 shape {}".format(
                n_valid_freq,
                phi2.shape,
            )
        )

    phi2 = pad_phi2_freq_axis(phi2[:, :, :n_valid_freq], n_total_freq=n_total_freq)

    if phi2.shape != (n_inputs, n_beams, n_total_freq):
        raise ValueError(
            "Unexpected normalized Phi2 shape {}, expected {}".format(
                phi2.shape,
                (n_inputs, n_beams, n_total_freq),
            )
        )

    out = phi2.transpose(2, 1, 0).reshape(-1)
    expected_count = n_total_freq * n_beams * n_inputs
    if out.size != expected_count:
        raise ValueError(
            "Unexpected output count {}, expected {}".format(out.size, expected_count)
        )

    return {
        "phi2_input_beam_freq": phi2,
        "flat_stream": out,
        "value_count": int(out.size),
        "active_signal_inputs": int(n_active_inputs),
        "missing_signal_inputs": int(n_missing_inputs),
        "missing_input_indices": missing_input_indices,
        "missing_input_fill_value": float(MISSING_INPUT_FILL_VALUE),
        "valid_frequency_channels": int(n_valid_freq),
        "padded_zero_frequency_channels": int(n_total_freq - n_valid_freq),
        "order_description": HARDWARE_COEFF_ORDER,
    }


def write_beam_coeff_txt_hardware_order(
    out_path,
    phi2_cube,
    n_inputs=20,
    n_beams=32,
    n_total_freq=2052,
    n_valid_freq=None,
    n_active_inputs=None,
):
    """Write beam_coeff.txt in hardware order: frequency -> beam -> input."""

    prepared = prepare_phi2_hardware_stream(
        phi2_cube,
        n_inputs=n_inputs,
        n_beams=n_beams,
        n_total_freq=n_total_freq,
        n_valid_freq=n_valid_freq,
        n_active_inputs=n_active_inputs,
    )
    out = prepared["flat_stream"]

    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines("{:.8e}\n".format(float(value)) for value in out)

    print("wrote {} float values to {}".format(prepared["value_count"], out_path))
    print("order: {}".format(prepared["order_description"]))
    print("active signal inputs: {}".format(prepared["active_signal_inputs"]))
    print("missing signal inputs: {}".format(prepared["missing_signal_inputs"]))
    print("missing input indices: {}".format(
        format_input_index_list(prepared["missing_input_indices"])
    ))
    print("missing input fill value: {:.1f}".format(prepared["missing_input_fill_value"]))
    print("valid frequency channels: {}".format(prepared["valid_frequency_channels"]))
    print(
        "padded zero frequency channels: {}".format(
            prepared["padded_zero_frequency_channels"]
        )
    )
    return prepared


def write_beam_coeff_bin_hardware_order(
    out_bin_path,
    phi2_cube,
    n_inputs=20,
    n_beams=32,
    n_total_freq=2052,
    n_valid_freq=None,
    n_active_inputs=None,
):
    """Write the same Phi2 stream to a compact little-endian float32 binary file."""

    prepared = prepare_phi2_hardware_stream(
        phi2_cube,
        n_inputs=n_inputs,
        n_beams=n_beams,
        n_total_freq=n_total_freq,
        n_valid_freq=n_valid_freq,
        n_active_inputs=n_active_inputs,
    )
    prepared["flat_stream"].astype("<f4").tofile(out_bin_path)

    print("wrote {} float32 values to {}".format(prepared["value_count"], out_bin_path))
    print("binary order: {}".format(prepared["order_description"]))
    print("binary valid frequency channels: {}".format(prepared["valid_frequency_channels"]))
    print(
        "binary padded zero frequency channels: {}".format(
            prepared["padded_zero_frequency_channels"]
        )
    )
    return prepared


def write_beam_layout_plot(png_path, beam_rows, metrics, beam_offsets_txt):
    """Render the externally provided 32-beam layout to a PNG image."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to write the beam layout plot: {}".format(png_path)
        ) from exc

    east_deg = [row["dEast_deg"] for row in beam_rows]
    north_deg = [row["dNorth_deg"] for row in beam_rows]
    offsets_deg = [row["offset_deg"] for row in beam_rows]

    fig, ax = plt.subplots(figsize=(8.5, 8.0), dpi=180)

    primary_beam_radius_deg = metrics.get("primary_beam_fwhm_radius_deg")
    if primary_beam_radius_deg is not None and math.isfinite(primary_beam_radius_deg):
        ax.add_patch(
            Circle(
                (0.0, 0.0),
                primary_beam_radius_deg,
                facecolor="none",
                edgecolor="#6b7280",
                linestyle="--",
                linewidth=1.5,
                zorder=1,
            )
        )

    scatter = ax.scatter(
        east_deg,
        north_deg,
        c=offsets_deg,
        cmap="viridis",
        s=150,
        edgecolors="black",
        linewidths=0.7,
        zorder=3,
    )
    ax.scatter(
        [0.0],
        [0.0],
        marker="+",
        s=200,
        linewidths=1.8,
        color="#111827",
        zorder=4,
    )

    for row in beam_rows:
        ax.text(
            row["dEast_deg"],
            row["dNorth_deg"],
            "{:02d}".format(row["beam_id"]),
            ha="center",
            va="center",
            fontsize=8.5,
            color="white" if row["offset_deg"] > 0.2 else "#111827",
            zorder=5,
        )

    limit_candidates = [abs(value) for value in east_deg + north_deg]
    if primary_beam_radius_deg is not None and math.isfinite(primary_beam_radius_deg):
        limit_candidates.append(primary_beam_radius_deg)
    max_extent_deg = max(limit_candidates) if limit_candidates else 0.5
    axis_limit_deg = max_extent_deg + max(0.05, 0.15 * max_extent_deg)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-axis_limit_deg, axis_limit_deg)
    ax.set_ylim(-axis_limit_deg, axis_limit_deg)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    ax.set_xlabel("dEast from center target (deg)")
    ax.set_ylabel("dNorth from center target (deg)")

    fig.suptitle(
        "32-beam layout from external offsets",
        y=0.985,
        fontsize=14,
    )
    fig.text(
        0.5,
        0.955,
        os.path.basename(beam_offsets_txt),
        ha="center",
        va="center",
        fontsize=10,
    )
    summary_lines = [
        "spacing {:.8f} deg | coverage radius {:.8f} deg".format(
            metrics["spacing_deg"],
            metrics["coverage_radius_deg"],
        )
    ]
    if primary_beam_radius_deg is not None and math.isfinite(primary_beam_radius_deg):
        summary_lines.append(
            "primary-beam FWHM radius {:.8f} deg | status {}".format(
                primary_beam_radius_deg,
                metrics.get("primary_beam_check_status", "unknown"),
            )
        )
    fig.text(
        0.5,
        0.928,
        " | ".join(summary_lines),
        ha="center",
        va="center",
        fontsize=9.5,
    )

    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.9)
    colorbar.set_label("offset from center (deg)")

    legend_lines = ["Markers are plotted directly from dEast_deg / dNorth_deg in the TXT file."]
    if primary_beam_radius_deg is not None and math.isfinite(primary_beam_radius_deg):
        legend_lines.append("Dashed circle: single-dish primary-beam FWHM radius.")
    ax.text(
        0.02,
        0.02,
        "\n".join(legend_lines),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, edgecolor="#d1d5db"),
    )

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(png_path)


def print_step(title):
    """Print one visible pipeline step header."""

    if PRINT_PIPELINE_STEPS:
        print("")
        print("=== {} ===".format(title))


def print_config_summary(runtime_config, output_paths):
    """Print the key user-configurable inputs before computation starts."""

    if not PRINT_RUNTIME_CONFIG:
        return

    zero_padded_freq_chans = COEFF_TOTAL_FREQ_CHANNELS - runtime_config.n_freq_channels
    hardware_coeff_count = (
        COEFF_TOTAL_FREQ_CHANNELS
        * EXPECTED_BEAM_COUNT
        * runtime_config.total_signal_inputs
    )

    print_step("Input Configuration")
    print("ants_txt                 : {}".format(runtime_config.ants_txt))
    print("first_solution_time_utc  : {}".format(runtime_config.first_solution_time_utc))
    print("freq range               : {:.3e} Hz -> {:.3e} Hz".format(
        runtime_config.freq_start_hz,
        runtime_config.freq_stop_hz,
    ))
    print("target input mode        : RA/Dec")
    print("target name              : {}".format(TARGET_NAME))
    print("target ra                : {}".format(TARGET_RA))
    print("target dec               : {}".format(TARGET_DEC))
    print("visibility check         : ignored")
    print("valid freq channels      : {}".format(runtime_config.n_freq_channels))
    print("output coeff freq chans  : {}".format(COEFF_TOTAL_FREQ_CHANNELS))
    print("zero padded freq chans   : {}".format(zero_padded_freq_chans))
    print("total signal inputs      : {}".format(runtime_config.total_signal_inputs))
    print("signals per antenna      : {}".format(runtime_config.signals_per_antenna))
    print("beam offsets txt         : {}".format(BEAM_OFFSETS_TXT))
    print("expected beam count      : {}".format(EXPECTED_BEAM_COUNT))
    print("hardware coeff order     : {}".format(HARDWARE_COEFF_ORDER))
    print("hardware coeff index     : {}".format(HARDWARE_COEFF_INDEX_FORMULA))
    print("hardware coeff count     : {}".format(hardware_coeff_count))
    print("direction txt            : {}".format(output_paths.direction_txt))
    print("direction npz            : {}".format(output_paths.direction_npz))
    print("beam coeff txt           : {}".format(output_paths.beam_coeff_txt))
    print("beam coeff npz           : {}".format(output_paths.beam_coeff_npz))
    print("beam coeff bin           : {}".format(OUT_BEAM_COEFF_BIN))
    print("beam layout png          : {}".format(OUT_BEAM_LAYOUT_PNG))


def print_intermediate_samples(results):
    """Print a few compact intermediate values that are useful when debugging."""

    if not PRINT_INTERMEDIATE_SAMPLES:
        return

    print_step("Intermediate Samples")
    print("antenna names            : {}".format(", ".join(results["antenna_names"])))
    print("solution time utc        : {}".format(results["when_utc"].strftime("%Y-%m-%d %H:%M:%S")))
    print("target input mode        : {}".format(results["target_info"].get("input_mode", "unknown")))
    print("center target ra/dec     : {} {}".format(
        results["target_info"]["derived_ra"],
        results["target_info"]["derived_dec"],
    ))
    print("visibility check         : {}".format(results["target_info"].get("visibility_check", "unknown")))
    print("beam offsets source      : {}".format(
        format_layout_source_label(results["metrics"].get("layout", "unknown"))
    ))
    print("beam offsets file        : {}".format(results["metrics"].get("beam_offsets_file", "n/a")))

    bmax_m = results["metrics"].get("bmax_m")
    if bmax_m is not None and not math.isnan(bmax_m):
        print("projected bmax m         : {:.10f}".format(bmax_m))
    else:
        print("projected bmax           : not used; external beam offsets are authoritative")

    spacing_deg = results["metrics"].get("spacing_deg")
    if spacing_deg is not None:
        print("min nonzero offset deg   : {:.10f}".format(spacing_deg))

    coverage_radius_deg = results["metrics"].get("coverage_radius_deg")
    if coverage_radius_deg is not None:
        print("coverage radius deg      : {:.10f}".format(coverage_radius_deg))
    print("frequency axis shape     : {}".format(results["frequencies_hz"].shape))
    print("coeff cube shape         : {}".format(results["beam_phase_data"]["coeff_cube"].shape))

    center_beam_row = results["beam_rows"][0]
    print("center beam offset       : BeamID 1 / beam_index 0 / dEast={:+.8f}, dNorth={:+.8f}".format(
        center_beam_row["dEast_deg"],
        center_beam_row["dNorth_deg"],
    ))
    for row in results["beam_rows"][:5]:
        print(
            "beam {:02d}: dEast={:+.8f} deg, dNorth={:+.8f} deg, offset={:.8f} deg, PA={:.6f} deg".format(
                row["beam_id"],
                row["dEast_deg"],
                row["dNorth_deg"],
                row["offset_deg"],
                row["PA_deg"],
            )
        )

    first_ant_result = results["antenna_results"][0]
    print("first antenna center beam alt/az : {:.6f} / {:.6f}".format(
        first_ant_result["beam0_alt_deg"],
        first_ant_result["beam0_az_deg"],
    ))
    print("phase reference for beam_index 0 : {}".format(
        results["beam_phase_data"]["reference_names"][0]
    ))

    info = results.get("beam_coeff_txt_info")
    if info:
        print("hardware coeff order     : {}".format(info["order_description"]))
        print("hardware coeff values    : {}".format(info["value_count"]))
        print("active signal inputs     : {}".format(info["active_signal_inputs"]))
        print("missing signal inputs    : {}".format(info["missing_signal_inputs"]))
        print("missing input indices    : {}".format(
            format_input_index_list(info["missing_input_indices"])
        ))
        print("missing input fill       : {:.1f}".format(info["missing_input_fill_value"]))
        print("valid freq channels      : {}".format(info["valid_frequency_channels"]))
        print("padded freq channels     : {}".format(info["padded_zero_frequency_channels"]))


def run_pipeline_step_by_step():
    """Run the direct-RA/Dec pipeline with each major step written out explicitly."""

    runtime_config = build_runtime_config()
    output_paths = build_output_paths()
    print_config_summary(runtime_config, output_paths)

    print_step("Step 1: Load Antennas")
    ants, ant_names = load_antennas(
        runtime_config.ants_txt,
        runtime_config.default_alt_m,
        runtime_config.default_diam_m,
    )
    print("loaded antennas          : {}".format(len(ants)))

    print_step("Step 2: Resolve Calculation Time")
    when_utc = resolve_first_solution_time(runtime_config.first_solution_time_utc)
    print("when_utc                 : {}".format(when_utc.strftime("%Y-%m-%d %H:%M:%S")))

    print_step("Step 3: Build Center Target From Input RA/Dec")
    target, target_info = build_target_from_radec(
        TARGET_NAME,
        TARGET_RA,
        TARGET_DEC,
    )
    print("target input mode        : {}".format(target_info["input_mode"]))
    print("target name              : {}".format(target_info["target_name"]))
    print("center target ra         : {}".format(target_info["input_ra"]))
    print("center target dec        : {}".format(target_info["input_dec"]))
    print("visibility check         : {}".format(target_info["visibility_check"]))

    print_step("Step 4: Load 32-Beam Offsets From External File")
    beam_rows = load_beam_rows_from_offsets_txt(
        BEAM_OFFSETS_TXT,
        expected_count=EXPECTED_BEAM_COUNT,
    )
    metrics = build_external_layout_metrics(runtime_config, beam_rows, BEAM_OFFSETS_TXT)
    print("beam offsets file        : {}".format(BEAM_OFFSETS_TXT))
    print("beam count               : {}".format(len(beam_rows)))
    print("center beam              : {}".format(CENTER_BEAM_DESCRIPTION))
    print("offset convention        : {}".format(BEAM_OFFSET_CONVENTION))
    print("layout source            : external offset file")
    print("min nonzero spacing deg  : {:.10f}".format(metrics["spacing_deg"]))
    print("coverage radius deg      : {:.10f}".format(metrics["coverage_radius_deg"]))
    print("primary beam check       : {}".format(metrics.get("primary_beam_check_status", "n/a")))

    print_step("Step 5: Build Valid Frequency Axis")
    frequencies_hz = build_frequency_axis_hz(runtime_config)
    print("valid frequency points   : {}".format(frequencies_hz.shape[0]))
    print("output coeff freq chans  : {}".format(COEFF_TOTAL_FREQ_CHANNELS))
    print("zero padded channels     : {}".format(COEFF_TOTAL_FREQ_CHANNELS - frequencies_hz.shape[0]))
    print("frequency start/end hz   : {:.3e} / {:.3e}".format(
        frequencies_hz[0],
        frequencies_hz[-1],
    ))

    print_step("Step 6: Compute 32 Beam Directions Per Antenna")
    antenna_results = collect_antenna_results(target, ants, ant_names, when_utc, beam_rows)
    print("antenna result count     : {}".format(len(antenna_results)))
    print("beam count per antenna   : {}".format(len(beam_rows)))
    print("center beam index        : 0")
    print("center beam id           : 1")

    print_step("Step 7: Compute Beam Phase Coefficients")
    beam_phase_data = compute_beam_phase_data(
        ants,
        ant_names,
        antenna_results,
        frequencies_hz,
        runtime_config,
        internal_anchor_index=INTERNAL_ANCHOR_INDEX,
        coeff_total_freq_channels=COEFF_TOTAL_FREQ_CHANNELS,
    )
    print("coeff cube shape         : {}".format(beam_phase_data["coeff_cube"].shape))
    print("coeff cube note          : axes are normalized before hardware-order writing")
    print("Phi2 output type         : float phase angle if coeff_cube is complex")

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
        "config": {
            "input_mode": "radec",
            "target_name": TARGET_NAME,
            "target_ra": TARGET_RA,
            "target_dec": TARGET_DEC,
            "visibility_check": "ignored",
            "beam_offset_convention": BEAM_OFFSET_CONVENTION,
            "hardware_coeff_order": HARDWARE_COEFF_ORDER,
        },
        "output_paths": output_paths,
        "runtime_config": runtime_config,
        "internal_anchor_index": INTERNAL_ANCHOR_INDEX,
        "coeff_total_freq_channels": COEFF_TOTAL_FREQ_CHANNELS,
    }

    print_intermediate_samples(results)

    if WRITE_OUTPUTS:
        active_signal_inputs = min(
            int(beam_phase_data.get("active_signal_inputs", runtime_config.total_signal_inputs)),
            runtime_config.total_signal_inputs,
        )
        missing_signal_inputs = runtime_config.total_signal_inputs - active_signal_inputs
        missing_input_indices = list(
            range(active_signal_inputs, runtime_config.total_signal_inputs)
        )
        expected_float_count = (
            COEFF_TOTAL_FREQ_CHANNELS
            * EXPECTED_BEAM_COUNT
            * runtime_config.total_signal_inputs
        )

        print_step("Step 8: Write Hardware-Order Phi2 Outputs")
        print("beam coeff txt order     : {}".format(HARDWARE_COEFF_ORDER))
        print("beam coeff bin order     : {}".format(HARDWARE_COEFF_ORDER))
        print("beam coeff index formula : {}".format(HARDWARE_COEFF_INDEX_FORMULA))
        print("valid freq channels      : {}".format(frequencies_hz.shape[0]))
        print("output freq channels     : {}".format(COEFF_TOTAL_FREQ_CHANNELS))
        print("expected float count     : {}".format(expected_float_count))
        print("active signal inputs     : {}".format(active_signal_inputs))
        print("missing signal inputs    : {}".format(missing_signal_inputs))
        print("missing input indices    : {}".format(
            format_input_index_list(missing_input_indices)
        ))
        print("missing input fill       : {:.1f}".format(MISSING_INPUT_FILL_VALUE))
        beam_coeff_txt_info = write_beam_coeff_txt_hardware_order(
            output_paths.beam_coeff_txt,
            beam_phase_data["coeff_cube"],
            n_inputs=runtime_config.total_signal_inputs,
            n_beams=EXPECTED_BEAM_COUNT,
            n_total_freq=COEFF_TOTAL_FREQ_CHANNELS,
            n_valid_freq=frequencies_hz.shape[0],
            n_active_inputs=beam_phase_data["active_signal_inputs"],
        )
        results["beam_coeff_txt_info"] = beam_coeff_txt_info
        if WRITE_BEAM_COEFF_BIN:
            beam_coeff_bin_info = write_beam_coeff_bin_hardware_order(
                OUT_BEAM_COEFF_BIN,
                beam_phase_data["coeff_cube"],
                n_inputs=runtime_config.total_signal_inputs,
                n_beams=EXPECTED_BEAM_COUNT,
                n_total_freq=COEFF_TOTAL_FREQ_CHANNELS,
                n_valid_freq=frequencies_hz.shape[0],
                n_active_inputs=beam_phase_data["active_signal_inputs"],
            )
            results["beam_coeff_bin_info"] = beam_coeff_bin_info
            results["beam_coeff_bin_path"] = os.path.abspath(OUT_BEAM_COEFF_BIN)

        print_step("Step 9: Write Diagnostic Reports And Plots")
        print("diagnostic beam coeff npz: {}".format(output_paths.beam_coeff_npz))
        print("direction text report    : {}".format(output_paths.direction_txt))
        print("direction npz            : {}".format(output_paths.direction_npz))
        write_beam_coeff_npz(
            output_paths.beam_coeff_npz,
            when_utc,
            metrics,
            beam_rows,
            beam_phase_data,
            target_info,
        )
        write_text_report(
            output_paths.direction_txt,
            when_utc,
            metrics,
            antenna_results,
            beam_phase_data,
            target_info,
            output_paths,
            runtime_config,
        )
        write_direction_npz(
            output_paths.direction_npz,
            when_utc,
            metrics,
            beam_rows,
            antenna_results,
            beam_phase_data,
            target_info,
        )
        if WRITE_LAYOUT_PLOT:
            layout_plot_path = write_beam_layout_plot(
                OUT_BEAM_LAYOUT_PNG,
                beam_rows,
                metrics,
                BEAM_OFFSETS_TXT,
            )
            results["layout_plot_path"] = layout_plot_path
            print("beam layout plot         : {}".format(layout_plot_path))
        print("outputs written")

    return results


def print_final_summary(results):
    """Print a concise final summary for the external beam-offset workflow."""

    expected_coeff_values = (
        results["coeff_total_freq_channels"]
        * len(results["beam_rows"])
        * results["runtime_config"].total_signal_inputs
    )

    print("")
    print("=== Final Summary ===")
    print("Solution time UTC        : {}".format(results["when_utc"].strftime("%Y-%m-%d %H:%M:%S")))
    print(
        "Center target RA/Dec     : {} {}".format(
            results["target_info"]["derived_ra"],
            results["target_info"]["derived_dec"],
        )
    )
    print("Target input mode        : RA/Dec")
    print("Visibility check         : ignored")
    print("Beam offset convention   : {}".format(BEAM_OFFSET_CONVENTION))
    print("Center beam              : {}".format(CENTER_BEAM_DESCRIPTION))
    print("Beam coeff order         : {}".format(HARDWARE_COEFF_ORDER))
    print("Beam coeff index         : {}".format(HARDWARE_COEFF_INDEX_FORMULA))
    print("Valid freq channels      : {}".format(results["frequencies_hz"].shape[0]))
    print("Output freq channels     : {}".format(results["coeff_total_freq_channels"]))
    print("Total signal inputs      : {}".format(results["runtime_config"].total_signal_inputs))
    print("Expected coeff values    : {}".format(expected_coeff_values))
    print("Beam offsets source      : {}".format(
        format_layout_source_label(results["metrics"].get("layout", "unknown"))
    ))
    print("Beam offsets file        : {}".format(results["metrics"].get("beam_offsets_file", "n/a")))
    print("Beam count               : {}".format(len(results["beam_rows"])))
    print("Spacing (deg)            : {:.10f}".format(results["metrics"]["spacing_deg"]))
    print("Coverage radius (deg)    : {:.10f}".format(results["metrics"]["coverage_radius_deg"]))
    txt_info = results.get("beam_coeff_txt_info")
    if txt_info:
        print("Actual TXT coeff values  : {}".format(txt_info["value_count"]))
        print("Active signal inputs     : {}".format(txt_info["active_signal_inputs"]))
        print("Missing signal inputs    : {}".format(txt_info["missing_signal_inputs"]))
        print("Missing input indices    : {}".format(
            format_input_index_list(txt_info["missing_input_indices"])
        ))
        print("Missing input fill       : {:.1f}".format(txt_info["missing_input_fill_value"]))
        print("TXT valid freq channels  : {}".format(txt_info["valid_frequency_channels"]))
        print("TXT padded freq channels : {}".format(txt_info["padded_zero_frequency_channels"]))
    print("Direction TXT report     : {}".format(results["output_paths"].direction_txt))
    print("Direction NPZ            : {}".format(results["output_paths"].direction_npz))
    print("Beam coeff TXT           : {}".format(results["output_paths"].beam_coeff_txt))
    print("Beam coeff NPZ           : {}".format(results["output_paths"].beam_coeff_npz))
    beam_coeff_bin_path = results.get("beam_coeff_bin_path")
    if beam_coeff_bin_path:
        print("Beam coeff BIN           : {}".format(beam_coeff_bin_path))
    layout_plot_path = results.get("layout_plot_path")
    if layout_plot_path:
        print("Beam layout PNG          : {}".format(layout_plot_path))


def main():
    """Run the direct-RA/Dec beam pipeline and print a short summary."""

    results = run_pipeline_step_by_step()
    print_final_summary(results)


if __name__ == "__main__":
    main()
