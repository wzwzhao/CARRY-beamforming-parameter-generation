#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Continuous 10-second slot publisher for 32-beam trace A, B, omega parameters.

This script is no longer a one-shot calculator. It now runs as a continuous
parameter publisher:

1. Build one tracking session aligned to a 10-second UTC grid
2. Precompute future slot packages for t_n = t_ref + n * 10 s
3. Publish each slot package slightly before its slot boundary
4. Atomically replace:
   - trace_mode_phase_coeff.txt

The startup/session flow is kept explicit in this file on purpose so it is
easy to read the steps and add temporary ``print(...)`` statements. Shared
geometry/time/beam helper functions are organized in
``beam32_from_azel_functions.py``.
"""

from __future__ import division, print_function

import io
import json
import math
import os
import queue
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import ephem
import numpy as np

from beam32_from_azel_functions import (
    angular_separation_deg,
    build_epoch_time_context,
    build_katpoint_antennas_from_records,
    build_katpoint_target_from_radec,
    ceil_utc_to_next_slot_boundary,
    compute_source_state,
    deg_to_dms_str,
    ensure_utc_datetime,
    format_dual_timestamp,
    format_local_timestamp,
    format_utc_timestamp,
    parse_dec_to_rad,
    parse_ra_to_rad,
    signed_angle_delta_rad,
    trace_build_source_aligned_horizontal_frame as build_source_aligned_horizontal_frame,
    trace_collect_antenna_beam_results as collect_antenna_beam_results,
    trace_compute_local_phase_angle_rad as compute_local_phase_angle_rad,
    utc_now,
    wrap_deg_360,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ANTS_TXT = os.path.join(SCRIPT_DIR, "ants.txt")
LEGACY_ANTS_TXT = r"d:\总\博\imaging\uvcovplot-master\ants.txt"


# =========================
# User-editable configuration
# =========================
# 1) Input files
ANTS_TXT = DEFAULT_ANTS_TXT if os.path.exists(DEFAULT_ANTS_TXT) else LEGACY_ANTS_TXT

# 2) Source and beam-model parameters
TARGET_RA = "19:35:00.00"
TARGET_DEC = "21:54:00.00"
MIN_ELEVATION_DEG = 32.0
SIMULATION_IGNORE_VISIBILITY = True
CENTER_FREQ_HZ = 1.25e9
OMEGA_DELTA_SECONDS = 1.0

# 3) Slot-based publishing schedule
UPDATE_PERIOD_SECONDS = 10
PUBLISH_LEAD_SECONDS = 2.0
PRECOMPUTE_QUEUE_SLOTS = 4

# 4) Required 32-beam offset inputs
BEAM_OFFSET_TABLE_FILE = os.path.join(SCRIPT_DIR, "config_32beam_hex37_drop5_beam_offsets.txt")

# 5) Terminal output:
# - "summary": one short slot summary plus Beam 0 / active channels
# - "full": all beams for the active channels
# - "quiet": slot log only
TERMINAL_PRINT_MODE = "summary"

# 6) Debug / inspection switches
# Keep None for normal continuous running.
MAX_PUBLISH_SLOTS = None
PRINT_PIPELINE_STEPS = True
PRINT_STARTUP_CONFIG = True
PRINT_STARTUP_SAMPLES = True


# =========================
# Fixed hardware / publish contract
# =========================
SIGNALS_PER_ANTENNA = 2
TOTAL_SIGNAL_INPUTS = 20
TOTAL_BEAMS = 32
TRACE_STREAM_FLOAT_COUNT = 3 * TOTAL_SIGNAL_INPUTS * TOTAL_BEAMS
TRACE_STREAM_BYTES = TRACE_STREAM_FLOAT_COUNT * 4
LIGHT_SPEED_M_PER_S = 299792458.0
WGS84_A_M = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
ANGLE_EPS = 1.0e-12
QUEUE_TIMEOUT_SECONDS = 0.25


# =========================
# Fixed output paths
# =========================
TRACE_MODE_PHASE_TXT = os.path.join(SCRIPT_DIR, "trace_mode_phase_coeff.txt")
BEAM_OFFSET_REPORT_TXT = os.path.join(SCRIPT_DIR, "trace_32beam_beam_offsets.txt")
BEAM_DIRECTION_REPORT_TXT = os.path.join(SCRIPT_DIR, "trace_32beam_beam_directions.txt")
LOG_FILE = os.path.join(SCRIPT_DIR, "trace_32beam_phase_coeff.log")


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
class SessionContext:
    """One continuous tracking session aligned to a 10-second UTC grid."""

    session_id: str
    program_start_local: datetime
    program_start_utc: datetime
    t_ref_utc: datetime
    update_period_seconds: int
    publish_lead_seconds: float


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


@dataclass
class SlotPackage:
    """Everything needed to publish one 10-second slot."""

    slot_index: int
    epoch_time_utc: datetime
    julian_date: float
    lst_hms: str
    main_az_deg: float
    main_el_deg: float
    active_signal_inputs: int
    visible_now: bool
    simulation_override: bool
    should_publish: bool
    compute_start_time_utc: datetime
    compute_finish_time_utc: datetime
    a_mat: np.ndarray
    b_mat: np.ndarray
    w_mat: np.ndarray
    beam_frames: list
    antenna_beam_results: list
    visibility_window: VisibilityWindow
    beam_offset_source: str
    beam_definition_file: Optional[str]
    beam_definition_module: Optional[str]

# Shared geometry/time helpers are imported from beam32_from_azel_functions.py
# so the startup flow in this entry script stays easier to read.


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


def build_live_status_snapshot(
    ants_txt=ANTS_TXT,
    target_ra=TARGET_RA,
    target_dec=TARGET_DEC,
    min_elevation_deg=MIN_ELEVATION_DEG,
    when_utc=None,
    antennas=None,
):
    """Compute a reusable real-time source-status snapshot for terminal or GUI use."""

    epoch_time_utc = ensure_utc_datetime(when_utc or utc_now())
    if antennas is None:
        antennas = read_antenna_file(ants_txt)
    ra_rad = parse_ra_to_rad(target_ra)
    dec_rad = parse_dec_to_rad(target_dec)
    time_context = build_epoch_time_context(epoch_time_utc)
    source_state = compute_source_state(
        antennas[0],
        epoch_time_utc,
        ra_rad,
        dec_rad,
        gmst_rad=time_context["gmst_rad"],
    )
    visibility_window = compute_visibility_window(
        antennas[0],
        epoch_time_utc,
        target_ra,
        target_dec,
        min_elevation_deg,
    )
    return {
        "epoch_time_utc": epoch_time_utc,
        "epoch_time_local": epoch_time_utc.astimezone(),
        "reference_antenna": antennas[0].name,
        "target_ra": target_ra,
        "target_dec": target_dec,
        "main_az_deg": float(source_state["azimuth_deg"]),
        "main_el_deg": float(source_state["elevation_deg"]),
        "lst_hms": source_state["lst_hms"],
        "visible_now": bool(visibility_window.visible_now),
        "visibility_window": visibility_window,
    }


def atomic_replace(src_path, dst_path):
    """Atomically replace a file on both Windows and POSIX."""

    if hasattr(os, "replace"):
        os.replace(src_path, dst_path)
        return

    if os.name == "nt" and os.path.exists(dst_path):
        os.remove(dst_path)
    os.rename(src_path, dst_path)


def write_float32_le_temp_file(final_path, values):
    """Write little-endian float32 bytes to a temporary file and fsync it."""

    tmp_path = final_path + ".tmp"
    array = np.asarray(values, dtype="<f4")
    with open(tmp_path, "wb") as f:
        f.write(array.tobytes(order="C"))
        f.flush()
        os.fsync(f.fileno())
    return tmp_path


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
        ).format(
            allowed_radius_deg,
            max_offset_deg,
            offending_text,
        )

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


def format_beam_offset_report(session, beam_offset_rows, beam_offset_meta, beam_state_tref):
    """Build a human-readable offset-table report with one-line records."""

    metrics = beam_offset_meta.get("layout_metrics")
    primary_beam_validation = beam_offset_meta.get("primary_beam_validation")
    lines = [
        "32-beam offset table report",
        "",
        "SessionSummary | t_ref_utc={} | source_mode={} | source={} | beam_definition_file={} | beam_definition_module={}".format(
            format_dual_timestamp(session.t_ref_utc),
            beam_offset_meta.get("source_mode", "unknown"),
            beam_offset_meta["source"],
            beam_offset_meta.get("beam_definition_file"),
            beam_offset_meta.get("beam_definition_module"),
        ),
        "",
        (
            "Note | this file is plain UTF-8 text | one beam/antenna record per line | "
            "it records both the fixed 32-beam offset table and the per-antenna beam pointings used at t_ref"
        ),
        "",
    ]

    if metrics is not None:
        lines.extend(
            [
                "GeometrySummary | f0_hz={:.6f} | dish_diameter_m={:.6f} | projected_bmax_m={:.6f} | projected_bmax_pair={} <-> {} | projected_bmax_uvw_m=[{:+.6f},{:+.6f},{:+.6f}] | projected_bmax_total_m={:.6f} | bmax_source={} | spacing_factor={:.6f} | spacing_deg={:.10f} | theta_syn_fwhm_deg={:.10f} | theta_pb_fwhm_deg={:.10f}".format(
                    metrics["f0_hz"],
                    metrics["dish_diameter_m"],
                    metrics["bmax_m"],
                    metrics["projected_bmax_ant1"],
                    metrics["projected_bmax_ant2"],
                    metrics["projected_bmax_u_m"],
                    metrics["projected_bmax_v_m"],
                    metrics["projected_bmax_w_m"],
                    metrics["projected_bmax_total_m"],
                    metrics["bmax_source"],
                    metrics["spacing_factor"],
                    metrics["spacing_deg"],
                    metrics["theta_syn_fwhm_deg"],
                    metrics["theta_pb_fwhm_deg"],
                ),
                "",
            ]
        )
    if primary_beam_validation is not None:
        lines.extend(
            [
                "PrimaryBeamCheck | status={} | allowed_radius_deg={:.10f} | max_offset_deg={:.10f} | result={}".format(
                    primary_beam_validation["primary_beam_check_status"],
                    primary_beam_validation["primary_beam_fwhm_radius_deg"],
                    primary_beam_validation["max_offset_deg"],
                    primary_beam_validation["primary_beam_validation_message"],
                ),
                "",
            ]
        )

    lines.extend(
        [
            "FixedOffsetRows | one row = one beam | file_beam_id is the ID in the external file | beam_index is the internal 0-based index",
        ]
    )

    for row in beam_offset_rows:
        file_beam_id = row.get("file_beam_id", row["beam_id"])
        beam_index = row.get("beam_index", row["beam_id"])
        lines.append(
            "BeamOffsetRow | file_beam_id={file_id:02d} | beam_index={beam_index:02d} | q={q:+d} | r={r:+d} | dEast_deg={de:+.8f} | dNorth_deg={dn:+.8f} | offset_deg={off:.8f} | PA_deg={pa:.6f}".format(
                file_id=file_beam_id,
                beam_index=beam_index,
                q=row["q"],
                r=row["r"],
                de=row["dEast_deg"],
                dn=row["dNorth_deg"],
                off=row["offset_deg"],
                pa=row["position_angle_deg"],
            )
        )

    lines.extend(
        [
            "",
            "CurrentAntennaBeamRowsAt_t_ref | one row = one antenna x one beam | AntennaBeamRow | antenna=... | beam_index=... | fixed_offset_deg=... | sep_deg=... | az_deg=... | el_deg=...",
        ]
    )

    for antenna_result in beam_state_tref["antenna_beam_results"]:
        antenna_name = antenna_result["antenna_name"]
        for row in antenna_result["rows"]:
            file_beam_id = row.get("file_beam_id", row["beam_id"])
            beam_index = row.get("beam_index", row["beam_id"])
            lines.append(
                (
                    "AntennaBeamRow | antenna={antenna} | file_beam_id={file_id:02d} | beam_index={beam_index:02d} | q={q:+d} | r={r:+d} | "
                    "dEast_deg={de:+.8f} | dNorth_deg={dn:+.8f} | fixed_offset_deg={off:.8f} | "
                    "PA_deg={pa:.6f} | sep_deg={sep:.8f} | az_deg={az:.8f} | el_deg={el:.8f} | "
                    "east={east:+.10f} | north={north:+.10f} | up={up:+.10f}"
                ).format(
                    antenna=antenna_name,
                    file_id=file_beam_id,
                    beam_index=beam_index,
                    q=row["q"],
                    r=row["r"],
                    de=row["dEast_deg"],
                    dn=row["dNorth_deg"],
                    off=row["offset_deg"],
                    pa=row["position_angle_deg"],
                    sep=row["separation_from_beam0_deg"],
                    az=row["az_deg"],
                    el=row["el_deg"],
                    east=float(row["enu"][0]),
                    north=float(row["enu"][1]),
                    up=float(row["enu"][2]),
                )
            )

    return "\n".join(lines)


def write_beam_offset_report(session, antennas, beam_offset_rows, beam_offset_meta):
    """Write the session beam-offset table plus t_ref beam pointings as a plain-text report."""

    ra_rad = parse_ra_to_rad(TARGET_RA)
    dec_rad = parse_dec_to_rad(TARGET_DEC)
    katpoint_antennas, antenna_names = build_katpoint_antennas_from_records(antennas)
    katpoint_target = build_katpoint_target_from_radec(TARGET_RA, TARGET_DEC)
    beam_state_tref = compute_beam_model_state(
        antennas[0],
        katpoint_target,
        katpoint_antennas,
        antenna_names,
        session.t_ref_utc,
        ra_rad,
        dec_rad,
        beam_offset_rows,
    )

    report_tmp = write_text_temp_file(
        BEAM_OFFSET_REPORT_TXT,
        format_beam_offset_report(session, beam_offset_rows, beam_offset_meta, beam_state_tref),
    )
    atomic_replace(report_tmp, BEAM_OFFSET_REPORT_TXT)


def format_beam_direction_report(package):
    """Build one human-readable per-slot beam-direction report."""

    lines = [
        "32-beam per-slot direction report",
        "",
        "Slot summary:",
        "  slot_index              = {}".format(package.slot_index),
        "  epoch_time_utc          = {}".format(package.epoch_time_utc.strftime("%Y-%m-%d %H:%M:%S")),
        "  visible_now             = {}".format(package.visible_now),
        "  simulation_override     = {}".format(package.simulation_override),
        "  should_publish          = {}".format(package.should_publish),
        "  target_ra               = {}".format(TARGET_RA),
        "  target_dec              = {}".format(TARGET_DEC),
        "  beam_offset_source      = {}".format(package.beam_offset_source),
        "  beam_definition_file    = {}".format(package.beam_definition_file),
        "  beam_definition_module  = {}".format(package.beam_definition_module),
        "  report_definition       = per-antenna 32-beam directions generated from katpoint uvw_basis at this slot time",
        "",
    ]

    for antenna_result in package.antenna_beam_results:
        lines.extend(
            [
                "Antenna: {}".format(antenna_result["antenna_name"]),
                "  beam0 az/el            = {:.8f} deg / {:.8f} deg".format(
                    antenna_result["beam0_az_deg"],
                    antenna_result["beam0_el_deg"],
                ),
                "  beam0 definition       = w-axis from uvw_basis(frozen target, antenna, slot time)",
                (
                    "  u_hat                 = [{:+.10f}, {:+.10f}, {:+.10f}]".format(
                        float(antenna_result["u_hat"][0]),
                        float(antenna_result["u_hat"][1]),
                        float(antenna_result["u_hat"][2]),
                    )
                ),
                (
                    "  v_hat                 = [{:+.10f}, {:+.10f}, {:+.10f}]".format(
                        float(antenna_result["v_hat"][0]),
                        float(antenna_result["v_hat"][1]),
                        float(antenna_result["v_hat"][2]),
                    )
                ),
                (
                    "  w_hat                 = [{:+.10f}, {:+.10f}, {:+.10f}]".format(
                        float(antenna_result["w_hat"][0]),
                        float(antenna_result["w_hat"][1]),
                        float(antenna_result["w_hat"][2]),
                    )
                ),
                (
                    "  {:>6s} {:>4s} {:>4s} {:>12s} {:>12s} {:>12s} {:>12s} {:>12s} {:>12s} {:>12s} {:>12s} {:>12s}".format(
                        "BeamIdx",
                        "q",
                        "r",
                        "dEast_deg",
                        "dNorth_deg",
                        "offset_deg",
                        "sep_deg",
                        "az_deg",
                        "el_deg",
                        "east",
                        "north",
                        "up",
                    )
                ),
            ]
        )
        for row in antenna_result["rows"]:
            lines.append(
                (
                    "  {beam_id:6d} {q:4d} {r:4d} {de:12.8f} {dn:12.8f} {off:12.8f} "
                    "{sep:12.8f} {az:12.8f} {el:12.8f} {east:+12.8f} {north:+12.8f} {up:+12.8f}"
                ).format(
                    beam_id=row["beam_id"],
                    q=row["q"],
                    r=row["r"],
                    de=row["dEast_deg"],
                    dn=row["dNorth_deg"],
                    off=row["offset_deg"],
                    sep=row["separation_from_beam0_deg"],
                    az=row["az_deg"],
                    el=row["el_deg"],
                    east=float(row["enu"][0]),
                    north=float(row["enu"][1]),
                    up=float(row["enu"][2]),
                )
            )
        lines.append("")

    return "\n".join(lines)


def write_beam_direction_report(package):
    """Write the current per-slot beam-direction report via atomic replace."""

    report_tmp = write_text_temp_file(BEAM_DIRECTION_REPORT_TXT, format_beam_direction_report(package))
    atomic_replace(report_tmp, BEAM_DIRECTION_REPORT_TXT)


def publish_slot_files(package):
    """Write trace_mode_phase_coeff.txt via temp file, then atomically replace it."""

    expected_shape = (TOTAL_SIGNAL_INPUTS, TOTAL_BEAMS)
    for name, mat in [("A", package.a_mat), ("B", package.b_mat), ("omega", package.w_mat)]:
        shape = np.asarray(mat).shape
        if shape != expected_shape:
            raise ValueError(
                "{} matrix shape mismatch: got {}, expected {}".format(
                    name,
                    shape,
                    expected_shape,
                )
            )

    trace_stream = np.concatenate(
        [
            package.a_mat.reshape(-1),
            package.b_mat.reshape(-1),
            package.w_mat.reshape(-1),
        ]
    ).astype(np.float32, copy=False)

    if trace_stream.size != TRACE_STREAM_FLOAT_COUNT:
        raise ValueError(
            "trace stream value count mismatch: got {}, expected {}".format(
                trace_stream.size,
                TRACE_STREAM_FLOAT_COUNT,
            )
        )

    trace_mode_tmp = write_float32_le_temp_file(TRACE_MODE_PHASE_TXT, trace_stream)
    atomic_replace(trace_mode_tmp, TRACE_MODE_PHASE_TXT)


def append_log_record(record):
    """Append one structured JSON log line and fsync the log file."""

    line = json.dumps(record, sort_keys=True, ensure_ascii=True)
    with io.open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())

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
    """Read antenna rows from the configured text file.

    The number of antennas is determined by the file contents.
    Each antenna maps to SIGNALS_PER_ANTENNA hardware inputs.
    The total active input count must not exceed TOTAL_SIGNAL_INPUTS.
    """

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

    return rows, {
        "source": "BEAM_OFFSET_TABLE_FILE: {}".format(path),
        "source_mode": "file",
        "beam_definition_file": path,
        "beam_definition_module": None,
    }


def resolve_beam_offset_table(session, antennas):
    """Load the required external 32-beam offset table."""

    if not BEAM_OFFSET_TABLE_FILE:
        raise ValueError(
            "BEAM_OFFSET_TABLE_FILE must be set; automatic beam offset generation is disabled."
        )

    rows, meta = load_beam_offset_table_from_file(BEAM_OFFSET_TABLE_FILE)

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
    expected_ids = list(range(TOTAL_BEAMS))
    actual_ids = [row["beam_id"] for row in rows]
    if actual_ids != expected_ids:
        raise ValueError("Beam offset table beam_id values must be 0..31 in order, got {}".format(actual_ids))

    for row in rows:
        row["beam_index"] = row["beam_id"]

    primary_beam_validation = build_primary_beam_validation_for_rows(
        rows,
        dish_diameter_m=float(antennas[0].diameter_m),
        f0_hz=CENTER_FREQ_HZ,
    )
    meta = dict(meta)
    if "layout_metrics" in meta:
        layout_metrics = dict(meta["layout_metrics"])
        layout_metrics.update(primary_beam_validation)
        meta["layout_metrics"] = layout_metrics
    meta["primary_beam_validation"] = primary_beam_validation

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
    """Compute current main-beam and 32-beam directions for one logical slot epoch."""

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
    beam_dirs_enu = np.asarray(reference_result["beam_vectors_enu"], dtype=np.float64)
    beam_el_rad = np.asarray(reference_result["beam_el_rad"], dtype=np.float64)
    beam_el_deg = np.asarray(reference_result["beam_el_deg"], dtype=np.float64)
    beam_az_deg = np.asarray(reference_result["beam_az_deg"], dtype=np.float64)
    return {
        "time_context": time_context,
        "source_state": source_state,
        "beam_dirs_enu": beam_dirs_enu,
        "beam_el_rad": beam_el_rad,
        "beam_el_deg": beam_el_deg,
        "beam_az_deg": beam_az_deg,
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
    """Compute per-antenna A[N_ant, 32], B[N_ant, 32] and omega[N_ant, 32] for one slot."""

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

    beam_frames = []
    for ant_index, ant in enumerate(antennas):
        antenna_name = ant.name
        antenna_frames = []
        antenna_state_tn = state_tn["antenna_beam_results"][ant_index]
        antenna_state_minus = state_minus["antenna_beam_results"][ant_index]
        antenna_state_plus = state_plus["antenna_beam_results"][ant_index]
        baseline_enu_m = np.asarray(ant.enu_m, dtype=np.float64)

        for beam_index in range(TOTAL_BEAMS):
            beam_dir_tn = antenna_state_tn["beam_vectors_enu"][beam_index]
            theta_rad = float(antenna_state_tn["beam_el_rad"][beam_index])
            cos_theta = float(math.cos(theta_rad))
            x_hat, y_hat, z_hat = build_source_aligned_horizontal_frame(beam_dir_tn)
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
            omega_rad_s = signed_angle_delta_rad(phi_plus, phi_minus) / (2.0 * OMEGA_DELTA_SECONDS)
            omega_ant[ant_index, beam_index] = np.float32(omega_rad_s)

            bx_m = float(np.dot(baseline_enu_m, x_hat))
            by_m = float(np.dot(baseline_enu_m, y_hat))
            a_ant[ant_index, beam_index] = np.float32(ab_scale * bx_m * cos_theta)
            b_ant[ant_index, beam_index] = np.float32(ab_scale * by_m * cos_theta)

            antenna_frames.append(
                {
                    "antenna_name": antenna_name,
                    "antenna_index": ant_index,
                    "beam_id": beam_index,
                    "theta_deg": float(np.rad2deg(theta_rad)),
                    "az_deg": float(antenna_state_tn["beam_az_deg"][beam_index]),
                    "el_deg": float(antenna_state_tn["beam_el_deg"][beam_index]),
                    "x_hat": x_hat,
                    "y_hat": y_hat,
                    "z_hat": z_hat,
                    "bx_m": bx_m,
                    "by_m": by_m,
                    "a_value": float(a_ant[ant_index, beam_index]),
                    "b_value": float(b_ant[ant_index, beam_index]),
                    "omega_value": float(omega_ant[ant_index, beam_index]),
                }
            )

        beam_frames.append(
            {
                "antenna_name": antenna_name,
                "antenna_index": ant_index,
                "frames": antenna_frames,
            }
        )
    return {
        "state_tn": state_tn,
        "state_minus": state_minus,
        "state_plus": state_plus,
        "a_ant": a_ant,
        "b_ant": b_ant,
        "omega_ant": omega_ant,
        "beam_frames": beam_frames,
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
        row_start = ant_index * SIGNALS_PER_ANTENNA
        row_stop = row_start + SIGNALS_PER_ANTENNA
        out[row_start:row_stop, :] = ant_beam_matrix[ant_index, :]
    return out


def print_session_summary(session, antennas, beam_offset_meta):
    """Print the fixed session configuration once at startup."""

    primary_beam_validation = beam_offset_meta.get("primary_beam_validation")
    active_signal_inputs = get_active_signal_inputs(len(antennas))
    padded_signal_inputs = TOTAL_SIGNAL_INPUTS - active_signal_inputs

    print("Session started")
    print("  session_id              : {}".format(session.session_id))
    print("  program_start_local     : {}".format(format_local_timestamp(session.program_start_local)))
    print("  program_start_utc       : {}".format(format_utc_timestamp(session.program_start_utc)))
    print("  t_ref_utc               : {}".format(format_utc_timestamp(session.t_ref_utc)))
    print("  update_period_seconds   : {}".format(session.update_period_seconds))
    print("  publish_lead_seconds    : {:.3f}".format(session.publish_lead_seconds))
    print("  target_ra               : {}".format(TARGET_RA))
    print("  target_dec              : {}".format(TARGET_DEC))
    print("  min_elevation_deg       : {:.6f}".format(MIN_ELEVATION_DEG))
    print("  simulation_ignore_vis   : {}".format(SIMULATION_IGNORE_VISIBILITY))
    print("  center_freq_hz          : {:.6f}".format(CENTER_FREQ_HZ))
    print("  omega_delta_seconds     : {:.6f}".format(OMEGA_DELTA_SECONDS))
    print("  antenna_count           : {}".format(len(antennas)))
    print("  active_signal_inputs    : {}".format(active_signal_inputs))
    print("  total_signal_inputs     : {}".format(TOTAL_SIGNAL_INPUTS))
    print("  padded_signal_inputs    : {}".format(padded_signal_inputs))
    if padded_signal_inputs > 0:
        print("  padded input indices    : {}..{}".format(active_signal_inputs, TOTAL_SIGNAL_INPUTS - 1))
        print("  padded input ids        : {}..{}".format(active_signal_inputs + 1, TOTAL_SIGNAL_INPUTS))
    else:
        print("  padded input indices    : none")
        print("  padded input ids        : none")
    print("  beam_count              : {}".format(TOTAL_BEAMS))
    print("  beam_offset_source      : {}".format(beam_offset_meta["source"]))
    if "spacing_deg" in beam_offset_meta:
        print("  default_spacing_deg     : {:.8f}".format(beam_offset_meta["spacing_deg"]))
    if "bmax_m" in beam_offset_meta:
        print("  default_bmax_m          : {:.6f}".format(beam_offset_meta["bmax_m"]))
    if primary_beam_validation is not None:
        print("  primary_beam_check      : {}".format(primary_beam_validation["primary_beam_check_status"]))
        print("  allowed_pb_radius_deg   : {:.8f}".format(primary_beam_validation["primary_beam_fwhm_radius_deg"]))
        print("  max_beam_offset_deg     : {:.8f}".format(primary_beam_validation["max_offset_deg"]))
    print("  trace_mode_phase_file   : {}".format(TRACE_MODE_PHASE_TXT))
    print("  trace file layout       : A[20,32] + B[20,32] + omega[20,32]")
    print("  trace file dtype        : little-endian float32")
    print("  trace file value count  : {}".format(TRACE_STREAM_FLOAT_COUNT))
    print("  trace file bytes        : {}".format(TRACE_STREAM_BYTES))
    print("  A/B frequency factor    : not included")
    print("  GPU formula             : Phi3 = nu * (A*cos(delta) + B*sin(delta) - A)")
    print("  delta definition        : delta = omega * (t - t0)")
    print("  t0 storage              : not written into trace_mode_phase_coeff.txt")
    print("  beam_offset_report      : {}".format(BEAM_OFFSET_REPORT_TXT))
    print("  beam_direction_report   : {}".format(BEAM_DIRECTION_REPORT_TXT))
    print("  log_file                : {}".format(LOG_FILE))
    if primary_beam_validation is not None:
        print("")
        print("Single-dish primary-beam FWHM check")
        print("  status                  : {}".format(primary_beam_validation["primary_beam_check_status"]))
        print("  allowed_radius_deg      : {:.8f}".format(primary_beam_validation["primary_beam_fwhm_radius_deg"]))
        print("  max_offset_deg          : {:.8f}".format(primary_beam_validation["max_offset_deg"]))
        print("  result                  : {}".format(primary_beam_validation["primary_beam_validation_message"]))
    print("")
    print("Antenna rows read from file for the first parameter calculation")
    print("  antenna_file            : {}".format(ANTS_TXT))
    print("  reference_antenna       : {} (first row)".format(antennas[0].name))
    for ant in antennas:
        print(
            "  {name:<10s} lat {lat:11.6f} deg | lon {lon:11.6f} deg | "
            "height {height:8.3f} m | diameter {diameter:5.2f} m".format(
                name=ant.name,
                lat=ant.lat_deg,
                lon=ant.lon_deg,
                height=ant.height_m,
                diameter=ant.diameter_m,
            )
        )
    print("")
    print("Fixed antenna geometry")
    for ant in antennas:
        print(
            "  {name:<10s} lat {lat:11.6f} deg | lon {lon:11.6f} deg | "
            "h {height:8.3f} m | d {diameter:5.2f} m | "
            "enu [{east:+10.4f}, {north:+10.4f}, {up:+10.4f}] m".format(
                name=ant.name,
                lat=ant.lat_deg,
                lon=ant.lon_deg,
                height=ant.height_m,
                diameter=ant.diameter_m,
                east=float(ant.enu_m[0]),
                north=float(ant.enu_m[1]),
                up=float(ant.enu_m[2]),
            )
        )
    print("")


def print_live_status_snapshot(snapshot, title):
    """Print one human-readable source-status snapshot."""

    print(title)
    print("  epoch_utc              : {}".format(format_dual_timestamp(snapshot["epoch_time_utc"])))
    print("  reference_antenna      : {}".format(snapshot["reference_antenna"]))
    print("  target_ra              : {}".format(snapshot["target_ra"]))
    print("  target_dec             : {}".format(snapshot["target_dec"]))
    print("  main_az_deg            : {:.6f}".format(snapshot["main_az_deg"]))
    print("  main_el_deg            : {:.6f}".format(snapshot["main_el_deg"]))
    print("  lst_hms                : {}".format(snapshot["lst_hms"]))
    for line in build_visibility_status_lines(snapshot["visibility_window"]):
        print(line)
    print("")


def print_beam_offset_input_summary(session, antennas, beam_offset_rows, beam_offset_meta):
    """Print where the beam offsets came from and how Beam 0 / Beam 1 look at t_ref."""

    primary_beam_validation = beam_offset_meta.get("primary_beam_validation")
    ra_rad = parse_ra_to_rad(TARGET_RA)
    dec_rad = parse_dec_to_rad(TARGET_DEC)
    katpoint_antennas, antenna_names = build_katpoint_antennas_from_records(antennas)
    katpoint_target = build_katpoint_target_from_radec(TARGET_RA, TARGET_DEC)
    beam_state = compute_beam_model_state(
        antennas[0],
        katpoint_target,
        katpoint_antennas,
        antenna_names,
        session.t_ref_utc,
        ra_rad,
        dec_rad,
        beam_offset_rows,
    )
    beam0_row = beam_offset_rows[0]
    beam1_row = beam_offset_rows[1]
    beam0_dir = beam_state["beam_dirs_enu"][0]
    beam1_dir = beam_state["beam_dirs_enu"][1]
    beam01_sep_deg = angular_separation_deg(beam0_dir, beam1_dir)

    print("Beam offset input check at t_ref")
    print("  slot_time_utc           : {}".format(format_dual_timestamp(session.t_ref_utc)))
    print("  beam_offset_source      : {}".format(beam_offset_meta["source"]))
    print("  beam_offset_report_txt  : {}".format(BEAM_OFFSET_REPORT_TXT))
    if beam_offset_meta.get("beam_definition_file"):
        print("  beam_definition_file    : {}".format(beam_offset_meta["beam_definition_file"]))
    else:
        print("  beam_definition_file    : None")
    if beam_offset_meta.get("beam_definition_module"):
        print("  beam_definition_module  : {}".format(beam_offset_meta["beam_definition_module"]))
    print("  source_mode             : {}".format(beam_offset_meta.get("source_mode", "unknown")))
    if primary_beam_validation is not None:
        print("  primary_beam_check      : {}".format(primary_beam_validation["primary_beam_check_status"]))
        print("  allowed_pb_radius_deg   : {:.8f}".format(primary_beam_validation["primary_beam_fwhm_radius_deg"]))
        print("  max_beam_offset_deg     : {:.8f}".format(primary_beam_validation["max_offset_deg"]))
        print("  primary_beam_result     : {}".format(primary_beam_validation["primary_beam_validation_message"]))
    print("")
    print("  Beam 0 (main beam)")
    print(
        "    offset row            : dEast {de:+.8f} deg | dNorth {dn:+.8f} deg | "
        "offset {off:.8f} deg".format(
            de=beam0_row["dEast_deg"],
            dn=beam0_row["dNorth_deg"],
            off=beam0_row["offset_deg"],
        )
    )
    print(
        "    pointing at t_ref     : az {az:.8f} deg | el {el:.8f} deg".format(
            az=beam_state["beam_az_deg"][0],
            el=beam_state["beam_el_deg"][0],
        )
    )
    print("")
    print("  Beam 1 (second beam)")
    print(
        "    offset row            : dEast {de:+.8f} deg | dNorth {dn:+.8f} deg | "
        "offset {off:.8f} deg | PA {pa:.8f} deg".format(
            de=beam1_row["dEast_deg"],
            dn=beam1_row["dNorth_deg"],
            off=beam1_row["offset_deg"],
            pa=beam1_row["position_angle_deg"],
        )
    )
    print(
        "    pointing at t_ref     : az {az:.8f} deg | el {el:.8f} deg".format(
            az=beam_state["beam_az_deg"][1],
            el=beam_state["beam_el_deg"][1],
        )
    )
    print("    separation from beam0 : {:.8f} deg".format(beam01_sep_deg))
    print("")


def print_summary_parameters(package):
    """Print Beam 0 active-channel summary for one slot package."""

    print("  Beam 0 | active CH0..CH{}".format(package.active_signal_inputs - 1))
    for channel_index in range(package.active_signal_inputs):
        print(
            "    CH{channel:02d} | A {a:+.9e} | B {b:+.9e} | omega {omega:+.9e}".format(
                channel=channel_index,
                a=float(package.a_mat[channel_index, 0]),
                b=float(package.b_mat[channel_index, 0]),
                omega=float(package.w_mat[channel_index, 0]),
            )
        )


def print_full_parameters(package):
    """Print all Beam / active channel values for one slot package."""

    for beam_index in range(TOTAL_BEAMS):
        print("  Beam {:02d}".format(beam_index))
        for channel_index in range(package.active_signal_inputs):
            print(
                "    CH{channel:02d} | A {a:+.9e} | B {b:+.9e} | omega {omega:+.9e}".format(
                    channel=channel_index,
                    a=float(package.a_mat[channel_index, beam_index]),
                    b=float(package.b_mat[channel_index, beam_index]),
                    omega=float(package.w_mat[channel_index, beam_index]),
                )
            )


def print_slot_terminal_update(package, publish_action, publish_time_utc):
    """Print one terminal update after publishing or skipping a slot."""

    print(
        "Slot {slot:05d} | epoch {epoch} UTC | action {action} | visible {visible} | "
        "override {override} | main az {az:9.3f} deg | el {el:9.3f} deg | publish {publish}".format(
            slot=package.slot_index,
            epoch=package.epoch_time_utc.strftime("%Y-%m-%d %H:%M:%S"),
            action=publish_action,
            visible=package.visible_now,
            override=package.simulation_override,
            az=package.main_az_deg,
            el=package.main_el_deg,
            publish=format_utc_timestamp(publish_time_utc),
        )
    )
    for line in build_visibility_status_lines(package.visibility_window):
        print(line)
    if publish_action != "published":
        return
    if TERMINAL_PRINT_MODE == "summary":
        print_summary_parameters(package)
    elif TERMINAL_PRINT_MODE == "full":
        print_full_parameters(package)


def build_slot_package(
    session,
    slot_index,
    antennas,
    katpoint_target,
    katpoint_antennas,
    antenna_names,
    ra_rad,
    dec_rad,
    beam_offset_rows,
    beam_offset_meta,
):
    """Build one logical 10-second slot package."""

    epoch_time_utc = session.t_ref_utc + timedelta(seconds=slot_index * session.update_period_seconds)
    compute_start_time_utc = utc_now()

    state_tn = compute_beam_model_state(
        antennas[0],
        katpoint_target,
        katpoint_antennas,
        antenna_names,
        epoch_time_utc,
        ra_rad,
        dec_rad,
        beam_offset_rows,
    )
    time_context = state_tn["time_context"]
    source_state = state_tn["source_state"]
    antenna_beam_results = state_tn["antenna_beam_results"]
    active_signal_inputs = get_active_signal_inputs(len(antennas))
    visibility_window = compute_visibility_window(
        antennas[0],
        epoch_time_utc,
        TARGET_RA,
        TARGET_DEC,
        MIN_ELEVATION_DEG,
    )
    visible_now = bool(visibility_window.visible_now)
    simulation_override = bool((not visible_now) and SIMULATION_IGNORE_VISIBILITY)
    should_publish = bool(visible_now or simulation_override)

    a_mat = np.zeros((TOTAL_SIGNAL_INPUTS, TOTAL_BEAMS), dtype=np.float32)
    b_mat = np.zeros((TOTAL_SIGNAL_INPUTS, TOTAL_BEAMS), dtype=np.float32)
    w_mat = np.zeros((TOTAL_SIGNAL_INPUTS, TOTAL_BEAMS), dtype=np.float32)
    beam_frames = []

    if should_publish:
        abomega = compute_abomega(
            antennas[0],
            antennas,
            katpoint_target,
            katpoint_antennas,
            antenna_names,
            epoch_time_utc,
            ra_rad,
            dec_rad,
            beam_offset_rows,
            state_tn=state_tn,
        )
        a_mat = expand_antennas_to_twenty_rows(abomega["a_ant"])
        b_mat = expand_antennas_to_twenty_rows(abomega["b_ant"])
        w_mat = expand_antennas_to_twenty_rows(abomega["omega_ant"])
        beam_frames = abomega["beam_frames"]

    compute_finish_time_utc = utc_now()
    return SlotPackage(
        slot_index=slot_index,
        epoch_time_utc=epoch_time_utc,
        julian_date=time_context["julian_date"],
        lst_hms=source_state["lst_hms"],
        main_az_deg=source_state["azimuth_deg"],
        main_el_deg=source_state["elevation_deg"],
        active_signal_inputs=active_signal_inputs,
        visible_now=visible_now,
        simulation_override=simulation_override,
        should_publish=should_publish,
        compute_start_time_utc=compute_start_time_utc,
        compute_finish_time_utc=compute_finish_time_utc,
        a_mat=np.asarray(a_mat, dtype=np.float32),
        b_mat=np.asarray(b_mat, dtype=np.float32),
        w_mat=np.asarray(w_mat, dtype=np.float32),
        beam_frames=beam_frames,
        antenna_beam_results=antenna_beam_results,
        visibility_window=visibility_window,
        beam_offset_source=beam_offset_meta["source"],
        beam_definition_file=beam_offset_meta.get("beam_definition_file"),
        beam_definition_module=beam_offset_meta.get("beam_definition_module"),
    )


def build_slot_log_record(session, package, publish_action, publish_time_utc):
    """Build one per-slot log record."""

    return {
        "session_id": session.session_id,
        "system_start_time_local": format_local_timestamp(session.program_start_local),
        "system_start_time_utc": format_utc_timestamp(session.program_start_utc),
        "t_ref_utc": format_utc_timestamp(session.t_ref_utc),
        "slot_index": package.slot_index,
        "epoch_time": format_utc_timestamp(package.epoch_time_utc),
        "compute_start_time": format_utc_timestamp(package.compute_start_time_utc),
        "compute_finish_time": format_utc_timestamp(package.compute_finish_time_utc),
        "publish_time": format_utc_timestamp(publish_time_utc),
        "publish_action": publish_action,
        "visible_now": bool(package.visible_now),
        "simulation_override": bool(package.simulation_override),
        "active_signal_inputs": int(package.active_signal_inputs),
        "visibility_rise_utc": format_utc_timestamp(package.visibility_window.rise_utc),
        "visibility_set_utc": format_utc_timestamp(package.visibility_window.set_utc),
        "visibility_next_rise_utc": format_utc_timestamp(package.visibility_window.next_rise_utc),
        "visibility_transit_utc": format_utc_timestamp(package.visibility_window.transit_utc),
        "visibility_circumpolar": bool(package.visibility_window.circumpolar),
        "visibility_never_up": bool(package.visibility_window.never_up),
        "visibility_window_contains_epoch": bool(package.visibility_window.window_contains_epoch),
        "main_az_deg": float(package.main_az_deg),
        "main_el_deg": float(package.main_el_deg),
        "lst_hms": package.lst_hms,
        "julian_date": float(package.julian_date),
        "beam_offset_report_txt": BEAM_OFFSET_REPORT_TXT,
        "beam_direction_report_txt": BEAM_DIRECTION_REPORT_TXT,
    }


def precompute_worker(
    session,
    antennas,
    beam_offset_rows,
    beam_offset_meta,
    slot_queue,
    stop_event,
    error_queue,
):
    """Precompute future slot packages and feed the queue."""

    try:
        ra_rad = parse_ra_to_rad(TARGET_RA)
        dec_rad = parse_dec_to_rad(TARGET_DEC)
        katpoint_antennas, antenna_names = build_katpoint_antennas_from_records(antennas)
        katpoint_target = build_katpoint_target_from_radec(TARGET_RA, TARGET_DEC)
        next_slot_to_compute = 0

        while not stop_event.is_set():
            package = build_slot_package(
                session,
                next_slot_to_compute,
                antennas,
                katpoint_target,
                katpoint_antennas,
                antenna_names,
                ra_rad,
                dec_rad,
                beam_offset_rows,
                beam_offset_meta,
            )
            while not stop_event.is_set():
                try:
                    slot_queue.put(package, timeout=QUEUE_TIMEOUT_SECONDS)
                    break
                except queue.Full:
                    continue
            next_slot_to_compute += 1
    except Exception as exc:
        error_queue.put(exc)
        stop_event.set()


def publisher_worker(session, slot_queue, stop_event, error_queue):
    """Publish slot packages based on their logical slot times."""

    try:
        next_slot_to_publish = 0
        while not stop_event.is_set():
            if MAX_PUBLISH_SLOTS is not None and next_slot_to_publish >= MAX_PUBLISH_SLOTS:
                stop_event.set()
                break

            slot_epoch_utc = session.t_ref_utc + timedelta(seconds=next_slot_to_publish * session.update_period_seconds)
            ready_time_utc = slot_epoch_utc - timedelta(seconds=session.publish_lead_seconds)
            now_utc = utc_now()

            if now_utc < ready_time_utc:
                remaining_seconds = (ready_time_utc - now_utc).total_seconds()
                stop_event.wait(min(remaining_seconds, QUEUE_TIMEOUT_SECONDS))
                continue

            try:
                package = slot_queue.get(timeout=QUEUE_TIMEOUT_SECONDS)
            except queue.Empty:
                continue

            if package.slot_index != next_slot_to_publish:
                raise RuntimeError(
                    "Slot queue order mismatch: expected slot {}, got slot {}".format(
                        next_slot_to_publish,
                        package.slot_index,
                    )
                )

            publish_time_utc = utc_now()
            write_beam_direction_report(package)
            if package.should_publish:
                publish_slot_files(package)
                publish_action = "published"
            else:
                publish_action = "skipped_invisible"

            publish_time_utc = utc_now()
            append_log_record(build_slot_log_record(session, package, publish_action, publish_time_utc))
            print_slot_terminal_update(package, publish_action, publish_time_utc)
            next_slot_to_publish += 1
    except Exception as exc:
        error_queue.put(exc)
        stop_event.set()


def print_trace_step(title):
    """Print one visible startup-flow step header."""

    if PRINT_PIPELINE_STEPS:
        print("")
        print("=== {} ===".format(title))


def print_trace_startup_config():
    """Print the key user-adjustable startup parameters."""

    if not PRINT_STARTUP_CONFIG:
        return

    print_trace_step("Input Configuration")
    print("[Input files]")
    print("ants_txt                 : {}".format(ANTS_TXT))
    print("")

    print("[Source and beam model]")
    print("target_ra                : {}".format(TARGET_RA))
    print("target_dec               : {}".format(TARGET_DEC))
    print("min_elevation_deg        : {:.3f}".format(MIN_ELEVATION_DEG))
    print("simulation_ignore_vis    : {}".format(SIMULATION_IGNORE_VISIBILITY))
    print("center_freq_hz           : {:.6f}".format(CENTER_FREQ_HZ))
    print("omega_delta_seconds      : {:.3f}".format(OMEGA_DELTA_SECONDS))
    print("")

    print("[Publish schedule]")
    print("update_period_seconds    : {}".format(UPDATE_PERIOD_SECONDS))
    print("publish_lead_seconds     : {:.3f}".format(PUBLISH_LEAD_SECONDS))
    print("precompute_queue_slots   : {}".format(PRECOMPUTE_QUEUE_SLOTS))
    print("")

    print("[Beam offset source]")
    print("beam_offset_table_file   : {}".format(BEAM_OFFSET_TABLE_FILE))
    print("auto_generate_offsets    : disabled")
    print("external BeamID support  : 1..32 or 0..31")
    print("internal beam_index      : 0..31")
    print("")

    print("[Debug and terminal]")
    print("terminal_print_mode      : {}".format(TERMINAL_PRINT_MODE))
    print("max_publish_slots        : {}".format(MAX_PUBLISH_SLOTS))
    print("print_pipeline_steps     : {}".format(PRINT_PIPELINE_STEPS))
    print("print_startup_config     : {}".format(PRINT_STARTUP_CONFIG))
    print("print_startup_samples    : {}".format(PRINT_STARTUP_SAMPLES))
    print("")

    print("[Output files]")
    print("trace_mode_phase_file    : {}".format(TRACE_MODE_PHASE_TXT))
    print("beam_offset_report       : {}".format(BEAM_OFFSET_REPORT_TXT))
    print("beam_direction_report    : {}".format(BEAM_DIRECTION_REPORT_TXT))
    print("log_file                 : {}".format(LOG_FILE))
    print("")

    print("[Trace output contract]")
    print("trace layout             : A[20,32] + B[20,32] + omega[20,32]")
    print("trace dtype              : little-endian float32")
    print("trace value count        : {}".format(TRACE_STREAM_FLOAT_COUNT))
    print("trace bytes              : {}".format(TRACE_STREAM_BYTES))
    print("A/B frequency factor     : not included")
    print("GPU formula              : Phi3 = nu * (A*cos(delta) + B*sin(delta) - A)")
    print("delta                    : omega * (t - t0)")
    print("t0 in parameter file     : no")

def print_trace_startup_samples(session, antennas, beam_offset_rows, beam_offset_meta, startup_snapshot, first_slot_snapshot):
    """Print a few compact startup samples that are useful while debugging."""

    if not PRINT_STARTUP_SAMPLES:
        return

    print_trace_step("Startup Samples")
    print("session_id               : {}".format(session.session_id))
    print("antenna_names            : {}".format(", ".join(ant.name for ant in antennas)))
    print("t_ref_utc                : {}".format(format_utc_timestamp(session.t_ref_utc)))
    print("beam_offset_source_mode  : {}".format(beam_offset_meta.get("source_mode")))
    print("beam_offset_row_count    : {}".format(len(beam_offset_rows)))
    print("trace matrix shape       : A/B/omega each {} x {}".format(TOTAL_SIGNAL_INPUTS, TOTAL_BEAMS))
    print("trace stream float count : {}".format(TRACE_STREAM_FLOAT_COUNT))
    print("trace output file        : {}".format(TRACE_MODE_PHASE_TXT))
    print("A/B frequency factor     : not included")
    print("startup_visible_now      : {}".format(startup_snapshot["visible_now"]))
    print("slot0_visible_now        : {}".format(first_slot_snapshot["visible_now"]))
    if beam_offset_rows:
        first_row = beam_offset_rows[0]
        print(
            "center_beam_offset_deg   : dEast={:+.8f}, dNorth={:+.8f}".format(
                first_row["dEast_deg"],
                first_row["dNorth_deg"],
            )
        )


def build_session_context():
    """Build the continuous slot-tracking session context."""

    program_start_local = datetime.now().astimezone()
    program_start_utc = program_start_local.astimezone(timezone.utc)
    return SessionContext(
        session_id=uuid.uuid4().hex[:12],
        program_start_local=program_start_local,
        program_start_utc=program_start_utc,
        t_ref_utc=ceil_utc_to_next_slot_boundary(program_start_utc, UPDATE_PERIOD_SECONDS),
        update_period_seconds=UPDATE_PERIOD_SECONDS,
        publish_lead_seconds=PUBLISH_LEAD_SECONDS,
    )


def run_trace_phase_coeff_session():
    """Run the trace-mode publisher with explicit startup steps."""

    print_trace_startup_config()

    print_trace_step("Step 1: Build Session Context")
    session = build_session_context()
    print("session_id               : {}".format(session.session_id))
    print("program_start_utc        : {}".format(format_utc_timestamp(session.program_start_utc)))
    print("t_ref_utc                : {}".format(format_utc_timestamp(session.t_ref_utc)))

    print_trace_step("Step 2: Load Antennas")
    antennas = read_antenna_file(ANTS_TXT)
    active_signal_inputs = get_active_signal_inputs(len(antennas))
    padded_signal_inputs = TOTAL_SIGNAL_INPUTS - active_signal_inputs
    print("loaded antennas          : {}".format(len(antennas)))
    print("active signal inputs     : {}".format(active_signal_inputs))
    print("total signal inputs      : {}".format(TOTAL_SIGNAL_INPUTS))
    print("padded signal inputs     : {}".format(padded_signal_inputs))
    if padded_signal_inputs > 0:
        print("padded input indices     : {}..{}".format(active_signal_inputs, TOTAL_SIGNAL_INPUTS - 1))
        print("padded input ids         : {}..{}".format(active_signal_inputs + 1, TOTAL_SIGNAL_INPUTS))
    else:
        print("padded input indices     : none")
        print("padded input ids         : none")

    print_trace_step("Step 3: Resolve Beam Offset Table")
    beam_offset_rows, beam_offset_meta = resolve_beam_offset_table(session, antennas)
    print("beam_offset_source       : {}".format(beam_offset_meta.get("source")))
    print("beam_offset_table_file   : {}".format(BEAM_OFFSET_TABLE_FILE))
    print("beam_offset_rows         : {}".format(len(beam_offset_rows)))
    print("auto_generate_offsets    : disabled")
    print("file BeamID convention   : 1..32 or 0..31 accepted")
    print("internal beam_index      : 0..31")
    if beam_offset_rows:
        first = beam_offset_rows[0]
        print("center beam file id      : {}".format(first.get("file_beam_id", first["beam_id"])))
        print("center beam index        : {}".format(first.get("beam_index", first["beam_id"])))
        print(
            "center beam offset deg   : dEast={:+.8f}, dNorth={:+.8f}".format(
                first["dEast_deg"],
                first["dNorth_deg"],
            )
        )

    print_trace_step("Step 4: Write Startup Beam Offset Report")
    write_beam_offset_report(session, antennas, beam_offset_rows, beam_offset_meta)
    print("beam_offset_report       : {}".format(BEAM_OFFSET_REPORT_TXT))

    print_trace_step("Step 5: Build Startup Snapshots")
    startup_snapshot = build_live_status_snapshot(
        ants_txt=ANTS_TXT,
        target_ra=TARGET_RA,
        target_dec=TARGET_DEC,
        min_elevation_deg=MIN_ELEVATION_DEG,
        when_utc=session.program_start_utc,
        antennas=antennas,
    )
    first_slot_snapshot = build_live_status_snapshot(
        ants_txt=ANTS_TXT,
        target_ra=TARGET_RA,
        target_dec=TARGET_DEC,
        min_elevation_deg=MIN_ELEVATION_DEG,
        when_utc=session.t_ref_utc,
        antennas=antennas,
    )
    print("startup_visible_now      : {}".format(startup_snapshot["visible_now"]))
    print("slot0_visible_now        : {}".format(first_slot_snapshot["visible_now"]))

    print_trace_startup_samples(
        session,
        antennas,
        beam_offset_rows,
        beam_offset_meta,
        startup_snapshot,
        first_slot_snapshot,
    )

    print_trace_step("Step 6: Print Startup Summary And Log Session")
    print_session_summary(session, antennas, beam_offset_meta)
    print_beam_offset_input_summary(session, antennas, beam_offset_rows, beam_offset_meta)
    print_live_status_snapshot(startup_snapshot, "Current source status at program start")
    print_live_status_snapshot(first_slot_snapshot, "Slot 0 source status at t_ref")
    append_log_record(
        {
            "session_id": session.session_id,
            "event": "session_start",
            "system_start_time_local": format_local_timestamp(session.program_start_local),
            "system_start_time_utc": format_utc_timestamp(session.program_start_utc),
            "program_start_local": format_local_timestamp(session.program_start_local),
            "program_start_utc": format_utc_timestamp(session.program_start_utc),
            "t_ref_utc": format_utc_timestamp(session.t_ref_utc),
            "update_period_seconds": session.update_period_seconds,
            "publish_lead_seconds": session.publish_lead_seconds,
            "simulation_ignore_visibility": bool(SIMULATION_IGNORE_VISIBILITY),
            "startup_visible_now": bool(startup_snapshot["visible_now"]),
            "startup_visibility_rise_utc": format_utc_timestamp(startup_snapshot["visibility_window"].rise_utc),
            "startup_visibility_set_utc": format_utc_timestamp(startup_snapshot["visibility_window"].set_utc),
            "startup_visibility_next_rise_utc": format_utc_timestamp(
                startup_snapshot["visibility_window"].next_rise_utc
            ),
        }
    )

    print_trace_step("Step 7: Create Queues And Worker Threads")
    slot_queue = queue.Queue(maxsize=PRECOMPUTE_QUEUE_SLOTS)
    error_queue = queue.Queue()
    stop_event = threading.Event()

    precompute_thread = threading.Thread(
        target=precompute_worker,
        name="precompute-thread",
        args=(
            session,
            antennas,
            beam_offset_rows,
            beam_offset_meta,
            slot_queue,
            stop_event,
            error_queue,
        ),
    )
    publisher_thread = threading.Thread(
        target=publisher_worker,
        name="publisher-thread",
        args=(session, slot_queue, stop_event, error_queue),
    )
    print("slot_queue_maxsize       : {}".format(PRECOMPUTE_QUEUE_SLOTS))

    print_trace_step("Step 8: Start Worker Threads")
    precompute_thread.start()
    publisher_thread.start()
    print("workers_started          : precompute-thread, publisher-thread")

    print_trace_step("Step 9: Monitor Publisher Loop")
    try:
        while publisher_thread.is_alive():
            publisher_thread.join(timeout=QUEUE_TIMEOUT_SECONDS)
            if not error_queue.empty():
                raise error_queue.get()
    except KeyboardInterrupt:
        print("")
        print("Stopping slot publisher because of KeyboardInterrupt")
        stop_event.set()
    finally:
        stop_event.set()
        precompute_thread.join(timeout=5.0)
        publisher_thread.join(timeout=5.0)

    if not error_queue.empty():
        raise error_queue.get()


def main():
    """Start one continuous slot-based publishing session."""

    run_trace_phase_coeff_session()


if __name__ == "__main__":
    main()
