# -*- coding: utf-8 -*-
from __future__ import division, print_function

import io
import os
import time
from datetime import datetime

import numpy as np


# =========================
# User configuration
# =========================
# Main loop wake-up cadence. Each cycle can print status and optionally reuse
# the previous solution if the TXT reread interval has not been reached yet.
LOOP_INTERVAL_SECONDS = 120.0
# Minimum interval between rereading the cable-delay TXT and regenerating the
# DAT/NPZ outputs. The actual reread cadence is quantized by LOOP_INTERVAL_SECONDS.
TXT_READ_INTERVAL_SECONDS = 60.0
RUN_ONCE = False

N_FREQ_CHANNELS = 2048
FREQ_START_HZ = 1.0e9
FREQ_STOP_HZ = 1.5e9
TOTAL_SIGNAL_INPUTS = 20
TIME_DELAY_STEP_NS = 0.9765625
MAX_TIME_DELAY_STEPS = 4096
TRACE_VALUE = 1
REQUIRE_REFERENCE_INPUTS = True
REFERENCE_INPUT_IDS = (1, 2)
MISSING_SIGNAL_NAME_PREFIX = "MISSING_INPUT_"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CABLE_DELAY_TXT = os.path.join(SCRIPT_DIR, "cable_relative_delay_ns.txt")
DAT_FILE = os.path.join(SCRIPT_DIR, "time_phase_coeff.dat")
NPZ_FILE = os.path.join(SCRIPT_DIR, "time_phase_coeff.npz")

# Example cable_relative_delay_ns.txt:
# input_id signal_name delay_ns
# 1 天线1_1 0.00
# 2 天线1_2 0.00
# 5 天线3_1 1.20
# 6 天线3_2 1.25
# 20 天线10_2 2.20
#
# Inputs 3,4,7,8,... not listed above are treated as missing and zero-filled.


def atomic_replace(src_path, dst_path):
    if hasattr(os, "replace"):
        os.replace(src_path, dst_path)
        return

    if os.name == "nt" and os.path.exists(dst_path):
        os.remove(dst_path)
    os.rename(src_path, dst_path)


def trunc_toward_zero(values):
    return np.trunc(values).astype(np.int32)


def quantize_q14(values):
    scaled = np.rint(values * (1 << 14))
    clipped = np.clip(scaled, -32768, 32767)
    return clipped.astype(np.int16)


def build_frequency_axis_hz():
    return np.linspace(FREQ_START_HZ, FREQ_STOP_HZ, N_FREQ_CHANNELS, dtype=np.float64)


def load_signal_relative_delays_ns(txt_path, total_inputs):
    signal_names = [None] * total_inputs
    cable_delay_ns = np.zeros(total_inputs, dtype=np.float64)
    active_mask = np.zeros(total_inputs, dtype=bool)

    with io.open(txt_path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 3:
                raise ValueError(
                    "Line {}: need <input_id> <signal_name> <delay_ns>".format(lineno)
                )

            input_text = parts[0]
            signal_name = parts[1]
            delay_text = parts[2]

            try:
                input_id = int(input_text)
            except ValueError:
                raise ValueError(
                    "Line {}: invalid input_id {!r}; expected integer 1..{}".format(
                        lineno,
                        input_text,
                        total_inputs,
                    )
                )

            if input_id < 1 or input_id > total_inputs:
                raise ValueError(
                    "Line {}: input_id {} out of range 1..{}".format(
                        lineno,
                        input_id,
                        total_inputs,
                    )
                )

            input_index = input_id - 1
            if active_mask[input_index]:
                raise ValueError("Line {}: duplicate input_id {}".format(lineno, input_id))

            try:
                delay_ns = float(delay_text)
            except ValueError:
                raise ValueError("Line {}: invalid delay_ns {!r}".format(lineno, delay_text))

            signal_names[input_index] = signal_name
            cable_delay_ns[input_index] = delay_ns
            active_mask[input_index] = True

    active_input_indices = np.nonzero(active_mask)[0].tolist()
    missing_input_indices = np.nonzero(~active_mask)[0].tolist()

    if not active_input_indices:
        raise ValueError("No active signal-delay rows found in {}".format(txt_path))

    if REQUIRE_REFERENCE_INPUTS:
        for ref_input_id in REFERENCE_INPUT_IDS:
            ref_index = ref_input_id - 1
            if ref_index < 0 or ref_index >= total_inputs:
                raise ValueError("Invalid reference input id {}".format(ref_input_id))
            if not active_mask[ref_index]:
                raise ValueError(
                    "Reference input {} is missing from {}".format(
                        ref_input_id,
                        txt_path,
                    )
                )

    for ref_input_id in REFERENCE_INPUT_IDS:
        ref_index = ref_input_id - 1
        if 0 <= ref_index < total_inputs and active_mask[ref_index]:
            if abs(cable_delay_ns[ref_index]) > 1.0e-6:
                raise ValueError(
                    "Reference input {} must be 0 ns, got {}={:+.6f} ns".format(
                        ref_input_id,
                        signal_names[ref_index],
                        cable_delay_ns[ref_index],
                    )
                )

    for input_index in missing_input_indices:
        signal_names[input_index] = "{}{:02d}".format(
            MISSING_SIGNAL_NAME_PREFIX,
            input_index + 1,
        )

    return (
        signal_names,
        cable_delay_ns,
        active_mask,
        active_input_indices,
        missing_input_indices,
    )


def convert_cable_delay_to_compensation_delay_ns(cable_delay_ns, active_mask):
    active_delays_ns = cable_delay_ns[active_mask]
    if active_delays_ns.size == 0:
        raise ValueError("No active inputs available for compensation delay calculation")

    reference_physical_delay_ns = float(np.max(active_delays_ns))
    compensation_delay_ns = np.zeros_like(cable_delay_ns, dtype=np.float64)
    compensation_delay_ns[active_mask] = (
        reference_physical_delay_ns - cable_delay_ns[active_mask]
    )
    if np.any(compensation_delay_ns[active_mask] < -1.0e-9):
        raise ValueError("Compensation delay must be non-negative for active inputs")
    compensation_delay_ns[active_mask] = np.maximum(compensation_delay_ns[active_mask], 0.0)
    return compensation_delay_ns.astype(np.float64), reference_physical_delay_ns


def time_delay_steps_to_ns(time_delay_steps_array):
    return time_delay_steps_array.astype(np.float64) * TIME_DELAY_STEP_NS


def split_time_and_fine_delay(relative_delay_ns_array):
    time_delay_steps_i32 = trunc_toward_zero(relative_delay_ns_array / TIME_DELAY_STEP_NS)
    if np.any(time_delay_steps_i32 < 0):
        raise ValueError("Time delay step count must be non-negative")
    if np.any(time_delay_steps_i32 > MAX_TIME_DELAY_STEPS):
        raise ValueError(
            "Time delay step count exceeds max {} steps ({:.6f} ns)".format(
                MAX_TIME_DELAY_STEPS,
                MAX_TIME_DELAY_STEPS * TIME_DELAY_STEP_NS,
            )
        )

    time_delay_ns = time_delay_steps_to_ns(time_delay_steps_i32)
    fine_delay_ns = relative_delay_ns_array - time_delay_ns
    return time_delay_steps_i32, fine_delay_ns, time_delay_ns


def build_fine_delay_coefficients_q14(frequencies_hz, fine_delay_ns_array):
    tau_fine_s = fine_delay_ns_array * 1e-9
    phase = 2.0 * np.pi * frequencies_hz[None, :] * tau_fine_s[:, None]
    coeff = np.exp(-1j * phase)
    coeff_real_q14 = quantize_q14(np.real(coeff))
    coeff_imag_q14 = quantize_q14(np.imag(coeff))
    return coeff_real_q14, coeff_imag_q14


def zero_missing_input_coefficients(
    time_delay_steps_i32_array,
    coeff_real_q14,
    coeff_imag_q14,
    active_mask,
):
    inactive_mask = ~active_mask

    time_delay_steps_i32_array = np.asarray(time_delay_steps_i32_array).copy()
    coeff_real_q14 = np.asarray(coeff_real_q14).copy()
    coeff_imag_q14 = np.asarray(coeff_imag_q14).copy()

    time_delay_steps_i32_array[inactive_mask] = 0
    coeff_real_q14[inactive_mask, :] = 0
    coeff_imag_q14[inactive_mask, :] = 0

    return time_delay_steps_i32_array, coeff_real_q14, coeff_imag_q14


def validate_solution_shapes(time_delay_u16_array, coeff_real_q14, coeff_imag_q14):
    if np.asarray(time_delay_u16_array).shape != (TOTAL_SIGNAL_INPUTS,):
        raise ValueError(
            "time_delay array shape must be ({},), got {}".format(
                TOTAL_SIGNAL_INPUTS,
                np.asarray(time_delay_u16_array).shape,
            )
        )

    expected_coeff_shape = (TOTAL_SIGNAL_INPUTS, N_FREQ_CHANNELS)
    if np.asarray(coeff_real_q14).shape != expected_coeff_shape:
        raise ValueError(
            "coeff_real_q14 shape must be {}, got {}".format(
                expected_coeff_shape,
                np.asarray(coeff_real_q14).shape,
            )
        )
    if np.asarray(coeff_imag_q14).shape != expected_coeff_shape:
        raise ValueError(
            "coeff_imag_q14 shape must be {}, got {}".format(
                expected_coeff_shape,
                np.asarray(coeff_imag_q14).shape,
            )
        )


def atomic_write_dat(out_path, time_delay_u16_array, coeff_real_q14, coeff_imag_q14, trace_value):
    validate_solution_shapes(time_delay_u16_array, coeff_real_q14, coeff_imag_q14)
    tmp_path = out_path + ".tmp"

    if trace_value < 0 or trace_value > 0xFFFF:
        raise ValueError("TRACE_VALUE must fit in uint16")

    time_delay_le = np.asarray(time_delay_u16_array, dtype="<u2")
    fine_interleaved = np.empty((coeff_real_q14.shape[0], coeff_real_q14.shape[1], 2), dtype="<i2")
    fine_interleaved[:, :, 0] = coeff_real_q14.astype("<i2")
    fine_interleaved[:, :, 1] = coeff_imag_q14.astype("<i2")
    trace_le = np.asarray([trace_value], dtype="<u2")

    with open(tmp_path, "wb") as f:
        f.write(time_delay_le.tobytes(order="C"))
        f.write(fine_interleaved.tobytes(order="C"))
        f.write(trace_le.tobytes(order="C"))
    atomic_replace(tmp_path, out_path)


def atomic_write_npz(out_path, time_delay_u16_array, coeff_real_q14, coeff_imag_q14, trace_value):
    validate_solution_shapes(time_delay_u16_array, coeff_real_q14, coeff_imag_q14)
    tmp_path = out_path + ".tmp"

    coeff_complex = (
        coeff_real_q14.astype(np.float32) / (1 << 14)
        + 1j * (coeff_imag_q14.astype(np.float32) / (1 << 14))
    ).astype(np.complex64)
    trace_u16 = np.asarray([trace_value], dtype=np.uint16)

    with open(tmp_path, "wb") as f:
        np.savez(
            f,
            time_delay=np.asarray(time_delay_u16_array, dtype=np.uint16),
            coeff=coeff_complex,
            trace=trace_u16,
        )
    atomic_replace(tmp_path, out_path)


def write_current_solution_from_txt(frequencies_hz):
    (
        signal_names,
        cable_delay_ns,
        active_mask,
        active_input_indices,
        missing_input_indices,
    ) = load_signal_relative_delays_ns(
        CABLE_DELAY_TXT,
        TOTAL_SIGNAL_INPUTS,
    )
    compensation_delay_ns, reference_physical_delay_ns = convert_cable_delay_to_compensation_delay_ns(
        cable_delay_ns,
        active_mask,
    )
    time_delay_steps_i32_array, fine_delay_ns_array, time_delay_ns_array = split_time_and_fine_delay(
        compensation_delay_ns
    )
    coeff_real_q14, coeff_imag_q14 = build_fine_delay_coefficients_q14(frequencies_hz, fine_delay_ns_array)
    (
        time_delay_steps_i32_array,
        coeff_real_q14,
        coeff_imag_q14,
    ) = zero_missing_input_coefficients(
        time_delay_steps_i32_array,
        coeff_real_q14,
        coeff_imag_q14,
        active_mask,
    )
    signal_time_delay_u16_array = np.asarray(time_delay_steps_i32_array, dtype=np.uint16)

    atomic_write_dat(
        DAT_FILE,
        signal_time_delay_u16_array,
        coeff_real_q14,
        coeff_imag_q14,
        TRACE_VALUE,
    )
    atomic_write_npz(
        NPZ_FILE,
        signal_time_delay_u16_array,
        coeff_real_q14,
        coeff_imag_q14,
        TRACE_VALUE,
    )

    return (
        signal_names,
        cable_delay_ns,
        compensation_delay_ns,
        time_delay_steps_i32_array,
        time_delay_ns_array,
        fine_delay_ns_array,
        reference_physical_delay_ns,
        active_mask,
        active_input_indices,
        missing_input_indices,
    )


def print_cycle_report(
    cycle_index,
    utc_now,
    input_path,
    txt_reloaded_this_cycle,
    signal_names,
    cable_delay_ns,
    compensation_delay_ns,
    time_delay_steps_i32_array,
    time_delay_ns_array,
    fine_delay_ns_array,
    reference_physical_delay_ns,
    active_mask,
    active_input_indices,
    missing_input_indices,
):
    print("Cycle {:04d} | UTC {}".format(cycle_index, utc_now.strftime("%Y-%m-%d %H:%M:%S")))
    print("  input cable delay file: {}".format(input_path))
    print("  txt reloaded this cycle: {}".format("yes" if txt_reloaded_this_cycle else "no"))
    print("  active signal inputs: {}".format(len(active_input_indices)))
    print("  missing signal inputs: {}".format(len(missing_input_indices)))
    print(
        "  active input ids: {}".format(
            ", ".join(str(i + 1) for i in active_input_indices) if active_input_indices else "none"
        )
    )
    print(
        "  missing input ids: {}".format(
            ", ".join(str(i + 1) for i in missing_input_indices) if missing_input_indices else "none"
        )
    )
    print("  missing input fill: time_delay=0, coeff=0+0j")
    print(
        "  physical max cable delay used as alignment reference: {:.6f} ns".format(
            reference_physical_delay_ns
        )
    )
    for i, name in enumerate(signal_names):
        input_id = i + 1
        if active_mask[i]:
            print(
                "  input {:02d} {:<12s} cable {:+.2f} ns | compensation {:9.6f} ns | time_delay {:6d} step ({:10.6f} ns) | fine {:+.6f} ns".format(
                    input_id,
                    name,
                    cable_delay_ns[i],
                    compensation_delay_ns[i],
                    int(time_delay_steps_i32_array[i]),
                    time_delay_ns_array[i],
                    fine_delay_ns_array[i],
                )
            )
        else:
            print(
                "  input {:02d} {:<12s} MISSING | time_delay      0 step | coeff 0+0j".format(
                    input_id,
                    name,
                )
            )
    print("  wrote {} hardware signal inputs".format(TOTAL_SIGNAL_INPUTS))
    print("")


def run_forever():
    frequencies_hz = build_frequency_axis_hz()
    if LOOP_INTERVAL_SECONDS <= 0:
        raise ValueError("LOOP_INTERVAL_SECONDS must be > 0")
    if TXT_READ_INTERVAL_SECONDS <= 0:
        raise ValueError("TXT_READ_INTERVAL_SECONDS must be > 0")

    print("Cable-delay-to-phase-coeff publisher")
    print("Input cable delay TXT: {}".format(CABLE_DELAY_TXT))
    print("TXT format: <input_id 1..20> <signal_name> <delay_ns>")
    print("Missing input ids are zero-filled")
    print("Reference input ids: {}".format(", ".join(str(x) for x in REFERENCE_INPUT_IDS)))
    print("Require reference inputs: {}".format(REQUIRE_REFERENCE_INPUTS))
    print("DAT file: {}".format(DAT_FILE))
    print("NPZ file: {}".format(NPZ_FILE))
    print("Hardware signal inputs: {}".format(TOTAL_SIGNAL_INPUTS))
    print("Loop interval: {:.1f} s".format(LOOP_INTERVAL_SECONDS))
    print("TXT read interval: {:.1f} s".format(TXT_READ_INTERVAL_SECONDS))
    print("Run once: {}".format(RUN_ONCE))
    print("Time delay step: {:.7f} ns".format(TIME_DELAY_STEP_NS))
    print("Max time delay: {} steps ({:.6f} ns)".format(MAX_TIME_DELAY_STEPS, MAX_TIME_DELAY_STEPS * TIME_DELAY_STEP_NS))
    print(
        "Frequency axis: {} channels from {:.3f} GHz to {:.3f} GHz".format(
            N_FREQ_CHANNELS,
            FREQ_START_HZ / 1.0e9,
            FREQ_STOP_HZ / 1.0e9,
        )
    )
    print("Trace value: {}".format(TRACE_VALUE))
    print(
        "DAT layout: 20x uint16 time_delay, then 20x2048 complex coeff "
        "(real int16 q14, imag int16 q14 interleaved), then trace uint16, little-endian"
    )
    print("NPZ keys: time_delay, coeff, trace")
    print("")

    cycle_index = 0
    if RUN_ONCE:
        utc_now = datetime.utcnow().replace(microsecond=0)
        results = write_current_solution_from_txt(frequencies_hz)
        print_cycle_report(cycle_index, utc_now, CABLE_DELAY_TXT, True, *results)
        return

    last_results = None
    last_txt_read_monotonic = None
    while True:
        utc_now = datetime.utcnow().replace(microsecond=0)
        now_monotonic = time.monotonic()
        txt_reloaded_this_cycle = (
            last_results is None
            or last_txt_read_monotonic is None
            or (now_monotonic - last_txt_read_monotonic) >= TXT_READ_INTERVAL_SECONDS
        )
        if txt_reloaded_this_cycle:
            last_results = write_current_solution_from_txt(frequencies_hz)
            last_txt_read_monotonic = now_monotonic
        print_cycle_report(cycle_index, utc_now, CABLE_DELAY_TXT, txt_reloaded_this_cycle, *last_results)
        cycle_index += 1
        time.sleep(LOOP_INTERVAL_SECONDS)


if __name__ == "__main__":
    run_forever()
