#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interactive editor for beam_coeff_antenna.txt.

The saved output is plain UTF-8 text so it can be opened directly in a text
editor:
- matrix shape: 20 x 32
- values are formatted from float32
- text layout: one flattened row with 640 space-separated values

Matrix meaning:
- rows    -> signal input index i
- columns -> beam index j
- value   -> M[i, j], typically 0.0 or 1.0
"""

from __future__ import division, print_function

import argparse
import hashlib
import io
import os

import numpy as np

try:
    import tkinter as tk
    from tkinter import messagebox, ttk
    TKINTER_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    tk = None
    messagebox = None
    ttk = None
    TKINTER_IMPORT_ERROR = exc


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = SCRIPT_DIR
DEFAULT_OUTPUT_FILENAME = "beam_coeff_antenna.txt"
DEFAULT_OUTPUT_PATH = os.path.join(DEFAULT_OUTPUT_DIR, DEFAULT_OUTPUT_FILENAME)

TOTAL_SIGNAL_INPUTS = 20
TOTAL_BEAMS = 32
DEFAULT_ACTIVE_SIGNAL_INPUTS = 8

CELL_WIDTH = 22
CELL_HEIGHT = 22
GRID_LEFT = 64
GRID_TOP = 42
GRID_BG = "#0e1520"
CELL_ON = "#49c384"
CELL_OFF = "#24313f"
CELL_GRID = "#4b5a6c"
CELL_TEXT = "#ebf3ff"
HEADER_TEXT = "#d7e4f3"
SELECTION_OUTLINE = "#ffcf66"


def build_default_matrix():
    """Return the same 20x32 default matrix used by the publisher today."""

    matrix = np.zeros((TOTAL_SIGNAL_INPUTS, TOTAL_BEAMS), dtype=np.float32)
    matrix[:DEFAULT_ACTIVE_SIGNAL_INPUTS, :] = 1.0
    return matrix


def build_output_path(output_dir, output_filename=DEFAULT_OUTPUT_FILENAME):
    """Build one output file path from a directory and filename."""

    if output_dir is None:
        raise ValueError("output_dir must not be None")
    if not output_filename:
        raise ValueError("output_filename must not be empty")

    normalized_dir = os.path.abspath(os.path.expanduser(str(output_dir).strip()))
    if os.path.exists(normalized_dir) and (not os.path.isdir(normalized_dir)):
        raise ValueError(
            "Output path exists but is not a directory: {}".format(normalized_dir)
        )
    return os.path.join(normalized_dir, output_filename)


def parse_args(argv=None):
    """Parse optional startup settings for the editor."""

    parser = argparse.ArgumentParser(
        description=(
            "Launch the beam_coeff_antenna editor with a configurable default "
            "output directory, or edit the matrix from the command line."
        )
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Default directory for beam_coeff_antenna.txt and its .md5 sidecar. "
            "The GUI output path field is prefilled from this directory."
        ),
    )
    parser.add_argument(
        "--write-default",
        action="store_true",
        help=(
            "Write the default 20x32 matrix to the output path and exit, "
            "without launching the Tk GUI."
        ),
    )
    parser.add_argument(
        "-i",
        "--input-file",
        dest="input_path",
        help=(
            "Optional existing beam_coeff_antenna file to load before applying "
            "command-line edits. If omitted, command-line edits start from the "
            "built-in default matrix."
        ),
    )
    parser.add_argument(
        "--set-all",
        dest="set_all_value",
        type=float,
        help=(
            "In non-GUI mode, set all 20x32 cells to one float value before "
            "applying any --set expressions."
        ),
    )
    parser.add_argument(
        "--set",
        dest="set_expressions",
        action="append",
        default=[],
        metavar="SIGNALS:BEAMS=VALUE",
        help=(
            "In non-GUI mode, set one signal/beam region. Example: "
            "\"0:3=1\", \"0-7:all=1\", \"8-19:0-31=0\". Repeat --set for "
            "multiple edits."
        ),
    )
    args = parser.parse_args(argv)

    if args.write_default and (args.input_path or (args.set_all_value is not None) or args.set_expressions):
        parser.error("--write-default cannot be combined with --input-file, --set-all, or --set")

    try:
        args.output_path = build_output_path(args.output_dir)
    except ValueError as exc:
        parser.error(str(exc))

    return args


def ensure_parent_dir(path):
    """Create the parent directory for one output file path if needed."""

    parent_dir = os.path.dirname(path)
    if parent_dir and (not os.path.exists(parent_dir)):
        os.makedirs(parent_dir)


def write_default_outputs(path):
    """Write the default matrix text file and its MD5 sidecar."""

    ensure_parent_dir(path)
    matrix = build_default_matrix()
    save_matrix_to_file(path, matrix)
    md5_path, checksum = write_md5_file_for_output(path)
    return path, md5_path, checksum


def parse_index_spec(spec_text, upper_bound, axis_name):
    """Parse one index spec such as 0, 0-7, 0,2,5, or all."""

    raw = str(spec_text).strip().lower()
    if not raw:
        raise ValueError("Empty {} index spec".format(axis_name))
    if raw in ("all", "*"):
        return list(range(upper_bound))

    indices = []
    seen = set()
    for part in raw.split(","):
        token = part.strip().lower()
        if not token:
            raise ValueError("Empty {} index token in {!r}".format(axis_name, spec_text))

        if token in ("all", "*"):
            values = range(upper_bound)
        elif "-" in token:
            range_parts = token.split("-", 1)
            if len(range_parts) != 2 or (not range_parts[0]) or (not range_parts[1]):
                raise ValueError(
                    "Invalid {} range {!r}; expected start-end".format(axis_name, token)
                )
            start = int(range_parts[0])
            stop = int(range_parts[1])
            if stop < start:
                raise ValueError(
                    "Invalid {} range {!r}; stop must be >= start".format(axis_name, token)
                )
            values = range(start, stop + 1)
        else:
            values = [int(token)]

        for value in values:
            if value < 0 or value >= upper_bound:
                raise ValueError(
                    "{} index {} out of range [0, {}]".format(
                        axis_name,
                        value,
                        upper_bound - 1,
                    )
                )
            if value not in seen:
                indices.append(value)
                seen.add(value)

    return indices


def parse_set_expression(expr_text):
    """Parse one command-line matrix edit expression."""

    raw = str(expr_text).strip()
    if "=" not in raw:
        raise ValueError(
            "Invalid --set expression {!r}; expected SIGNALS:BEAMS=VALUE".format(raw)
        )

    region_text, value_text = raw.rsplit("=", 1)
    if ":" not in region_text:
        raise ValueError(
            "Invalid --set expression {!r}; expected SIGNALS:BEAMS=VALUE".format(raw)
        )

    signal_text, beam_text = region_text.split(":", 1)
    signal_indices = parse_index_spec(signal_text, TOTAL_SIGNAL_INPUTS, "signal")
    beam_indices = parse_index_spec(beam_text, TOTAL_BEAMS, "beam")

    try:
        value = np.float32(float(value_text.strip()))
    except ValueError:
        raise ValueError(
            "Invalid value {!r} in --set expression {!r}".format(value_text.strip(), raw)
        )

    return {
        "raw": raw,
        "signal_indices": signal_indices,
        "beam_indices": beam_indices,
        "value": value,
    }


def should_run_cli_mode(args):
    """Return whether the request should execute in non-GUI mode."""

    return bool(
        args.write_default
        or args.input_path
        or (args.set_all_value is not None)
        or args.set_expressions
    )


def build_matrix_from_cli_args(args):
    """Build one matrix from command-line options without launching the GUI."""

    if args.input_path:
        input_path = os.path.abspath(os.path.expanduser(args.input_path))
        if not os.path.exists(input_path):
            raise ValueError("Input file does not exist: {}".format(input_path))
        matrix = load_matrix_from_file(input_path)
        source_label = input_path
    else:
        matrix = build_default_matrix()
        source_label = "built-in default"

    matrix = np.asarray(matrix, dtype=np.float32).copy()
    operation_summaries = []

    if args.set_all_value is not None:
        fill_value = np.float32(args.set_all_value)
        matrix[:, :] = fill_value
        operation_summaries.append(
            "Set all {} cell(s) to {}".format(
                TOTAL_SIGNAL_INPUTS * TOTAL_BEAMS,
                "{:.6f}".format(float(fill_value)),
            )
        )

    for expr_text in args.set_expressions:
        op = parse_set_expression(expr_text)
        matrix[np.ix_(op["signal_indices"], op["beam_indices"])] = op["value"]
        operation_summaries.append(
            "Applied {!r} to {} cell(s)".format(
                op["raw"],
                len(op["signal_indices"]) * len(op["beam_indices"]),
            )
        )

    return validate_matrix_shape(matrix), source_label, operation_summaries


def write_cli_outputs(args):
    """Apply command-line edits and save the resulting matrix."""

    matrix, source_label, operation_summaries = build_matrix_from_cli_args(args)
    ensure_parent_dir(args.output_path)
    save_matrix_to_file(args.output_path, matrix)
    md5_path, checksum = write_md5_file_for_output(args.output_path)
    ones_count = int(np.count_nonzero(matrix > 0.5))

    print("Matrix source            : {}".format(source_label))
    for summary in operation_summaries:
        print(summary)
    print("Active cells             : {} / {}".format(ones_count, TOTAL_SIGNAL_INPUTS * TOTAL_BEAMS))
    print("Saved matrix txt         : {}".format(args.output_path))
    print("Saved md5 sidecar        : {}".format(md5_path))
    print("MD5                      : {}".format(checksum))


def require_tkinter():
    """Fail with a friendly message when Tkinter is unavailable."""

    if TKINTER_IMPORT_ERROR is None:
        return

    raise SystemExit(
        "Tkinter is not available in this Python environment: {}. "
        "On Ubuntu, install a Tk-enabled Python such as the system package "
        "`python3-tk`, or run this script with `--write-default` or `--set` "
        "for a non-GUI save.".format(TKINTER_IMPORT_ERROR)
    )


def validate_matrix_shape(matrix):
    """Validate the fixed 20x32 matrix shape."""

    arr = np.asarray(matrix, dtype=np.float32)
    expected_shape = (TOTAL_SIGNAL_INPUTS, TOTAL_BEAMS)
    if arr.shape != expected_shape:
        raise ValueError("Expected matrix shape {}, got {}".format(expected_shape, arr.shape))
    return arr


def _load_text_matrix(path):
    """Load one legacy UTF-8 text matrix from disk.

    Supported text layouts:
    - 1 row x 640 values, matching beam_coeff_antenna_zhao.txt
    - 20 rows x 32 values, for compatibility with the previous text layout
    """

    rows = []
    with io.open(path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            rows.append([np.float32(float(token)) for token in parts])

    if len(rows) == 1:
        flat = np.asarray(rows[0], dtype=np.float32)
        expected_size = TOTAL_SIGNAL_INPUTS * TOTAL_BEAMS
        if flat.size != expected_size:
            raise ValueError(
                "Expected {} float values in the single text row of {}, got {}".format(
                    expected_size,
                    path,
                    flat.size,
                )
            )
        return flat.reshape((TOTAL_SIGNAL_INPUTS, TOTAL_BEAMS))

    if len(rows) == TOTAL_SIGNAL_INPUTS:
        for row_index, row in enumerate(rows):
            if len(row) != TOTAL_BEAMS:
                raise ValueError(
                    "Line {} in {} must contain exactly {} float values, got {}".format(
                        row_index + 1,
                        path,
                        TOTAL_BEAMS,
                        len(row),
                    )
                )
        return np.asarray(rows, dtype=np.float32)

    raise ValueError(
        "Unsupported text layout in {}: expected either 1 row x {} values or {} rows x {} values, got {} row(s).".format(
            path,
            TOTAL_SIGNAL_INPUTS * TOTAL_BEAMS,
            TOTAL_SIGNAL_INPUTS,
            TOTAL_BEAMS,
            len(rows),
        )
    )


def _load_binary_matrix(path):
    """Load one little-endian float32 20x32 binary matrix from disk."""

    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype="<f4")
    expected_size = TOTAL_SIGNAL_INPUTS * TOTAL_BEAMS
    if data.size != expected_size:
        raise ValueError(
            "Expected {} float32 values in {}, got {}".format(expected_size, path, data.size)
        )
    return data.reshape((TOTAL_SIGNAL_INPUTS, TOTAL_BEAMS)).astype(np.float32)


def _looks_like_binary_matrix_file(path):
    """Heuristically detect the fixed-size binary float32 output format."""

    expected_bytes = TOTAL_SIGNAL_INPUTS * TOTAL_BEAMS * np.dtype("<f4").itemsize
    if os.path.getsize(path) != expected_bytes:
        return False

    with open(path, "rb") as f:
        sample = f.read(min(256, expected_bytes))
    return b"\x00" in sample


def load_matrix_from_file(path):
    """Load either the current text format or one legacy binary format."""

    if _looks_like_binary_matrix_file(path):
        try:
            return _load_binary_matrix(path)
        except ValueError:
            pass

    try:
        return _load_text_matrix(path)
    except UnicodeDecodeError:
        return _load_binary_matrix(path)
    except ValueError as text_exc:
        try:
            return _load_binary_matrix(path)
        except ValueError:
            raise text_exc


def save_matrix_to_file(path, matrix):
    """Save a 20x32 matrix as one flattened UTF-8 text row."""

    matrix = validate_matrix_shape(matrix)
    tmp_path = path + ".tmp"
    payload = np.asarray(matrix, dtype=np.float32)
    with io.open(tmp_path, "w", encoding="utf-8") as f:
        flat = payload.reshape(-1)
        f.write(" ".join("{:.6f}".format(float(value)) for value in flat))
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


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


def write_md5_file_for_output(path):
    """Write one UTF-8 .md5 sidecar for the requested output file."""

    md5_path = build_md5_output_path(path)
    checksum = compute_file_md5(path)
    tmp_path = md5_path + ".tmp"
    with io.open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("{}  {}\n".format(checksum, os.path.basename(path)))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, md5_path)
    return md5_path, checksum


class BeamCoeffEditor:
    """Tk editor for the 20x32 beam coefficient enable matrix."""

    def __init__(self, root, initial_output_path=DEFAULT_OUTPUT_PATH):
        self.root = root
        self.root.title("beam_coeff_antenna editor")
        self.root.minsize(1280, 760)

        self.output_path_var = tk.StringVar(value=initial_output_path)
        self.status_var = tk.StringVar(value="Ready")
        self.matrix = build_default_matrix()

        self.signal_listbox = None
        self.beam_listbox = None
        self.canvas = None

        self._build_layout()
        self._populate_selectors()
        self._redraw_canvas()
        self._update_status("Loaded default 20x32 matrix")

    def _build_layout(self):
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill="both", expand=True)

        top = ttk.LabelFrame(main, text="File", padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="Output path").grid(row=0, column=0, sticky="w", padx=(0, 10))
        ttk.Entry(top, textvariable=self.output_path_var, width=110).grid(row=0, column=1, sticky="ew")
        ttk.Button(top, text="Load", command=self._load_matrix).grid(row=0, column=2, padx=(10, 0))
        ttk.Button(top, text="Save", command=self._save_matrix).grid(row=0, column=3, padx=(8, 0))
        top.columnconfigure(1, weight=1)

        body = ttk.Frame(main, padding=(0, 12, 0, 0))
        body.pack(fill="both", expand=True)

        controls = ttk.LabelFrame(body, text="Batch Controls", padding=10)
        controls.pack(side="left", fill="y")

        ttk.Label(
            controls,
            text="Choose one or more signal rows and one or more beam columns,\nthen set the selected cross-region to 0 or 1.",
            justify="left",
        ).pack(anchor="w")

        lists_row = ttk.Frame(controls)
        lists_row.pack(fill="x", pady=(12, 0))

        signal_frame = ttk.Frame(lists_row)
        signal_frame.pack(side="left", fill="y")
        ttk.Label(signal_frame, text="Signal inputs i").pack(anchor="w")
        self.signal_listbox = tk.Listbox(
            signal_frame,
            selectmode="extended",
            exportselection=False,
            width=16,
            height=20,
        )
        self.signal_listbox.pack(fill="y", expand=False)

        beam_frame = ttk.Frame(lists_row)
        beam_frame.pack(side="left", fill="y", padx=(12, 0))
        ttk.Label(beam_frame, text="Beams j").pack(anchor="w")
        self.beam_listbox = tk.Listbox(
            beam_frame,
            selectmode="extended",
            exportselection=False,
            width=16,
            height=20,
        )
        self.beam_listbox.pack(fill="y", expand=False)

        selection_buttons = ttk.Frame(controls)
        selection_buttons.pack(fill="x", pady=(12, 0))

        ttk.Button(selection_buttons, text="Signals 0-7", command=self._select_default_active_signals).pack(fill="x")
        ttk.Button(selection_buttons, text="All signals", command=self._select_all_signals).pack(fill="x", pady=(6, 0))
        ttk.Button(selection_buttons, text="Clear signals", command=self._clear_signal_selection).pack(fill="x", pady=(6, 0))
        ttk.Button(selection_buttons, text="All beams", command=self._select_all_beams).pack(fill="x", pady=(12, 0))
        ttk.Button(selection_buttons, text="Clear beams", command=self._clear_beam_selection).pack(fill="x", pady=(6, 0))

        edit_buttons = ttk.Frame(controls)
        edit_buttons.pack(fill="x", pady=(16, 0))

        ttk.Button(edit_buttons, text="Set selected -> 1", command=lambda: self._apply_selection(1.0)).pack(fill="x")
        ttk.Button(edit_buttons, text="Set selected -> 0", command=lambda: self._apply_selection(0.0)).pack(fill="x", pady=(6, 0))
        ttk.Button(edit_buttons, text="Toggle selected", command=self._toggle_selection).pack(fill="x", pady=(6, 0))

        matrix_buttons = ttk.Frame(controls)
        matrix_buttons.pack(fill="x", pady=(16, 0))

        ttk.Button(matrix_buttons, text="Reset default 8x32 on", command=self._reset_default).pack(fill="x")
        ttk.Button(matrix_buttons, text="All matrix -> 1", command=lambda: self._fill_all(1.0)).pack(fill="x", pady=(6, 0))
        ttk.Button(matrix_buttons, text="All matrix -> 0", command=lambda: self._fill_all(0.0)).pack(fill="x", pady=(6, 0))

        ttk.Label(
            controls,
            text=(
                "Format reminder:\n"
                "20 x 32 matrix in memory\n"
                "Saved as 1 text row x 640 values\n"
                "Order: input0 beam0..31, ..., input19 beam0..31\n"
                "Click any cell on the right to toggle it."
            ),
            justify="left",
        ).pack(anchor="w", pady=(16, 0))

        canvas_frame = ttk.LabelFrame(body, text="20 x 32 Matrix M[i, j]", padding=10)
        canvas_frame.pack(side="left", fill="both", expand=True, padx=(12, 0))

        self.canvas = tk.Canvas(
            canvas_frame,
            bg=GRID_BG,
            highlightthickness=0,
            width=GRID_LEFT + TOTAL_BEAMS * CELL_WIDTH + 40,
            height=GRID_TOP + TOTAL_SIGNAL_INPUTS * CELL_HEIGHT + 40,
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self._handle_canvas_click)

        status_row = ttk.Frame(main, padding=(0, 10, 0, 0))
        status_row.pack(fill="x")
        ttk.Label(status_row, textvariable=self.status_var).pack(side="left")

    def _populate_selectors(self):
        self.signal_listbox.delete(0, "end")
        self.beam_listbox.delete(0, "end")

        for signal_index in range(TOTAL_SIGNAL_INPUTS):
            tag = "active" if signal_index < DEFAULT_ACTIVE_SIGNAL_INPUTS else "unused"
            self.signal_listbox.insert("end", "Signal {:02d} ({})".format(signal_index, tag))

        for beam_index in range(TOTAL_BEAMS):
            self.beam_listbox.insert("end", "Beam {:02d}".format(beam_index))

    def _selected_signal_indices(self):
        return [int(index) for index in self.signal_listbox.curselection()]

    def _selected_beam_indices(self):
        return [int(index) for index in self.beam_listbox.curselection()]

    def _select_default_active_signals(self):
        self.signal_listbox.selection_clear(0, "end")
        for signal_index in range(DEFAULT_ACTIVE_SIGNAL_INPUTS):
            self.signal_listbox.selection_set(signal_index)
        self._update_status("Selected signal rows 0..7")
        self._redraw_canvas()

    def _select_all_signals(self):
        self.signal_listbox.selection_set(0, "end")
        self._update_status("Selected all signal rows")
        self._redraw_canvas()

    def _clear_signal_selection(self):
        self.signal_listbox.selection_clear(0, "end")
        self._update_status("Cleared signal row selection")
        self._redraw_canvas()

    def _select_all_beams(self):
        self.beam_listbox.selection_set(0, "end")
        self._update_status("Selected all beam columns")
        self._redraw_canvas()

    def _clear_beam_selection(self):
        self.beam_listbox.selection_clear(0, "end")
        self._update_status("Cleared beam column selection")
        self._redraw_canvas()

    def _apply_selection(self, value):
        signal_indices = self._selected_signal_indices()
        beam_indices = self._selected_beam_indices()
        if (not signal_indices) or (not beam_indices):
            messagebox.showwarning("Selection required", "Please select at least one signal row and one beam column.")
            return

        self.matrix[np.ix_(signal_indices, beam_indices)] = np.float32(value)
        self._redraw_canvas()
        self._update_status(
            "Set {} selected cell(s) to {:.0f}".format(len(signal_indices) * len(beam_indices), value)
        )

    def _toggle_selection(self):
        signal_indices = self._selected_signal_indices()
        beam_indices = self._selected_beam_indices()
        if (not signal_indices) or (not beam_indices):
            messagebox.showwarning("Selection required", "Please select at least one signal row and one beam column.")
            return

        region = self.matrix[np.ix_(signal_indices, beam_indices)]
        self.matrix[np.ix_(signal_indices, beam_indices)] = np.where(region > 0.5, 0.0, 1.0).astype(np.float32)
        self._redraw_canvas()
        self._update_status("Toggled {} selected cell(s)".format(len(signal_indices) * len(beam_indices)))

    def _reset_default(self):
        self.matrix = build_default_matrix()
        self._redraw_canvas()
        self._update_status("Reset matrix to default: rows 0..7 on, rows 8..19 off")

    def _fill_all(self, value):
        self.matrix[:, :] = np.float32(value)
        self._redraw_canvas()
        self._update_status("Set all {} cells to {:.0f}".format(TOTAL_SIGNAL_INPUTS * TOTAL_BEAMS, value))

    def _load_matrix(self):
        path = self.output_path_var.get().strip()
        if not path:
            messagebox.showwarning("Missing path", "Please enter a file path.")
            return
        if not os.path.exists(path):
            messagebox.showwarning("File not found", "{} does not exist.".format(path))
            return

        try:
            self.matrix = load_matrix_from_file(path)
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc))
            return

        self._redraw_canvas()
        self._update_status("Loaded matrix from {}".format(path))

    def _save_matrix(self):
        path = self.output_path_var.get().strip()
        if not path:
            messagebox.showwarning("Missing path", "Please enter a file path.")
            return

        ensure_parent_dir(path)

        try:
            save_matrix_to_file(path, self.matrix)
            md5_path, checksum = write_md5_file_for_output(path)
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return

        self._update_status(
            "Saved txt to {} and md5 to {} ({})".format(path, md5_path, checksum)
        )

    def _handle_canvas_click(self, event):
        row_index = int((event.y - GRID_TOP) // CELL_HEIGHT)
        col_index = int((event.x - GRID_LEFT) // CELL_WIDTH)
        if row_index < 0 or row_index >= TOTAL_SIGNAL_INPUTS:
            return
        if col_index < 0 or col_index >= TOTAL_BEAMS:
            return

        self.matrix[row_index, col_index] = np.float32(0.0 if self.matrix[row_index, col_index] > 0.5 else 1.0)
        self._redraw_canvas()
        self._update_status(
            "Toggled M[{:02d}, {:02d}] -> {:.0f}".format(
                row_index,
                col_index,
                float(self.matrix[row_index, col_index]),
            )
        )

    def _cell_selected(self, row_index, col_index):
        return (row_index in self._selected_signal_indices()) and (col_index in self._selected_beam_indices())

    def _redraw_canvas(self):
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, self.canvas.winfo_width(), self.canvas.winfo_height(), fill=GRID_BG, outline="")

        self.canvas.create_text(
            GRID_LEFT + (TOTAL_BEAMS * CELL_WIDTH) / 2.0,
            14,
            text="Beam index j",
            fill=HEADER_TEXT,
            font=("Consolas", 11, "bold"),
        )
        self.canvas.create_text(
            16,
            GRID_TOP + (TOTAL_SIGNAL_INPUTS * CELL_HEIGHT) / 2.0,
            text="Signal i",
            angle=90,
            fill=HEADER_TEXT,
            font=("Consolas", 11, "bold"),
        )

        for col_index in range(TOTAL_BEAMS):
            x0 = GRID_LEFT + col_index * CELL_WIDTH
            x1 = x0 + CELL_WIDTH
            self.canvas.create_text(
                (x0 + x1) / 2.0,
                GRID_TOP - 14,
                text=str(col_index),
                fill=HEADER_TEXT,
                font=("Consolas", 9),
            )

        for row_index in range(TOTAL_SIGNAL_INPUTS):
            y0 = GRID_TOP + row_index * CELL_HEIGHT
            y1 = y0 + CELL_HEIGHT
            self.canvas.create_text(
                GRID_LEFT - 20,
                (y0 + y1) / 2.0,
                text=str(row_index),
                fill=HEADER_TEXT,
                font=("Consolas", 9),
            )

            for col_index in range(TOTAL_BEAMS):
                x0 = GRID_LEFT + col_index * CELL_WIDTH
                x1 = x0 + CELL_WIDTH
                value = float(self.matrix[row_index, col_index])
                is_selected = self._cell_selected(row_index, col_index)
                fill = CELL_ON if value > 0.5 else CELL_OFF
                outline = SELECTION_OUTLINE if is_selected else CELL_GRID
                line_width = 2 if is_selected else 1

                self.canvas.create_rectangle(
                    x0,
                    y0,
                    x1,
                    y1,
                    fill=fill,
                    outline=outline,
                    width=line_width,
                )
                self.canvas.create_text(
                    (x0 + x1) / 2.0,
                    (y0 + y1) / 2.0,
                    text="1" if value > 0.5 else "0",
                    fill=CELL_TEXT,
                    font=("Consolas", 9, "bold"),
                )

        ones_count = int(np.count_nonzero(self.matrix > 0.5))
        self.canvas.create_text(
            GRID_LEFT,
            GRID_TOP + TOTAL_SIGNAL_INPUTS * CELL_HEIGHT + 24,
            anchor="w",
            text="Active cells = {} / {}".format(ones_count, TOTAL_SIGNAL_INPUTS * TOTAL_BEAMS),
            fill=HEADER_TEXT,
            font=("Consolas", 10),
        )

    def _update_status(self, text):
        self.status_var.set(text)


def main():
    args = parse_args()

    if args.write_default:
        path, md5_path, checksum = write_default_outputs(args.output_path)
        print("Saved default matrix txt : {}".format(path))
        print("Saved md5 sidecar        : {}".format(md5_path))
        print("MD5                      : {}".format(checksum))
        return

    if should_run_cli_mode(args):
        write_cli_outputs(args)
        return

    require_tkinter()
    root = tk.Tk()
    BeamCoeffEditor(root, initial_output_path=args.output_path)
    root.mainloop()


if __name__ == "__main__":
    main()
