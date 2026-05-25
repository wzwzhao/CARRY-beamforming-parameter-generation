#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interactive editor for beam_coeff_antenna.txt.

The output format matches the publisher:
- matrix shape: 20 x 32
- dtype: little-endian float32
- storage order: C-order row-major

Matrix meaning:
- rows    -> signal input index i
- columns -> beam index j
- value   -> M[i, j], typically 0.0 or 1.0
"""

from __future__ import division, print_function

import os
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_PATH = os.path.join(SCRIPT_DIR, "beam_coeff_antenna.txt")

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


def validate_matrix_shape(matrix):
    """Validate the fixed 20x32 matrix shape."""

    arr = np.asarray(matrix, dtype=np.float32)
    expected_shape = (TOTAL_SIGNAL_INPUTS, TOTAL_BEAMS)
    if arr.shape != expected_shape:
        raise ValueError("Expected matrix shape {}, got {}".format(expected_shape, arr.shape))
    return arr


def load_matrix_from_file(path):
    """Load a little-endian float32 20x32 matrix from disk."""

    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype="<f4")
    expected_size = TOTAL_SIGNAL_INPUTS * TOTAL_BEAMS
    if data.size != expected_size:
        raise ValueError(
            "Expected {} float32 values in {}, got {}".format(expected_size, path, data.size)
        )
    return data.reshape((TOTAL_SIGNAL_INPUTS, TOTAL_BEAMS)).astype(np.float32)


def save_matrix_to_file(path, matrix):
    """Save a 20x32 matrix using the publisher's binary float32 format."""

    matrix = validate_matrix_shape(matrix)
    tmp_path = path + ".tmp"
    payload = np.asarray(matrix, dtype="<f4")
    with open(tmp_path, "wb") as f:
        f.write(payload.tobytes(order="C"))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


class BeamCoeffEditor:
    """Tk editor for the 20x32 beam coefficient enable matrix."""

    def __init__(self, root):
        self.root = root
        self.root.title("beam_coeff_antenna editor")
        self.root.minsize(1280, 760)

        self.output_path_var = tk.StringVar(value=DEFAULT_OUTPUT_PATH)
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

        ttk.Label(top, text="beam_coeff_antenna.txt").grid(row=0, column=0, sticky="w", padx=(0, 10))
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
                "20 rows = signal inputs i\n"
                "32 cols = beam indices j\n"
                "Saved as little-endian float32\n"
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

        parent_dir = os.path.dirname(path)
        if parent_dir and (not os.path.exists(parent_dir)):
            os.makedirs(parent_dir)

        try:
            save_matrix_to_file(path, self.matrix)
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return

        self._update_status("Saved 20x32 float32 matrix to {}".format(path))

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
    root = tk.Tk()
    BeamCoeffEditor(root)
    root.mainloop()


if __name__ == "__main__":
    main()
