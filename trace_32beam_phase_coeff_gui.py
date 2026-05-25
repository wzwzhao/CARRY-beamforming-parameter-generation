#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI preview tool for compute_trace_mode_phase_coeff.py.

This GUI does not publish parameters by itself.
It previews source visibility, next slot timing, external 32-beam offsets,
current beam az/el, active/padded input channels, and the output status of
trace_mode_phase_coeff.txt.
"""

from __future__ import division, print_function

import math
import os
import subprocess
import sys
import tkinter as tk
import traceback
from datetime import datetime, timedelta
from tkinter import ttk

import compute_trace_mode_phase_coeff as phase


DEFAULT_REFRESH_INTERVAL_SECONDS = 1.0
ANIMATION_INTERVAL_MS = 50
SCENE_EASE = 0.14

VISUAL_CANVAS_MIN_WIDTH = 520
VISUAL_CANVAS_MIN_HEIGHT = 460
BEAM_CANVAS_MIN_WIDTH = 760
BEAM_CANVAS_MIN_HEIGHT = 560


class PhaseCoeffStatusGUI:
    """Tkinter GUI for previewing trace-mode source status and output health."""

    VISIBILITY_PREFIX_MAP = {
        "  visibility status      :": "  \u53ef\u89c1\u6027 visibility status                 :",
        "  current window start   :": "  \u5f53\u524d\u7a97\u53e3\u5f00\u59cb current window start        :",
        "  current window end     :": "  \u5f53\u524d\u7a97\u53e3\u7ed3\u675f current window end          :",
        "  next visible start     :": "  \u4e0b\u6b21\u53ef\u89c1\u5f00\u59cb next visible start          :",
        "  next visible end       :": "  \u4e0b\u6b21\u53ef\u89c1\u7ed3\u675f next visible end            :",
        "  next transit           :": "  \u4e0b\u6b21\u8fc7\u4e2d\u5929 next transit                  :",
        "  visibility window      :": "  \u53ef\u89c1\u7a97\u53e3 visibility window               :",
    }

    def __init__(self, root):
        self.root = root
        self.root.title("Trace Preview | compute_trace_mode_phase_coeff.py")
        self.root.minsize(1280, 860)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.refresh_after_id = None
        self.animation_after_id = None
        self.scene_payload = None
        self.current_focus_point = None
        self.target_focus_point = None
        self.preview_beam_model = None
        self.preview_beam_model_signature = None
        self.beam_layout_payload = None
        self.beam_layout_click_targets = []
        self.selected_beam_index = 0

        self.beam_detail_var = tk.StringVar(
            value="\u70b9\u51fb\u4efb\u610f\u6ce2\u675f\u67e5\u770b file BeamID, beam_index, and current az/el."
        )
        self.input_mapping_var = tk.StringVar(value="\u8f93\u5165\u901a\u9053\u6620\u5c04\u5c1a\u672a\u68c0\u67e5")
        self.beam_offset_status_var = tk.StringVar(value="32-beam offset file not checked")
        self.trace_file_status_var = tk.StringVar(value="trace_mode_phase_coeff.txt not checked")

        self.ants_txt_var = tk.StringVar(value=phase.ANTS_TXT)
        self.target_ra_var = tk.StringVar(value=phase.TARGET_RA)
        self.target_dec_var = tk.StringVar(value=phase.TARGET_DEC)
        self.min_elevation_var = tk.StringVar(value=str(phase.MIN_ELEVATION_DEG))
        self.simulation_ignore_visibility_var = tk.BooleanVar(
            value=phase.SIMULATION_IGNORE_VISIBILITY
        )
        self.refresh_interval_var = tk.StringVar(value=str(DEFAULT_REFRESH_INTERVAL_SECONDS))
        self.auto_refresh_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Ready")

        self._build_layout()
        self._set_output_text("Refresh log and tracebacks will appear here.")
        self._schedule_animation()
        self.refresh_now()

    def _build_layout(self):
        main = ttk.Frame(self.root, padding=8)
        main.pack(fill="both", expand=True)

        controls = ttk.LabelFrame(main, text="\u8f93\u5165\u53c2\u6570 Inputs", padding=8)
        controls.pack(fill="x")
        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(3, weight=1)

        ttk.Label(controls, text="Antenna file").grid(row=0, column=0, sticky="w", pady=2, padx=(0, 8))
        ttk.Entry(controls, textvariable=self.ants_txt_var, width=96).grid(
            row=0, column=1, columnspan=3, sticky="ew", pady=2
        )

        ttk.Label(controls, text="Target RA").grid(row=1, column=0, sticky="w", pady=2, padx=(0, 8))
        ttk.Entry(controls, textvariable=self.target_ra_var, width=22).grid(
            row=1, column=1, sticky="ew", pady=2, padx=(0, 12)
        )
        ttk.Label(controls, text="Target Dec").grid(row=1, column=2, sticky="w", pady=2, padx=(0, 8))
        ttk.Entry(controls, textvariable=self.target_dec_var, width=22).grid(
            row=1, column=3, sticky="ew", pady=2
        )

        ttk.Label(controls, text="Min elevation (deg)").grid(row=2, column=0, sticky="w", pady=2, padx=(0, 8))
        ttk.Entry(controls, textvariable=self.min_elevation_var, width=22).grid(
            row=2, column=1, sticky="ew", pady=2, padx=(0, 12)
        )
        ttk.Label(controls, text="Refresh interval (s)").grid(
            row=2, column=2, sticky="w", pady=2, padx=(0, 8)
        )
        ttk.Entry(controls, textvariable=self.refresh_interval_var, width=22).grid(
            row=2, column=3, sticky="ew", pady=2
        )

        ttk.Checkbutton(
            controls,
            text="\u6a21\u62df\u65f6\u5ffd\u7565\u4e0d\u53ef\u89c1 Simulation ignore visibility",
            variable=self.simulation_ignore_visibility_var,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Checkbutton(
            controls,
            text="Auto refresh",
            variable=self.auto_refresh_var,
            command=self._handle_auto_refresh_toggle,
        ).grid(row=3, column=2, columnspan=2, sticky="w", pady=(4, 0))

        button_row = ttk.Frame(controls)
        button_row.grid(row=4, column=0, columnspan=4, sticky="w", pady=(8, 0))
        ttk.Button(button_row, text="Refresh now", command=self.refresh_now).pack(side="left")
        ttk.Button(button_row, text="Stop auto", command=self.stop_auto_refresh).pack(side="left", padx=(8, 0))
        ttk.Button(button_row, text="Start auto", command=self.start_auto_refresh).pack(side="left", padx=(8, 0))
        ttk.Button(button_row, text="Start publisher", command=self.start_publisher).pack(
            side="left", padx=(8, 0)
        )

        status_row = ttk.Frame(main, padding=(0, 6, 0, 0))
        status_row.pack(fill="x")
        ttk.Label(status_row, textvariable=self.status_var).pack(side="left")

        self.notebook = ttk.Notebook(main)
        self.notebook.pack(fill="both", expand=True, pady=(8, 0))

        status_tab = ttk.Frame(self.notebook, padding=8)
        beam_tab = ttk.Frame(self.notebook, padding=8)
        output_tab = ttk.Frame(self.notebook, padding=8)
        self.log_tab = ttk.Frame(self.notebook, padding=8)

        self.notebook.add(status_tab, text="\u72b6\u6001 Status")
        self.notebook.add(beam_tab, text="32\u6ce2\u675f Beam layout")
        self.notebook.add(output_tab, text="\u8f93\u51fa Output")
        self.notebook.add(self.log_tab, text="\u65e5\u5fd7 Log")

        status_pane = ttk.PanedWindow(status_tab, orient="horizontal")
        status_pane.pack(fill="both", expand=True)

        left_status = ttk.Frame(status_pane)
        right_status = ttk.Frame(status_pane)
        status_pane.add(left_status, weight=3)
        status_pane.add(right_status, weight=2)

        left_pane = ttk.PanedWindow(left_status, orient="vertical")
        left_pane.pack(fill="both", expand=True)

        current_frame = ttk.LabelFrame(
            left_pane,
            text="\u5f53\u524d\u6e90\u72b6\u6001 Current source status",
            padding=8,
        )
        current_text_frame, self.current_status_text = self._make_scrolled_text(current_frame, height=12)
        current_text_frame.pack(fill="both", expand=True)
        left_pane.add(current_frame, weight=1)

        next_frame = ttk.LabelFrame(
            left_pane,
            text="\u4e0b\u4e00\u69fd\u4f4d\u9884\u89c8 Next slot preview",
            padding=8,
        )
        next_text_frame, self.next_slot_text = self._make_scrolled_text(next_frame, height=12)
        next_text_frame.pack(fill="both", expand=True)
        left_pane.add(next_frame, weight=1)

        visual_frame = ttk.LabelFrame(
            right_status,
            text="\u671b\u8fdc\u955c\u793a\u610f Telescope view",
            padding=8,
        )
        visual_frame.pack(fill="both", expand=True)

        self.visual_canvas = tk.Canvas(
            visual_frame,
            width=VISUAL_CANVAS_MIN_WIDTH,
            height=VISUAL_CANVAS_MIN_HEIGHT,
            bg="#08131f",
            highlightthickness=0,
        )
        self.visual_canvas.pack(fill="both", expand=True)
        self.visual_canvas.bind("<Configure>", lambda event: self._redraw_visual_scene())

        beam_canvas_frame = ttk.LabelFrame(
            beam_tab,
            text="32\u6ce2\u675f\u6392\u5e03 32-beam layout",
            padding=8,
        )
        beam_canvas_frame.pack(fill="both", expand=True)

        self.beam_layout_canvas = tk.Canvas(
            beam_canvas_frame,
            width=BEAM_CANVAS_MIN_WIDTH,
            height=BEAM_CANVAS_MIN_HEIGHT,
            bg="#101722",
            highlightthickness=0,
        )
        self.beam_layout_canvas.pack(fill="both", expand=True)
        self.beam_layout_canvas.bind("<Button-1>", self._handle_beam_layout_click)
        self.beam_layout_canvas.bind("<Configure>", lambda event: self._redraw_beam_layout())

        detail_frame = ttk.LabelFrame(beam_tab, text="Beam detail", padding=8)
        detail_frame.pack(fill="x", pady=(8, 0))
        ttk.Label(
            detail_frame,
            textvariable=self.beam_detail_var,
            justify="left",
            wraplength=900,
        ).pack(fill="x")

        self._build_status_label_frame(
            output_tab,
            "Input mapping",
            self.input_mapping_var,
            wraplength=900,
        ).pack(fill="x")
        self._build_status_label_frame(
            output_tab,
            "Beam offset file",
            self.beam_offset_status_var,
            wraplength=900,
        ).pack(fill="x", pady=(8, 0))
        self._build_status_label_frame(
            output_tab,
            "Trace output file",
            self.trace_file_status_var,
            wraplength=900,
        ).pack(fill="x", pady=(8, 0))

        log_frame = ttk.LabelFrame(self.log_tab, text="Refresh log and errors", padding=8)
        log_frame.pack(fill="both", expand=True)
        log_text_frame, self.output_text = self._make_scrolled_text(log_frame, height=18)
        log_text_frame.pack(fill="both", expand=True)

    def _make_scrolled_text(self, parent, height=12):
        frame = ttk.Frame(parent)
        text = tk.Text(
            frame,
            wrap="word",
            height=height,
            font=("Consolas", 10),
            padx=10,
            pady=8,
        )
        yscroll = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=yscroll.set)
        text.pack(side="left", fill="both", expand=True)
        yscroll.pack(side="right", fill="y")
        text.configure(state="disabled")
        return frame, text

    def _build_status_label_frame(self, parent, title, variable, wraplength=400):
        frame = ttk.LabelFrame(parent, text=title, padding=8)
        ttk.Label(
            frame,
            textvariable=variable,
            justify="left",
            anchor="w",
            wraplength=wraplength,
            font=("Consolas", 10),
        ).pack(fill="x")
        return frame

    def _set_text_widget(self, widget, text):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def _set_output_text(self, text):
        self._set_text_widget(self.output_text, text)

    def _build_refresh_log_text(self, now_utc, current_snapshot, next_slot_snapshot, simulation_ignore_visibility):
        current_publish = self._compute_would_publish(current_snapshot, simulation_ignore_visibility)
        next_publish = self._compute_would_publish(next_slot_snapshot, simulation_ignore_visibility)
        return "\n".join(
            [
                "Refresh OK",
                "",
                "refresh_utc              : {}".format(now_utc.strftime("%Y-%m-%d %H:%M:%S")),
                "current_visible_now      : {}".format(current_snapshot["visible_now"]),
                "current_would_publish    : {}".format(current_publish),
                "next_slot_visible_now    : {}".format(next_slot_snapshot["visible_now"]),
                "next_slot_would_publish  : {}".format(next_publish),
                "trace_output_file        : {}".format(phase.TRACE_MODE_PHASE_TXT),
            ]
        )

    def _handle_auto_refresh_toggle(self):
        if self.auto_refresh_var.get():
            self.start_auto_refresh()
        else:
            self.stop_auto_refresh()

    def _on_close(self):
        self._cancel_scheduled_refresh()
        self._cancel_animation()
        self.root.destroy()

    def _cancel_scheduled_refresh(self):
        if self.refresh_after_id is not None:
            self.root.after_cancel(self.refresh_after_id)
            self.refresh_after_id = None

    def _cancel_animation(self):
        if self.animation_after_id is not None:
            self.root.after_cancel(self.animation_after_id)
            self.animation_after_id = None

    def _schedule_refresh(self):
        self._cancel_scheduled_refresh()
        interval_ms = max(100, int(round(self._get_refresh_interval_seconds() * 1000.0)))
        self.refresh_after_id = self.root.after(interval_ms, self.refresh_now)

    def _schedule_animation(self):
        self._cancel_animation()
        self.animation_after_id = self.root.after(ANIMATION_INTERVAL_MS, self._animation_tick)

    def stop_auto_refresh(self):
        self._cancel_scheduled_refresh()
        self.auto_refresh_var.set(False)

    def start_auto_refresh(self):
        self.auto_refresh_var.set(True)
        self._schedule_refresh()

    def start_publisher(self):
        script_path = os.path.join(phase.SCRIPT_DIR, "compute_trace_mode_phase_coeff.py")
        try:
            subprocess.Popen([sys.executable, script_path], cwd=phase.SCRIPT_DIR)
            self.status_var.set("Publisher started")
        except Exception as exc:
            self.status_var.set("Failed to start publisher: {}".format(exc))

    def _get_refresh_interval_seconds(self):
        interval = float(self.refresh_interval_var.get())
        if interval <= 0.0:
            raise ValueError("Refresh interval must be > 0")
        return interval

    def _collect_config(self):
        return {
            "ants_txt": self.ants_txt_var.get().strip(),
            "target_ra": self.target_ra_var.get().strip(),
            "target_dec": self.target_dec_var.get().strip(),
            "min_elevation_deg": float(self.min_elevation_var.get()),
            "simulation_ignore_visibility": bool(self.simulation_ignore_visibility_var.get()),
        }

    def _build_antenna_signature(self, antennas):
        return tuple(
            (
                antenna.name,
                round(float(antenna.lat_deg), 8),
                round(float(antenna.lon_deg), 8),
                round(float(antenna.height_m), 3),
                round(float(antenna.diameter_m), 3),
            )
            for antenna in antennas
        )

    def _get_beam_offset_file_signature(self):
        path = phase.BEAM_OFFSET_TABLE_FILE
        if not path:
            return None
        try:
            return (path, os.path.getmtime(path), os.path.getsize(path))
        except OSError:
            return (path, None, None)

    def _build_preview_beam_model_signature(self, config, antennas):
        return (
            config["ants_txt"],
            config["target_ra"],
            config["target_dec"],
            round(float(phase.CENTER_FREQ_HZ), 3),
            self._build_antenna_signature(antennas),
            self._get_beam_offset_file_signature(),
        )

    def _build_preview_session(self, anchor_utc):
        return phase.SessionContext(
            session_id="gui-preview",
            program_start_local=anchor_utc.astimezone(),
            program_start_utc=anchor_utc,
            t_ref_utc=anchor_utc,
            update_period_seconds=phase.UPDATE_PERIOD_SECONDS,
            publish_lead_seconds=phase.PUBLISH_LEAD_SECONDS,
        )

    def _build_beam_offset_metrics(self, beam_offset_rows, beam_offset_meta):
        metrics = dict(beam_offset_meta.get("primary_beam_validation") or {})
        if beam_offset_rows:
            nonzero_offsets = sorted(
                {
                    round(float(row["offset_deg"]), 10)
                    for row in beam_offset_rows
                    if float(row["offset_deg"]) > 0.0
                }
            )
            metrics.setdefault("spacing_deg", None if not nonzero_offsets else float(nonzero_offsets[0]))
            metrics.setdefault("max_offset_deg", max(float(row["offset_deg"]) for row in beam_offset_rows))
        return metrics

    def _build_preview_beam_model(self, config, antennas, anchor_utc):
        katpoint_antennas, antenna_names = phase.build_katpoint_antennas_from_records(antennas)
        katpoint_target = phase.build_katpoint_target_from_radec(
            config["target_ra"],
            config["target_dec"],
        )

        preview_session = self._build_preview_session(anchor_utc)
        beam_offset_rows, beam_offset_meta = phase.resolve_beam_offset_table(
            preview_session,
            antennas,
        )

        return {
            "anchor_utc": anchor_utc,
            "katpoint_target": katpoint_target,
            "katpoint_antennas": katpoint_antennas,
            "antenna_names": antenna_names,
            "beam_offset_rows": beam_offset_rows,
            "beam_offset_meta": beam_offset_meta,
            "beam_offset_metrics": self._build_beam_offset_metrics(beam_offset_rows, beam_offset_meta),
        }

    def _get_or_build_preview_beam_model(self, config, antennas, anchor_utc):
        signature = self._build_preview_beam_model_signature(config, antennas)
        if (self.preview_beam_model is None) or (signature != self.preview_beam_model_signature):
            self.preview_beam_model = self._build_preview_beam_model(config, antennas, anchor_utc)
            self.preview_beam_model_signature = signature
            self.selected_beam_index = 0
        return self.preview_beam_model

    def _merge_beam_row_metadata(self, result_rows, offset_rows):
        metadata_by_beam = {int(row["beam_id"]): row for row in offset_rows}
        merged_rows = []
        for row in result_rows:
            beam_id = int(row["beam_id"])
            merged = dict(row)
            meta = metadata_by_beam.get(beam_id, {})
            merged["file_beam_id"] = int(meta.get("file_beam_id", beam_id + 1))
            merged["beam_index"] = int(meta.get("beam_index", beam_id))
            merged["q"] = int(meta.get("q", row.get("q", 0)))
            merged["r"] = int(meta.get("r", row.get("r", 0)))
            merged_rows.append(merged)
        return merged_rows

    def _build_beam_layout_payload(self, preview_beam_model, epoch_utc):
        antenna_beam_results = phase.collect_antenna_beam_results(
            preview_beam_model["katpoint_target"],
            preview_beam_model["katpoint_antennas"],
            preview_beam_model["antenna_names"],
            epoch_utc,
            preview_beam_model["beam_offset_rows"],
        )
        reference_result = antenna_beam_results[0]
        rows = self._merge_beam_row_metadata(
            reference_result["rows"],
            preview_beam_model["beam_offset_rows"],
        )
        unique_rings = sorted(
            {
                round(float(row["offset_deg"]), 10)
                for row in preview_beam_model["beam_offset_rows"]
                if float(row["offset_deg"]) > 0.0
            }
        )
        metrics = dict(preview_beam_model["beam_offset_metrics"] or {})
        primary_beam_radius_deg = metrics.get("primary_beam_fwhm_radius_deg")
        max_offset_deg = metrics.get(
            "max_offset_deg",
            max(float(row["offset_deg"]) for row in rows),
        )

        if primary_beam_radius_deg is not None and primary_beam_radius_deg > 0.0:
            display_radius_deg = max(float(primary_beam_radius_deg), float(max_offset_deg) * 1.10, 1.0e-6)
            if display_radius_deg > float(primary_beam_radius_deg):
                scale_mode = "max(primary-beam radius, current max offset)"
            else:
                scale_mode = "single-dish primary-beam FWHM radius"
        else:
            display_radius_deg = max(float(max_offset_deg) * 1.10, 1.0e-6)
            scale_mode = "fallback from current max offset"

        return {
            "epoch_utc": epoch_utc,
            "anchor_utc": preview_beam_model["anchor_utc"],
            "antenna_name": reference_result["antenna_name"],
            "rows": rows,
            "beam_offset_meta": preview_beam_model["beam_offset_meta"],
            "beam_offset_metrics": metrics,
            "unique_rings": unique_rings,
            "max_offset_deg": max_offset_deg,
            "primary_beam_radius_deg": primary_beam_radius_deg,
            "display_radius_deg": display_radius_deg,
            "scale_mode": scale_mode,
        }

    def _layout_offset_to_canvas(self, d_east_deg, d_north_deg, display_radius_deg):
        width = max(int(self.beam_layout_canvas.winfo_width()), BEAM_CANVAS_MIN_WIDTH)
        height = max(int(self.beam_layout_canvas.winfo_height()), BEAM_CANVAS_MIN_HEIGHT)
        center_x = width * 0.5
        center_y = height * 0.54
        usable_radius = min(width, height) * 0.40
        scale = usable_radius / max(display_radius_deg, 1.0e-9)
        x = center_x + float(d_east_deg) * scale
        y = center_y - float(d_north_deg) * scale
        return x, y, center_x, center_y, usable_radius

    def _find_selected_beam_row(self):
        if not self.beam_layout_payload:
            return None
        for row in self.beam_layout_payload["rows"]:
            if int(row.get("beam_index", row["beam_id"])) == int(self.selected_beam_index):
                return row
        if self.beam_layout_payload["rows"]:
            return self.beam_layout_payload["rows"][0]
        return None

    def _update_beam_detail_text(self):
        if not self.beam_layout_payload:
            self.beam_detail_var.set("No beam information")
            return

        selected_row = self._find_selected_beam_row()
        if selected_row is None:
            self.beam_detail_var.set("No beam information")
            return

        metrics = self.beam_layout_payload.get("beam_offset_metrics") or {}
        file_beam_id = selected_row.get("file_beam_id", selected_row.get("beam_id", 0) + 1)
        beam_index = selected_row.get("beam_index", selected_row["beam_id"])
        detail_lines = [
            "Selected beam: file BeamID {} / beam_index {}".format(file_beam_id, beam_index),
            "fixed dEast/dNorth      : {:+.8f} deg / {:+.8f} deg".format(
                selected_row["dEast_deg"],
                selected_row["dNorth_deg"],
            ),
            "fixed offset            : {:.8f} deg".format(selected_row["offset_deg"]),
            "position angle          : {:.6f} deg".format(selected_row["position_angle_deg"]),
            "separation from beam0   : {:.8f} deg".format(selected_row["separation_from_beam0_deg"]),
            "current az/el           : {:.8f} deg / {:.8f} deg".format(
                selected_row["az_deg"],
                selected_row["el_deg"],
            ),
            "reference antenna       : {}".format(self.beam_layout_payload["antenna_name"]),
            "epoch                   : {}".format(phase.format_dual_timestamp(self.beam_layout_payload["epoch_utc"])),
            "anchor                  : {}".format(phase.format_dual_timestamp(self.beam_layout_payload["anchor_utc"])),
            "primary beam check      : {}".format(metrics.get("primary_beam_check_status", "n/a")),
            "scale mode              : {}".format(self.beam_layout_payload.get("scale_mode", "unknown")),
        ]
        spacing_deg = metrics.get("spacing_deg")
        if spacing_deg is not None:
            detail_lines.append("spacing                 : {:.8f} deg".format(spacing_deg))
        self.beam_detail_var.set("\n".join(detail_lines))

    def _redraw_beam_layout(self):
        width = max(int(self.beam_layout_canvas.winfo_width()), BEAM_CANVAS_MIN_WIDTH)
        height = max(int(self.beam_layout_canvas.winfo_height()), BEAM_CANVAS_MIN_HEIGHT)
        self.beam_layout_canvas.delete("all")
        self.beam_layout_click_targets = []

        self.beam_layout_canvas.create_rectangle(0, 0, width, height, fill="#101722", outline="")
        self.beam_layout_canvas.create_text(
            18,
            18,
            text="External 32-beam offsets",
            anchor="w",
            fill="#d9e6f2",
            font=("Microsoft YaHei UI", 11, "bold"),
        )
        self.beam_layout_canvas.create_text(
            18,
            40,
            text="Click a beam to inspect file BeamID, beam_index, and current az/el",
            anchor="w",
            fill="#9fb6cb",
            font=("Microsoft YaHei UI", 9),
        )

        if not self.beam_layout_payload:
            return

        rows = self.beam_layout_payload["rows"]
        display_radius_deg = self.beam_layout_payload["display_radius_deg"]
        primary_beam_radius_deg = self.beam_layout_payload.get("primary_beam_radius_deg")
        _, _, center_x, center_y, usable_radius = self._layout_offset_to_canvas(0.0, 0.0, display_radius_deg)

        self.beam_layout_canvas.create_line(
            center_x - usable_radius - 24,
            center_y,
            center_x + usable_radius + 24,
            center_y,
            fill="#35506b",
            dash=(4, 4),
        )
        self.beam_layout_canvas.create_line(
            center_x,
            center_y - usable_radius - 24,
            center_x,
            center_y + usable_radius + 24,
            fill="#35506b",
            dash=(4, 4),
        )
        self.beam_layout_canvas.create_text(
            center_x + usable_radius + 28,
            center_y,
            text="E",
            anchor="w",
            fill="#7eb5ff",
            font=("Microsoft YaHei UI", 9, "bold"),
        )
        self.beam_layout_canvas.create_text(
            center_x,
            center_y - usable_radius - 28,
            text="N",
            anchor="s",
            fill="#7ef0bf",
            font=("Microsoft YaHei UI", 9, "bold"),
        )

        if primary_beam_radius_deg is not None and primary_beam_radius_deg > 0.0:
            primary_radius = usable_radius * (primary_beam_radius_deg / display_radius_deg)
            self.beam_layout_canvas.create_oval(
                center_x - primary_radius,
                center_y - primary_radius,
                center_x + primary_radius,
                center_y + primary_radius,
                outline="#4c746d",
                width=2,
            )
            self.beam_layout_canvas.create_text(
                center_x,
                center_y + primary_radius + 16,
                text="Single-dish PB FWHM radius",
                anchor="n",
                fill="#87c3b4",
                font=("Microsoft YaHei UI", 8),
            )

        for offset_deg in self.beam_layout_payload["unique_rings"]:
            ring_radius = usable_radius * (offset_deg / display_radius_deg)
            self.beam_layout_canvas.create_oval(
                center_x - ring_radius,
                center_y - ring_radius,
                center_x + ring_radius,
                center_y + ring_radius,
                outline="#22384e",
                width=1,
            )
            self.beam_layout_canvas.create_text(
                center_x + ring_radius + 8,
                center_y - 4,
                text="{:.3f} deg".format(offset_deg),
                anchor="w",
                fill="#6f88a0",
                font=("Microsoft YaHei UI", 8),
            )

        selected_beam_index = int(self.selected_beam_index)
        for row in rows:
            x, y, _, _, _ = self._layout_offset_to_canvas(
                row["dEast_deg"],
                row["dNorth_deg"],
                display_radius_deg,
            )
            beam_index = int(row.get("beam_index", row["beam_id"]))
            file_beam_id = int(row.get("file_beam_id", beam_index + 1))
            is_selected = beam_index == selected_beam_index
            is_main_beam = beam_index == 0
            radius = 15 if is_selected else 12
            fill = "#f3c969" if is_main_beam else "#4e98ff"
            outline = "#fff6bf" if is_main_beam else "#cfe6ff"
            if is_selected:
                fill = "#56e2b3" if not is_main_beam else "#ffd57d"
                outline = "#eafff7"
                self.beam_layout_canvas.create_oval(
                    x - radius - 5,
                    y - radius - 5,
                    x + radius + 5,
                    y + radius + 5,
                    outline="#295b57",
                    width=3,
                )
            self.beam_layout_canvas.create_oval(
                x - radius,
                y - radius,
                x + radius,
                y + radius,
                fill=fill,
                outline=outline,
                width=2,
            )
            self.beam_layout_canvas.create_text(
                x,
                y,
                text=str(file_beam_id),
                fill="#08131f",
                font=("Microsoft YaHei UI", 9, "bold"),
            )
            self.beam_layout_click_targets.append((beam_index, x, y, radius + 6))

        metrics = self.beam_layout_payload.get("beam_offset_metrics") or {}
        spacing_text = "n/a" if metrics.get("spacing_deg") is None else "{:.6f} deg".format(metrics["spacing_deg"])
        summary = "source=file only | spacing={} | max offset={:.6f} deg | scale={:.6f} deg".format(
            spacing_text,
            self.beam_layout_payload["max_offset_deg"],
            display_radius_deg,
        )
        self.beam_layout_canvas.create_text(
            18,
            height - 18,
            text=summary,
            anchor="w",
            fill="#b7c9da",
            font=("Microsoft YaHei UI", 8),
        )

    def _handle_beam_layout_click(self, event):
        if not self.beam_layout_click_targets:
            return

        clicked_beam_index = None
        closest_distance = None
        for beam_index, x, y, radius in self.beam_layout_click_targets:
            distance = math.hypot(float(event.x) - x, float(event.y) - y)
            if distance <= radius and ((closest_distance is None) or (distance < closest_distance)):
                clicked_beam_index = beam_index
                closest_distance = distance

        if clicked_beam_index is None:
            return

        self.selected_beam_index = int(clicked_beam_index)
        self._update_beam_detail_text()
        self._redraw_beam_layout()

    def _sky_to_canvas(self, az_deg, el_deg):
        width = max(int(self.visual_canvas.winfo_width()), VISUAL_CANVAS_MIN_WIDTH)
        height = max(int(self.visual_canvas.winfo_height()), VISUAL_CANVAS_MIN_HEIGHT)
        margin_x = 24.0
        horizon_y = height - 58.0
        dome_top = 28.0
        clipped_el = max(0.0, min(90.0, float(el_deg)))
        x = margin_x + (float(az_deg) / 360.0) * (width - 2.0 * margin_x)
        y = horizon_y - (clipped_el / 90.0) * (horizon_y - dome_top)
        return x, y

    def _sample_window_track(self, antennas, target_ra, target_dec, visibility, sample_count=36):
        if visibility.never_up:
            return []

        start_utc = visibility.rise_utc
        stop_utc = visibility.set_utc
        if visibility.circumpolar:
            start_utc = visibility.epoch_time_utc - timedelta(hours=6)
            stop_utc = visibility.epoch_time_utc + timedelta(hours=6)
        if (start_utc is None) or (stop_utc is None):
            return []

        total_seconds = max((stop_utc - start_utc).total_seconds(), 1.0)
        ra_rad = phase.parse_ra_to_rad(target_ra)
        dec_rad = phase.parse_dec_to_rad(target_dec)
        ref_ant = antennas[0]

        points = []
        for index in range(sample_count + 1):
            when_utc = start_utc + timedelta(seconds=total_seconds * index / float(sample_count))
            time_context = phase.build_epoch_time_context(when_utc)
            source_state = phase.compute_source_state(
                ref_ant,
                when_utc,
                ra_rad,
                dec_rad,
                gmst_rad=time_context["gmst_rad"],
            )
            points.append(self._sky_to_canvas(source_state["azimuth_deg"], source_state["elevation_deg"]))
        return points

    def _split_path_segments(self, points):
        if not points:
            return []

        width = max(int(self.visual_canvas.winfo_width()), VISUAL_CANVAS_MIN_WIDTH)
        segments = [[points[0]]]
        for point in points[1:]:
            prev = segments[-1][-1]
            if abs(point[0] - prev[0]) > width * 0.45:
                segments.append([point])
            else:
                segments[-1].append(point)
        return [segment for segment in segments if len(segment) >= 2]

    def _build_scene_payload(self, antennas, config, snapshot):
        visibility = snapshot["visibility_window"]
        window_points = self._sample_window_track(
            antennas,
            config["target_ra"],
            config["target_dec"],
            visibility,
        )
        window_segments = self._split_path_segments(window_points)

        source_point = None
        if visibility.visible_now:
            source_point = self._sky_to_canvas(snapshot["main_az_deg"], snapshot["main_el_deg"])

        rise_point = window_points[0] if window_points else None
        set_point = window_points[-1] if window_points else None

        width = max(int(self.visual_canvas.winfo_width()), VISUAL_CANVAS_MIN_WIDTH)
        height = max(int(self.visual_canvas.winfo_height()), VISUAL_CANVAS_MIN_HEIGHT)
        horizon_y = height - 58.0
        parked_point = (width * 0.5, horizon_y - 18.0)

        if source_point is not None:
            aim_point = source_point
        elif rise_point is not None:
            aim_point = rise_point
        else:
            aim_point = parked_point

        if visibility.never_up:
            headline = "Never up"
        elif visibility.visible_now:
            headline = "Tracking now"
        elif visibility.circumpolar:
            headline = "Circumpolar"
        else:
            headline = "Waiting for visibility"

        return {
            "headline": headline,
            "window_segments": window_segments,
            "window_points": window_points,
            "rise_point": rise_point,
            "set_point": set_point,
            "source_point": source_point,
            "aim_point": aim_point,
            "visible_now": bool(visibility.visible_now),
            "visibility": visibility,
        }

    def _set_scene_payload(self, payload):
        self.scene_payload = payload
        self.target_focus_point = payload.get("aim_point")
        if self.current_focus_point is None and self.target_focus_point is not None:
            self.current_focus_point = self.target_focus_point

    def _ease_focus_point(self):
        if self.target_focus_point is None:
            return
        if self.current_focus_point is None:
            self.current_focus_point = self.target_focus_point
            return

        cx, cy = self.current_focus_point
        tx, ty = self.target_focus_point
        self.current_focus_point = (
            cx + (tx - cx) * SCENE_EASE,
            cy + (ty - cy) * SCENE_EASE,
        )

    def _draw_star_field(self, width, height):
        stars = [
            (44, 38, 1.6), (110, 62, 1.2), (180, 30, 1.4), (270, 54, 1.8),
            (340, 34, 1.1), (410, 66, 1.5), (462, 106, 1.0), (72, 116, 1.3),
            (154, 142, 1.0), (370, 130, 1.4), (438, 186, 1.1), (220, 98, 0.9),
        ]
        x_scale = width / float(VISUAL_CANVAS_MIN_WIDTH)
        y_scale = height / float(VISUAL_CANVAS_MIN_HEIGHT)
        for x, y, radius in stars:
            sx = x * x_scale
            sy = y * y_scale
            self.visual_canvas.create_oval(
                sx - radius,
                sy - radius,
                sx + radius,
                sy + radius,
                fill="#dbefff",
                outline="",
            )

    def _draw_telescope(self, width, height):
        horizon_y = height - 58.0
        pivot_x = width * 0.50
        pivot_y = horizon_y + 14.0

        focus_point = self.current_focus_point or (pivot_x, horizon_y - 18.0)
        dx = focus_point[0] - pivot_x
        dy = min(focus_point[1] - pivot_y, -10.0)
        length = max(math.hypot(dx, dy), 1.0)
        ux = dx / length
        uy = dy / length

        tube_len = 84.0
        end_x = pivot_x + ux * tube_len
        end_y = pivot_y + uy * tube_len
        dish_x = end_x + ux * 12.0
        dish_y = end_y + uy * 12.0

        self.visual_canvas.create_line(
            pivot_x - 20.0, pivot_y + 44.0, pivot_x, pivot_y + 10.0, width=5, fill="#5a6e7f"
        )
        self.visual_canvas.create_line(
            pivot_x + 20.0, pivot_y + 44.0, pivot_x, pivot_y + 10.0, width=5, fill="#5a6e7f"
        )
        self.visual_canvas.create_line(
            pivot_x, pivot_y + 10.0, pivot_x, pivot_y + 48.0, width=6, fill="#6f8598"
        )
        self.visual_canvas.create_oval(
            pivot_x - 10.0, pivot_y + 2.0, pivot_x + 10.0, pivot_y + 20.0,
            fill="#90a5b5", outline="#d9e4ec", width=2,
        )
        self.visual_canvas.create_line(
            pivot_x, pivot_y + 10.0, end_x, end_y,
            width=12, fill="#c7d2db", capstyle="round",
        )
        self.visual_canvas.create_line(
            pivot_x, pivot_y + 10.0, end_x, end_y,
            width=3, fill="#eef4f8",
        )
        self.visual_canvas.create_oval(
            dish_x - 16.0, dish_y - 10.0, dish_x + 16.0, dish_y + 10.0,
            outline="#7fd7ff", width=3,
        )
        self.visual_canvas.create_line(
            end_x, end_y, dish_x - ux * 4.0, dish_y - uy * 4.0,
            fill="#7fd7ff", width=3,
        )
        if self.scene_payload and self.scene_payload.get("visible_now"):
            self.visual_canvas.create_line(
                dish_x,
                dish_y,
                focus_point[0],
                focus_point[1],
                fill="#60f0bf",
                dash=(6, 4),
                width=2,
            )

    def _redraw_visual_scene(self):
        width = max(int(self.visual_canvas.winfo_width()), VISUAL_CANVAS_MIN_WIDTH)
        height = max(int(self.visual_canvas.winfo_height()), VISUAL_CANVAS_MIN_HEIGHT)
        self.visual_canvas.delete("all")

        self.visual_canvas.create_rectangle(0, 0, width, height, fill="#08131f", outline="")
        self.visual_canvas.create_rectangle(0, 0, width, height * 0.52, fill="#11243a", outline="")
        self._draw_star_field(width, height)

        horizon_y = height - 58.0
        self.visual_canvas.create_rectangle(0, horizon_y, width, height, fill="#10261e", outline="")
        self.visual_canvas.create_line(0, horizon_y, width, horizon_y, fill="#6aa38c", width=2)
        self.visual_canvas.create_text(
            18,
            horizon_y + 14,
            text="Horizon",
            anchor="w",
            fill="#b7d7c9",
            font=("Microsoft YaHei UI", 9),
        )

        payload = self.scene_payload or {}
        segments = payload.get("window_segments", [])
        glow_color = "#284d62"
        path_color = "#5bc2ff" if not payload.get("visible_now") else "#55f0b3"
        for segment in segments:
            flat = [coord for point in segment for coord in point]
            self.visual_canvas.create_line(*flat, smooth=True, width=10, fill=glow_color)
            self.visual_canvas.create_line(*flat, smooth=True, width=4, fill=path_color)

        rise_point = payload.get("rise_point")
        set_point = payload.get("set_point")
        if rise_point is not None:
            self.visual_canvas.create_oval(
                rise_point[0] - 5, rise_point[1] - 5, rise_point[0] + 5, rise_point[1] + 5,
                fill="#ffe17a", outline="",
            )
            self.visual_canvas.create_text(
                rise_point[0], rise_point[1] + 16, text="rise", fill="#f8e69d", font=("Microsoft YaHei UI", 9)
            )
        if set_point is not None:
            self.visual_canvas.create_oval(
                set_point[0] - 5, set_point[1] - 5, set_point[0] + 5, set_point[1] + 5,
                fill="#ffb879", outline="",
            )
            self.visual_canvas.create_text(
                set_point[0], set_point[1] + 16, text="set", fill="#ffd0a0", font=("Microsoft YaHei UI", 9)
            )

        source_point = payload.get("source_point")
        if source_point is not None:
            self.visual_canvas.create_oval(
                source_point[0] - 12, source_point[1] - 12, source_point[0] + 12, source_point[1] + 12,
                fill="#1f7f6e", outline="",
            )
            self.visual_canvas.create_oval(
                source_point[0] - 6, source_point[1] - 6, source_point[0] + 6, source_point[1] + 6,
                fill="#b8fff0", outline="#effff8",
            )
            self.visual_canvas.create_text(
                source_point[0], source_point[1] - 18, text="source", fill="#ddfff7", font=("Microsoft YaHei UI", 9)
            )

        self._draw_telescope(width, height)

        headline = payload.get("headline", "Waiting for data")
        self.visual_canvas.create_text(
            16, 16, text=headline, anchor="w", fill="#e3f0ff", font=("Microsoft YaHei UI", 11, "bold")
        )
        self.visual_canvas.create_text(
            16, 36, text="Visible window", anchor="w", fill="#c2d6e8", font=("Microsoft YaHei UI", 9)
        )

    def _animation_tick(self):
        self._ease_focus_point()
        self._redraw_visual_scene()
        self.animation_after_id = self.root.after(ANIMATION_INTERVAL_MS, self._animation_tick)

    def _compute_would_publish(self, snapshot, simulation_ignore_visibility):
        visibility = snapshot["visibility_window"]
        return bool(visibility.visible_now or simulation_ignore_visibility)

    def _format_visibility_lines_for_gui(self, visibility):
        lines = []
        for line in phase.build_visibility_status_lines(visibility):
            translated = line
            for english_prefix, bilingual_prefix in self.VISIBILITY_PREFIX_MAP.items():
                if line.startswith(english_prefix):
                    translated = line.replace(english_prefix, bilingual_prefix, 1)
                    break
            lines.append(translated)
        return lines

    def _format_snapshot_block(self, title, snapshot, simulation_ignore_visibility):
        visibility = snapshot["visibility_window"]
        would_publish = self._compute_would_publish(snapshot, simulation_ignore_visibility)

        lines = [
            title,
            "",
            "epoch                                : {}".format(
                phase.format_dual_timestamp(snapshot["epoch_time_utc"])
            ),
            "reference antenna                    : {}".format(snapshot["reference_antenna"]),
            "target_ra                            : {}".format(snapshot["target_ra"]),
            "target_dec                           : {}".format(snapshot["target_dec"]),
            "main_az_deg                          : {:.6f}".format(snapshot["main_az_deg"]),
            "main_el_deg                          : {:.6f}".format(snapshot["main_el_deg"]),
            "lst_hms                              : {}".format(snapshot["lst_hms"]),
            "simulation_ignore_visibility         : {}".format(simulation_ignore_visibility),
            "would_publish                        : {}".format(would_publish),
        ]
        lines.extend(self._format_visibility_lines_for_gui(visibility))
        return "\n".join(lines)

    def _build_input_mapping_text(self, antennas):
        antenna_count = len(antennas)
        active_signal_inputs = phase.get_active_signal_inputs(antenna_count)
        padded_signal_inputs = phase.TOTAL_SIGNAL_INPUTS - active_signal_inputs

        lines = [
            "antenna_count            : {}".format(antenna_count),
            "signals_per_antenna      : {}".format(phase.SIGNALS_PER_ANTENNA),
            "active_signal_inputs     : {}".format(active_signal_inputs),
            "total_signal_inputs      : {}".format(phase.TOTAL_SIGNAL_INPUTS),
            "padded_signal_inputs     : {}".format(padded_signal_inputs),
        ]

        if padded_signal_inputs > 0:
            lines.append(
                "padded input indices     : {}..{}".format(
                    active_signal_inputs,
                    phase.TOTAL_SIGNAL_INPUTS - 1,
                )
            )
            lines.append(
                "padded input ids         : {}..{}".format(
                    active_signal_inputs + 1,
                    phase.TOTAL_SIGNAL_INPUTS,
                )
            )
            lines.append("padded value in A/B/omega: 0.0")
        else:
            lines.append("padded input indices     : none")
            lines.append("padded input ids         : none")

        return "\n".join(lines)

    def _build_beam_offset_status_text(self, preview_beam_model):
        rows = preview_beam_model["beam_offset_rows"]
        meta = preview_beam_model["beam_offset_meta"]

        lines = [
            "beam_offset_table_file   : {}".format(phase.BEAM_OFFSET_TABLE_FILE),
            "auto_generate_offsets    : disabled",
            "beam rows                : {}".format(len(rows)),
            "source                   : {}".format(meta.get("source", "unknown")),
            "file BeamID support      : 1..32 or 0..31",
            "internal beam_index      : 0..31",
        ]

        if rows:
            first = rows[0]
            lines.append(
                "center beam              : file BeamID {} / beam_index {}".format(
                    first.get("file_beam_id", first.get("beam_id", 0) + 1),
                    first.get("beam_index", first.get("beam_id", 0)),
                )
            )
            lines.append(
                "center offset            : dEast={:+.8f}, dNorth={:+.8f}".format(
                    first["dEast_deg"],
                    first["dNorth_deg"],
                )
            )

        validation = meta.get("primary_beam_validation")
        if validation:
            lines.append(
                "primary beam check       : {}".format(validation.get("primary_beam_check_status", "n/a"))
            )
            lines.append(
                "max offset deg           : {:.8f}".format(validation.get("max_offset_deg", 0.0))
            )

        return "\n".join(lines)

    def _build_trace_file_status_text(self):
        path = phase.TRACE_MODE_PHASE_TXT
        expected_bytes = phase.TRACE_STREAM_BYTES
        expected_values = phase.TRACE_STREAM_FLOAT_COUNT

        lines = [
            "trace_mode_phase_coeff.txt",
            "path                    : {}".format(path),
            "layout                  : A[20,32] + B[20,32] + omega[20,32]",
            "dtype                   : little-endian float32",
            "expected values         : {}".format(expected_values),
            "expected bytes          : {}".format(expected_bytes),
            "A/B frequency factor    : not included",
            "GPU formula             : Phi3 = nu * (A*cos(delta) + B*sin(delta) - A)",
            "t0 in file              : no",
        ]

        if os.path.exists(path):
            size = os.path.getsize(path)
            mtime = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S")
            lines.append("exists                  : yes")
            lines.append("actual bytes            : {}".format(size))
            lines.append("modified time           : {}".format(mtime))
            lines.append("size check              : {}".format("PASS" if size == expected_bytes else "FAIL"))
        else:
            lines.append("exists                  : no")
            lines.append("actual bytes            : n/a")
            lines.append("size check              : FAIL")

        return "\n".join(lines)

    def _build_refresh_status_line(self, now_utc, current_snapshot, next_slot_snapshot, simulation_ignore_visibility):
        current_publish = self._compute_would_publish(current_snapshot, simulation_ignore_visibility)
        next_publish = self._compute_would_publish(next_slot_snapshot, simulation_ignore_visibility)
        return (
            "Last refresh: {time} UTC | current would_publish={current_publish} | "
            "next_slot would_publish={next_publish}"
        ).format(
            time=now_utc.strftime("%Y-%m-%d %H:%M:%S"),
            current_publish=current_publish,
            next_publish=next_publish,
        )

    def _build_friendly_error_text(self, exc):
        return "\n".join(
            [
                "Refresh failed",
                "",
                "Possible causes:",
                "1. ants.txt path is wrong or the file format is invalid",
                "2. config_32beam_hex37_drop5_beam_offsets.txt is missing or malformed",
                "3. RA/Dec format is invalid",
                "4. compute_trace_mode_phase_coeff.py and GUI variables are out of sync",
                "",
                "Error:",
                str(exc),
                "",
                "Traceback:",
                traceback.format_exc(),
            ]
        )

    def refresh_now(self):
        self._cancel_scheduled_refresh()
        try:
            config = self._collect_config()
            now_utc = phase.utc_now()
            next_slot_utc = phase.ceil_utc_to_next_slot_boundary(now_utc, phase.UPDATE_PERIOD_SECONDS)
            antennas = phase.read_antenna_file(config["ants_txt"])
            preview_beam_model = self._get_or_build_preview_beam_model(config, antennas, next_slot_utc)

            current_snapshot = phase.build_live_status_snapshot(
                ants_txt=config["ants_txt"],
                target_ra=config["target_ra"],
                target_dec=config["target_dec"],
                min_elevation_deg=config["min_elevation_deg"],
                when_utc=now_utc,
                antennas=antennas,
            )
            next_slot_snapshot = phase.build_live_status_snapshot(
                ants_txt=config["ants_txt"],
                target_ra=config["target_ra"],
                target_dec=config["target_dec"],
                min_elevation_deg=config["min_elevation_deg"],
                when_utc=next_slot_utc,
                antennas=antennas,
            )

            current_text = self._format_snapshot_block(
                "Current source status",
                current_snapshot,
                config["simulation_ignore_visibility"],
            )
            next_text = self._format_snapshot_block(
                "Next slot preview",
                next_slot_snapshot,
                config["simulation_ignore_visibility"],
            )
            self._set_text_widget(self.current_status_text, current_text)
            self._set_text_widget(self.next_slot_text, next_text)

            self._set_scene_payload(self._build_scene_payload(antennas, config, current_snapshot))
            self.beam_layout_payload = self._build_beam_layout_payload(preview_beam_model, now_utc)
            self._update_beam_detail_text()
            self._redraw_beam_layout()

            self.input_mapping_var.set(self._build_input_mapping_text(antennas))
            self.beam_offset_status_var.set(self._build_beam_offset_status_text(preview_beam_model))
            self.trace_file_status_var.set(self._build_trace_file_status_text())
            self._set_output_text(
                self._build_refresh_log_text(
                    now_utc,
                    current_snapshot,
                    next_slot_snapshot,
                    config["simulation_ignore_visibility"],
                )
            )
            self.status_var.set(
                self._build_refresh_status_line(
                    now_utc,
                    current_snapshot,
                    next_slot_snapshot,
                    config["simulation_ignore_visibility"],
                )
            )
        except Exception as exc:
            self.status_var.set("Refresh failed: {}".format(exc))
            self._set_text_widget(self.current_status_text, "Refresh failed. See the Log tab for details.")
            self._set_text_widget(self.next_slot_text, "Refresh failed. See the Log tab for details.")
            self._set_output_text(self._build_friendly_error_text(exc))
            self.scene_payload = None
            self.beam_layout_payload = None
            self.preview_beam_model = None
            self.preview_beam_model_signature = None
            self.input_mapping_var.set("input mapping not available due to refresh failure")
            self.beam_offset_status_var.set("beam offset file not checked")
            self.trace_file_status_var.set(self._build_trace_file_status_text())
            self._update_beam_detail_text()
            self._redraw_beam_layout()
            self.notebook.select(self.log_tab)
        finally:
            if self.auto_refresh_var.get():
                self._schedule_refresh()


def main():
    root = tk.Tk()
    PhaseCoeffStatusGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
