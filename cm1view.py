#!/opt/miniconda3/envs/met/bin/python3
"""cm1view.py — CM1 netCDF output viewer.

Supports:
  • Plan view (x-y at a fixed height level)
  • X-Z cross section (at fixed y)
  • Y-Z cross section (at fixed x)
  • Time stepping with slider, back/forward buttons, and auto-play
  • Wind-vector overlay (u/v on plan view; u/w or v/w on cross-sections)
  • Save current frame as PNG
  • Save a time range as an animated GIF
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import glob
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.colors import Normalize
from matplotlib.animation import PillowWriter
import matplotlib.pyplot as plt

import netCDF4 as nc


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

class CM1Dataset:
    """Wraps one or more CM1 netCDF files as a unified time series."""

    def __init__(self, paths):
        paths = sorted(paths)
        self._dsets = [nc.Dataset(p, 'r') for p in paths]
        self._build_index()

    def _build_index(self):
        """Build (dset_idx, time_idx_in_dset) for every time step."""
        self._time_map = []   # list of (ds_idx, t_idx)
        self._times    = []   # seconds
        for di, ds in enumerate(self._dsets):
            try:
                t = ds.variables['time'][:]
            except KeyError:
                t = np.array([0.0])
            for ti, tv in enumerate(t):
                self._time_map.append((di, ti))
                self._times.append(float(tv))
        self._times = np.array(self._times)

        # coordinates from first dataset
        ds0 = self._dsets[0]
        self.xh = np.array(ds0.variables['xh'][:])
        self.yh = np.array(ds0.variables['yh'][:])
        try:
            self.zh = np.array(ds0.variables['zh'][:])
        except KeyError:
            nk = next(iter(ds0.dimensions.get(k) for k in ('zh', 'nz') if k in ds0.dimensions), None)
            self.zh = np.arange(len(self.xh)) if nk is None else np.arange(nk)

        # available 3-D and 2-D field names (skip coordinate/metadata vars)
        _coord = {'xh', 'xf', 'yh', 'yf', 'zh', 'zf', 'time', 'f_cor',
                  'ztop', 'umove', 'vmove', 'zs', 'zh_sfc'}
        self.fields_3d = []
        self.fields_2d = []
        for name, var in ds0.variables.items():
            if name in _coord:
                continue
            dims = var.dimensions
            if 'time' not in dims:
                continue
            # spatial dims excluding time
            sp = [d for d in dims if d != 'time']
            if len(sp) == 3:
                self.fields_3d.append(name)
            elif len(sp) == 2:
                self.fields_2d.append(name)
        self.fields_3d.sort()
        self.fields_2d.sort()

    @property
    def ntimes(self):
        return len(self._times)

    @property
    def times(self):
        return self._times

    def get_field(self, name, time_idx):
        di, ti = self._time_map[time_idx]
        ds = self._dsets[di]
        var = ds.variables[name]
        dims = var.dimensions
        t_ax = list(dims).index('time')
        slices = [slice(None)] * len(dims)
        slices[t_ax] = ti
        return np.array(var[tuple(slices)], dtype=float)

    def get_units(self, name):
        ds0 = self._dsets[0]
        var = ds0.variables.get(name)
        if var is None:
            return ''
        return getattr(var, 'units', '') or ''

    def get_longname(self, name):
        ds0 = self._dsets[0]
        var = ds0.variables.get(name)
        if var is None:
            return name
        return getattr(var, 'long_name', name) or name

    def add_paths(self, new_paths):
        """Append new files; returns number successfully added."""
        n_added = 0
        times_list = list(self._times)
        for p in sorted(new_paths):
            try:
                ds = nc.Dataset(p, 'r')
            except Exception:
                continue
            di = len(self._dsets)
            self._dsets.append(ds)
            try:
                t = ds.variables['time'][:]
            except KeyError:
                t = np.array([0.0])
            for ti, tv in enumerate(t):
                self._time_map.append((di, ti))
                times_list.append(float(tv))
            n_added += 1
        self._times = np.array(times_list)
        return n_added

    def close(self):
        for ds in self._dsets:
            ds.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CMAPS = [
    'RdBu_r', 'bwr', 'seismic',
    'viridis', 'plasma', 'inferno', 'magma', 'cividis',
    'Blues', 'Reds', 'Greens', 'BuGn',
    'turbo', 'jet', 'rainbow',
    'Greys', 'binary',
    'gist_ncar', 'gist_rainbow',
]

def _sec_label(s):
    m = s / 60.0
    if m < 60:
        return f"{m:.1f} min"
    h = m / 60.0
    return f"{h:.2f} h"


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class CM1Viewer(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("CM1 netCDF Viewer")
        self.minsize(1100, 700)

        self._ds           = None
        self._t_idx        = 0
        self._play_id      = None
        self._playing      = False
        self._last_field   = None
        self._sounding_mode = False
        self._mpl_cid      = None
        self._watch_dir    = None
        self._watch_known  = set()
        self._watch_id     = None

        self._build_vars()
        self._build_ui()

    # ── variables ────────────────────────────────────────────────────────────

    def _build_vars(self):
        self.v_field      = tk.StringVar()
        self.v_cmap       = tk.StringVar(value='RdBu_r')
        self.v_view       = tk.StringVar(value='plan')
        self.v_symcb      = tk.BooleanVar(value=False)
        self.v_vmin       = tk.StringVar(value='')
        self.v_vmax       = tk.StringVar(value='')
        self.v_winds      = tk.BooleanVar(value=False)
        self.v_wind_skip  = tk.IntVar(value=4)
        self.v_wind_type  = tk.StringVar(value='arrows')
        self.v_ctr_field  = tk.StringVar(value='')
        self.v_ctr_levels = tk.IntVar(value=8)
        self.v_ctr_lw     = tk.DoubleVar(value=1.2)
        self.v_ctr_color  = tk.StringVar(value='black')
        self.v_ctr_labels = tk.BooleanVar(value=False)
        self.v_ctr_style  = tk.StringVar(value='solid')
        self.v_ctr_min    = tk.StringVar(value='')
        self.v_ctr_max    = tk.StringVar(value='')
        self.v_ctr_sym    = tk.BooleanVar(value=False)
        self.v_xmin = tk.StringVar(value='')
        self.v_xmax = tk.StringVar(value='')
        self.v_ymin = tk.StringVar(value='')
        self.v_ymax = tk.StringVar(value='')
        self.v_xlim_sym = tk.BooleanVar(value=False)
        self.v_ylim_sym = tk.BooleanVar(value=False)
        self.v_gif_t0     = tk.StringVar(value='0')
        self.v_gif_t1     = tk.StringVar(value='')
        self.v_live       = tk.BooleanVar(value=True)

    # ── UI ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _autocomplete(combo, get_full_list, on_commit=None):
        """Let the user type to filter a combobox; commit on Return or selection."""
        def _filter(event=None):
            if event and event.keysym in ('Return', 'KP_Enter'):
                if on_commit:
                    on_commit()
                return
            if event and event.keysym in ('Up', 'Down', 'Escape', 'Tab'):
                return
            typed = combo.get().lower()
            full  = get_full_list()
            combo['values'] = [v for v in full if typed in v.lower()] if typed else full

        combo.configure(state='normal')
        combo.bind('<KeyRelease>', _filter)
        combo.bind('<<ComboboxSelected>>', lambda _: on_commit() if on_commit else None)

    def _build_ui(self):
        # ── top bar: file + field ────────────────────────────────────────
        top = ttk.Frame(self)
        top.pack(fill='x', padx=6, pady=4)

        ttk.Button(top, text="Open file(s)…", command=self._open).pack(side='left', padx=4)
        ttk.Button(top, text="Open dir…", command=self._open_dir).pack(side='left', padx=4)
        self._snd_btn = ttk.Button(top, text="Take Sounding",
                                   command=self._toggle_sounding_mode)
        self._snd_btn.pack(side='left', padx=4)
        self._file_lbl = ttk.Label(top, text="No file loaded.", foreground='gray')
        self._file_lbl.pack(side='left', padx=8)
        self._watch_lbl = ttk.Label(top, text="", foreground='#080')
        self._watch_lbl.pack(side='left', padx=4)

        ttk.Label(top, text="Field:").pack(side='left')
        self._field_cb = ttk.Combobox(top, textvariable=self.v_field, width=18)
        self._field_cb.pack(side='left', padx=4)
        self._all_fields = []
        self._autocomplete(self._field_cb, lambda: self._all_fields, self._plot)

        ttk.Label(top, text="Colormap:").pack(side='left', padx=(12, 2))
        self._cmap_cb = ttk.Combobox(top, textvariable=self.v_cmap, width=14)
        self._cmap_cb.pack(side='left', padx=4)
        self._autocomplete(self._cmap_cb, lambda: CMAPS, self._plot)
        self.v_cmap.trace_add('write', lambda *_: self._plot())

        # ── main area: fixed left panel + expanding right plot ───────────
        main = ttk.Frame(self)
        main.pack(fill='both', expand=True, padx=6, pady=2)

        # Left controls — scrollable panel
        left_outer = ttk.Frame(main, width=248)
        left_outer.pack(side='left', fill='y')
        left_outer.pack_propagate(False)

        _lsb = ttk.Scrollbar(left_outer, orient='vertical')
        _lsb.pack(side='right', fill='y')
        _lcv = tk.Canvas(left_outer, yscrollcommand=_lsb.set,
                         highlightthickness=0, width=230)
        _lcv.pack(side='left', fill='both', expand=True)
        _lsb.config(command=_lcv.yview)

        left = ttk.Frame(_lcv)
        _lcv_win = _lcv.create_window((0, 0), window=left, anchor='nw')

        def _on_left_configure(e):
            _lcv.configure(scrollregion=_lcv.bbox('all'))
        def _on_lcv_resize(e):
            _lcv.itemconfig(_lcv_win, width=e.width)
        left.bind('<Configure>', _on_left_configure)
        _lcv.bind('<Configure>', _on_lcv_resize)

        def _on_mousewheel(e):
            _lcv.yview_scroll(int(-1 * (e.delta / 120)), 'units')
        left_outer.bind_all('<MouseWheel>', _on_mousewheel)

        self._build_controls(left)

        ttk.Separator(main, orient='vertical').pack(side='left', fill='y', padx=2)

        # Right plot — fills all remaining space
        right = ttk.Frame(main)
        right.pack(side='left', fill='both', expand=True)

        self._fig = Figure(dpi=100)
        # Fixed axes positions so colorbar never steals space from the plot
        self._ax  = self._fig.add_axes([0.10, 0.10, 0.74, 0.82])
        self._cax = self._fig.add_axes([0.87, 0.10, 0.03, 0.82])
        self._cbar = None
        self._canvas = FigureCanvasTkAgg(self._fig, master=right)
        self._canvas.get_tk_widget().pack(fill='both', expand=True)
        tb = NavigationToolbar2Tk(self._canvas, right, pack_toolbar=False)
        tb.pack(fill='x')
        self._mpl_cid = self._fig.canvas.mpl_connect(
            'button_press_event', self._on_canvas_click)

        # ── bottom: time controls + save ────────────────────────────────
        bot = ttk.Frame(self)
        bot.pack(fill='x', padx=6, pady=4)
        self._build_time_bar(bot)

    def _build_controls(self, parent):
        P = {"padx": 6, "pady": 3}

        # View
        vf = ttk.LabelFrame(parent, text="View")
        vf.pack(fill='x', padx=6, pady=6)
        for text, val in [("Plan  (x-y)", 'plan'),
                          ("X-Z  (fix y)", 'xz'),
                          ("Y-Z  (fix x)", 'yz')]:
            ttk.Radiobutton(vf, text=text, variable=self.v_view,
                            value=val, command=self._plot).pack(
                anchor='w', padx=8, pady=2)

        # Level / slice position
        lf = ttk.LabelFrame(parent, text="Level / position")
        lf.pack(fill='x', padx=6, pady=4)

        ttk.Label(lf, text="Z level (plan view):").pack(anchor='w', padx=6, pady=2)
        self._z_slider = ttk.Scale(lf, from_=0, to=1, orient='horizontal',
                                   command=self._on_z_change)
        self._z_slider.pack(fill='x', padx=6)
        self._z_lbl = ttk.Label(lf, text="z = — km", font=('Courier', 9))
        self._z_lbl.pack(anchor='w', padx=6)

        ttk.Label(lf, text="Cross-section pos:").pack(anchor='w', padx=6, pady=(8, 2))
        self._cs_slider = ttk.Scale(lf, from_=0, to=1, orient='horizontal',
                                    command=self._on_cs_change)
        self._cs_slider.set(0.5)
        self._cs_slider.pack(fill='x', padx=6)
        self._cs_lbl = ttk.Label(lf, text="pos = —", font=('Courier', 9))
        self._cs_lbl.pack(anchor='w', padx=6)

        # Colorbar range
        cf = ttk.LabelFrame(parent, text="Color range")
        cf.pack(fill='x', padx=6, pady=4)
        ttk.Checkbutton(cf, text="Symmetric (±max)",
                        variable=self.v_symcb,
                        command=self._plot).pack(anchor='w', padx=6, pady=2)
        for lbl, var in [("Min:", self.v_vmin), ("Max:", self.v_vmax)]:
            row = ttk.Frame(cf)
            row.pack(fill='x', padx=6, pady=1)
            ttk.Label(row, text=lbl, width=4).pack(side='left')
            e = ttk.Entry(row, textvariable=var, width=10)
            e.pack(side='left')
            e.bind('<Return>', lambda _: self._plot())
        ttk.Button(cf, text="Reset to auto",
                   command=self._reset_range).pack(padx=6, pady=4)

        # Axis limits
        af = ttk.LabelFrame(parent, text="Axis limits")
        af.pack(fill='x', padx=6, pady=4)

        def _lim_entry(parent_row, v_this, v_other, v_sym, is_max):
            """Entry that mirrors its partner when symmetric is on."""
            e = ttk.Entry(parent_row, textvariable=v_this, width=7)
            def _apply(event=None):
                if v_sym.get():
                    try:
                        val = float(v_this.get())
                        v_other.set(f'{-val:.6g}')
                    except ValueError:
                        pass
                self._plot()
            e.bind('<Return>', _apply)
            e.bind('<FocusOut>', _apply)
            return e

        for axis_lbl, v_min, v_max, v_sym, sym_cmd in [
            ('X', self.v_xmin, self.v_xmax, self.v_xlim_sym, self._on_xlim_sym),
            ('Y', self.v_ymin, self.v_ymax, self.v_ylim_sym, self._on_ylim_sym),
        ]:
            row = ttk.Frame(af)
            row.pack(fill='x', padx=6, pady=2)
            ttk.Label(row, text=axis_lbl, width=2).pack(side='left')
            ttk.Label(row, text="min", width=3).pack(side='left')
            _lim_entry(row, v_min, v_max, v_sym, is_max=False).pack(side='left', padx=(0, 4))
            ttk.Label(row, text="max", width=3).pack(side='left')
            _lim_entry(row, v_max, v_min, v_sym, is_max=True).pack(side='left')

            sym_row = ttk.Frame(af)
            sym_row.pack(fill='x', padx=6, pady=(0, 2))
            ttk.Checkbutton(sym_row, text=f"Symmetric {axis_lbl} (±)",
                            variable=v_sym,
                            command=sym_cmd).pack(side='left')

        ttk.Button(af, text="Reset to auto",
                   command=self._reset_lims).pack(padx=6, pady=(0, 4))

        # Wind overlay
        wf = ttk.LabelFrame(parent, text="Wind overlay")
        wf.pack(fill='x', padx=6, pady=4)
        ttk.Checkbutton(wf, text="Show winds",
                        variable=self.v_winds,
                        command=self._plot).pack(anchor='w', padx=6, pady=2)
        row = ttk.Frame(wf)
        row.pack(fill='x', padx=6, pady=2)
        ttk.Label(row, text="Skip N:").pack(side='left')
        ttk.Spinbox(row, from_=1, to=20, textvariable=self.v_wind_skip,
                    width=4, command=self._plot).pack(side='left', padx=4)
        row2 = ttk.Frame(wf)
        row2.pack(fill='x', padx=6, pady=(0, 4))
        ttk.Radiobutton(row2, text="Arrows", variable=self.v_wind_type,
                        value='arrows', command=self._plot).pack(side='left')
        ttk.Radiobutton(row2, text="Barbs", variable=self.v_wind_type,
                        value='barbs', command=self._plot).pack(side='left', padx=6)

        # Contour overlay
        ctrf = ttk.LabelFrame(parent, text="Contour overlay")
        ctrf.pack(fill='x', padx=6, pady=4)
        ttk.Label(ctrf, text="Variable:").pack(anchor='w', padx=6, pady=(4, 0))
        self._ctr_cb = ttk.Combobox(ctrf, textvariable=self.v_ctr_field, width=18)
        self._ctr_cb.pack(fill='x', padx=6, pady=2)
        self._all_ctr_fields = ['']
        self._autocomplete(self._ctr_cb,
                           lambda: self._all_ctr_fields, self._plot)

        row = ttk.Frame(ctrf)
        row.pack(fill='x', padx=6, pady=2)
        ttk.Label(row, text="Levels:").pack(side='left')
        ttk.Spinbox(row, from_=2, to=40, textvariable=self.v_ctr_levels,
                    width=4, command=self._plot).pack(side='left', padx=4)
        ttk.Label(row, text="LW:").pack(side='left', padx=(8, 0))
        ttk.Spinbox(row, from_=0.1, to=10.0, increment=0.1,
                    textvariable=self.v_ctr_lw, width=5,
                    command=self._plot).pack(side='left', padx=4)
        ttk.Checkbutton(ctrf, text="Label contours", variable=self.v_ctr_labels,
                        command=self._plot).pack(anchor='w', padx=6, pady=(0, 2))

        ttk.Label(ctrf, text="Color / colormap:").pack(anchor='w', padx=6)
        _ctr_colors = [
            'black', 'white', 'gray', 'red', 'blue', 'green',
            'orange', 'cyan', 'magenta', 'yellow',
            'viridis', 'plasma', 'RdBu_r', 'bwr', 'Reds', 'Blues',
            'turbo', 'jet', 'gist_ncar',
        ]
        self._ctr_color_cb = ttk.Combobox(
            ctrf, textvariable=self.v_ctr_color, width=18, values=_ctr_colors)
        self._ctr_color_cb.pack(fill='x', padx=6, pady=(2, 4))
        self._autocomplete(self._ctr_color_cb, lambda: _ctr_colors, self._plot)
        self.v_ctr_color.trace_add('write', lambda *_: self._plot())

        ttk.Label(ctrf, text="Line style:").pack(anchor='w', padx=6)
        _style_row = ttk.Frame(ctrf)
        _style_row.pack(fill='x', padx=6, pady=(0, 6))
        ttk.Radiobutton(_style_row, text="Solid", variable=self.v_ctr_style,
                        value='solid', command=self._plot).pack(side='left')
        ttk.Radiobutton(_style_row, text="Dashed", variable=self.v_ctr_style,
                        value='dashed', command=self._plot).pack(side='left', padx=4)
        ttk.Radiobutton(_style_row, text="+solid /−dash", variable=self.v_ctr_style,
                        value='pn', command=self._plot).pack(side='left')

        ttk.Label(ctrf, text="Range (min / max):").pack(anchor='w', padx=6, pady=(4, 0))
        _cr = ttk.Frame(ctrf)
        _cr.pack(fill='x', padx=6, pady=2)
        _lim_entry(_cr, self.v_ctr_min, self.v_ctr_max, self.v_ctr_sym, is_max=False).pack(side='left')
        ttk.Label(_cr, text="/").pack(side='left', padx=4)
        _lim_entry(_cr, self.v_ctr_max, self.v_ctr_min, self.v_ctr_sym, is_max=True).pack(side='left')
        ttk.Checkbutton(ctrf, text="Symmetric", variable=self.v_ctr_sym,
                        command=self._plot).pack(anchor='w', padx=6, pady=(0, 6))

    def _build_time_bar(self, parent):
        # Playback row
        pb = ttk.Frame(parent)
        pb.pack(fill='x')

        ttk.Button(pb, text="◀◀", width=3,
                   command=self._t_first).pack(side='left', padx=2)
        ttk.Button(pb, text="◀",  width=3,
                   command=self._t_prev).pack(side='left', padx=2)
        self._play_btn = ttk.Button(pb, text="▶", width=3,
                                    command=self._toggle_play)
        self._play_btn.pack(side='left', padx=2)
        ttk.Button(pb, text="▶",  width=3,
                   command=self._t_next).pack(side='left', padx=2)
        ttk.Button(pb, text="▶▶", width=3,
                   command=self._t_last).pack(side='left', padx=2)

        self._t_slider = ttk.Scale(pb, from_=0, to=1, orient='horizontal',
                                   command=self._on_t_change)
        self._t_slider.pack(side='left', fill='x', expand=True, padx=8)

        self._t_lbl = ttk.Label(pb, text="t = —", width=14,
                                 font=('Courier', 10, 'bold'))
        self._t_lbl.pack(side='left', padx=4)

        # Speed
        ttk.Label(pb, text="Speed (fps):").pack(side='left', padx=(12, 2))
        self.v_fps = tk.IntVar(value=4)
        ttk.Spinbox(pb, from_=1, to=30, textvariable=self.v_fps,
                    width=4).pack(side='left')

        ttk.Checkbutton(pb, text="Live", variable=self.v_live).pack(
            side='left', padx=(12, 2))

        # Save row
        sv = ttk.Frame(parent)
        sv.pack(fill='x', pady=(4, 0))

        ttk.Button(sv, text="Save PNG",
                   command=self._save_png).pack(side='left', padx=4)

        ttk.Label(sv, text="  Save GIF  t₀ (s):").pack(side='left')
        ttk.Entry(sv, textvariable=self.v_gif_t0, width=8).pack(side='left', padx=2)
        ttk.Label(sv, text="t₁ (s):").pack(side='left', padx=(4, 0))
        ttk.Entry(sv, textvariable=self.v_gif_t1, width=8).pack(side='left', padx=2)
        ttk.Button(sv, text="Save GIF",
                   command=self._save_gif).pack(side='left', padx=4)
        self._gif_progress = ttk.Label(sv, text="", foreground='#226')
        self._gif_progress.pack(side='left', padx=4)

    # ── sounding ─────────────────────────────────────────────────────────────

    def _toggle_sounding_mode(self):
        if self._ds is None:
            messagebox.showwarning("No data", "Open a file first.")
            return
        self._sounding_mode = not self._sounding_mode
        if self._sounding_mode:
            self._snd_btn.config(text="Cancel Sounding")
            self._canvas.get_tk_widget().config(cursor='crosshair')
        else:
            self._snd_btn.config(text="Take Sounding")
            self._canvas.get_tk_widget().config(cursor='')

    def _on_canvas_click(self, event):
        if not self._sounding_mode or event.inaxes != self._ax:
            return
        if event.button != 1:
            return
        self._sounding_mode = False
        self._snd_btn.config(text="Take Sounding")
        self._canvas.get_tk_widget().config(cursor='')
        self._take_sounding(event.xdata, event.ydata)

    def _take_sounding(self, cx, cy):
        try:
            import sounderpy as spy
        except ImportError:
            messagebox.showerror("sounderpy missing",
                                 "Install sounderpy:  pip install sounderpy")
            return

        ds  = self._ds
        t   = self._t_idx
        xh, yh, zh = ds.xh, ds.yh, ds.zh
        view = self.v_view.get()
        frac = float(self._cs_slider.get()) if hasattr(self, '_cs_slider') else 0.5

        # Resolve clicked coordinates to grid indices
        if view == 'plan':
            xi = int(np.argmin(np.abs(xh - cx)))
            yi = int(np.argmin(np.abs(yh - cy)))
        elif view == 'xz':
            xi = int(np.argmin(np.abs(xh - cx)))
            yi = int(np.clip(frac * (len(yh) - 1), 0, len(yh) - 1))
        else:  # yz
            xi = int(np.clip(frac * (len(xh) - 1), 0, len(xh) - 1))
            yi = int(np.argmin(np.abs(yh - cx)))

        def _col(name):
            if name in ds.fields_3d:
                return ds.get_field(name, t)[:, yi, xi]
            return None

        th_col  = _col('th')
        prs_col = _col('prs')
        qv_col  = _col('qv')
        u_col   = _col('uinterp') if 'uinterp' in ds.fields_3d else _col('u')
        v_col   = _col('vinterp') if 'vinterp' in ds.fields_3d else _col('v')

        if th_col is None or prs_col is None:
            messagebox.showerror("Sounding error",
                                 "Need 'th' and 'prs' output fields to build a sounding.")
            return

        Rd, cp = 287.04, 1004.0
        T_c  = th_col * (prs_col / 1e5) ** (Rd / cp) - 273.15

        if qv_col is not None:
            qv_c  = np.maximum(qv_col, 1e-10)
            e_pa  = qv_c * prs_col / (0.622 + qv_c)
            e_pa  = np.maximum(e_pa, 1e-3)
            Td_c  = 243.5 * np.log(e_pa / 611.2) / (17.67 - np.log(e_pa / 611.2))
        else:
            Td_c  = T_c - 30.0

        from metpy.units import units as munits

        # zh in km → m
        z_m   = zh * 1000.0 if np.nanmax(zh) < 1000 else zh
        p_hpa = prs_col / 100.0
        u_ms  = u_col if u_col is not None else np.zeros_like(T_c)
        v_ms  = v_col if v_col is not None else np.zeros_like(T_c)
        u_kt  = u_ms * 1.94384
        v_kt  = v_ms * 1.94384

        clean_data = {
            'p':  p_hpa * munits('hPa'),
            'T':  T_c   * munits('degC'),
            'Td': Td_c  * munits('degC'),
            'u':  u_kt  * munits('kt'),
            'v':  v_kt  * munits('kt'),
            'z':  z_m   * munits('m'),
        }
        t_str = _sec_label(ds.times[t])
        loc_str = f'({xh[xi]:.1f}, {yh[yi]:.1f}) km'
        clean_data['site_info'] = {
            'site-name':   f'CM1  {loc_str}',
            'site-latlon': (0.0, 0.0),
            'site-elv':    0,
            'source':      'CM1',
            'model':       'CM1',
            'fcst-hour':   0,
            'run-time':    ['', '', '', ''],
            'valid-time':  ['', '', '', ''],
        }
        clean_data['titles'] = {
            'top_title':   f'CM1 Model Sounding  |  {loc_str}  |  t = {t_str}',
            'left_title':  'CM1',
            'right_title': '',
        }
        import matplotlib
        import matplotlib.pyplot as mplt
        import tempfile, subprocess, sys

        before = set(mplt.get_fignums())
        _orig_show = mplt.show
        mplt.show = lambda *a, **kw: None
        try:
            spy.build_sounding(clean_data, style='full', color_blind=False,
                               dark_mode=False)
        except Exception as e:
            mplt.show = _orig_show
            messagebox.showerror("sounderpy error", str(e))
            return
        finally:
            mplt.show = _orig_show

        new_figs = set(mplt.get_fignums()) - before
        if not new_figs:
            return
        fn = next(iter(new_figs))
        fig = mplt.figure(fn)
        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        fig.savefig(tmp.name, dpi=150, bbox_inches='tight')
        mplt.close(fig)
        tmp.close()
        if sys.platform == 'darwin':
            subprocess.Popen(['open', tmp.name])
        elif sys.platform.startswith('linux'):
            subprocess.Popen(['xdg-open', tmp.name])
        else:
            subprocess.Popen(['start', tmp.name], shell=True)

    # ── open ─────────────────────────────────────────────────────────────────

    def _open(self):
        paths = filedialog.askopenfilenames(
            title="Open CM1 netCDF file(s)",
            filetypes=[("NetCDF files", "*.nc"), ("All files", "*.*")])
        if not paths:
            return
        try:
            new_ds = CM1Dataset(list(paths))
        except Exception as e:
            messagebox.showerror("Load error", str(e))
            return
        self._stop_watching()
        names = [os.path.basename(p) for p in paths]
        label = (f"{len(names)} file(s): {', '.join(names[:3])}"
                 + ("…" if len(names) > 3 else ""))
        self._apply_dataset(new_ds, label)

    def _open_dir(self):
        directory = filedialog.askdirectory(title="Open CM1 output directory")
        if not directory:
            return
        paths = sorted(glob.glob(os.path.join(directory, '*.nc')))
        if not paths:
            messagebox.showwarning("No files", "No .nc files found in that directory.")
            return
        try:
            new_ds = CM1Dataset(paths)
        except Exception as e:
            messagebox.showerror("Load error", str(e))
            return
        self._stop_watching()
        self._apply_dataset(new_ds, f"{os.path.basename(directory)}/ ({len(paths)} files)")
        self._start_watching(directory, set(paths))

    def _apply_dataset(self, new_ds, label_text):
        if self._ds:
            self._ds.close()
        self._ds = new_ds

        self._file_lbl.config(text=label_text, foreground='black')

        all_fields = self._ds.fields_3d + self._ds.fields_2d
        self._all_fields = all_fields
        self._all_ctr_fields = [''] + all_fields
        self._field_cb['values'] = all_fields
        self._ctr_cb['values']   = self._all_ctr_fields
        self._cmap_cb['values']  = CMAPS
        if all_fields and not self.v_field.get():
            self.v_field.set(all_fields[0])
        self.v_ctr_field.set('')

        nt = self._ds.ntimes
        self._t_slider.config(to=max(nt - 1, 1))
        self._t_slider.set(0)
        self._t_idx = 0

        nk = len(self._ds.zh)
        self._z_slider.config(to=max(nk - 1, 1))
        self._z_slider.set(nk // 2)

        self.v_gif_t1.set(str(int(self._ds.times[-1])) if nt > 0 else '')
        self._plot()
        self._canvas.draw()

    def _start_watching(self, directory, known_paths):
        self._watch_dir   = directory
        self._watch_known = known_paths
        self._watch_lbl.config(text=f"⏺ Watching {os.path.basename(directory)}/")
        self._schedule_poll()

    def _stop_watching(self):
        if self._watch_id is not None:
            self.after_cancel(self._watch_id)
            self._watch_id = None
        self._watch_dir = None
        self._watch_known.clear()
        self._watch_lbl.config(text="")

    def _schedule_poll(self):
        self._watch_id = self.after(5000, self._poll_dir)

    def _poll_dir(self):
        if self._watch_dir is None or self._ds is None:
            return
        current = set(glob.glob(os.path.join(self._watch_dir, '*.nc')))
        new_paths = sorted(current - self._watch_known)
        if new_paths:
            n = self._ds.add_paths(new_paths)
            if n:
                self._watch_known |= set(new_paths)
                nt = self._ds.ntimes
                was_at_end = (self._t_idx >= nt - n - 1)
                self._t_slider.config(to=max(nt - 1, 1))
                self.v_gif_t1.set(str(int(self._ds.times[-1])))
                name = os.path.basename(self._watch_dir)
                self._watch_lbl.config(
                    text=f"⏺ Watching {name}/ ({len(self._watch_known)} files)")
                if self.v_live.get() and (was_at_end or not self._playing):
                    self._set_time(nt - 1)
        self._schedule_poll()

    # ── slider callbacks ─────────────────────────────────────────────────────

    def _on_t_change(self, val):
        if self._ds is None:
            return
        idx = min(int(float(val)), self._ds.ntimes - 1)
        if idx != self._t_idx:
            self._t_idx = idx
            self._plot()

    def _on_z_change(self, val):
        if self._ds is None:
            return
        self._update_z_label()
        self._plot()

    def _on_cs_change(self, val):
        if self._ds is None:
            return
        self._update_cs_label()
        self._plot()

    def _update_z_label(self):
        if self._ds is None:
            return
        ki = int(self._z_slider.get())
        ki = np.clip(ki, 0, len(self._ds.zh) - 1)
        self._z_lbl.config(text=f"z = {self._ds.zh[ki]:.3f} km")

    def _update_cs_label(self):
        if self._ds is None:
            return
        view = self.v_view.get()
        frac = float(self._cs_slider.get())
        if view == 'xz':
            coords = self._ds.yh
            val = coords[0] + frac * (coords[-1] - coords[0])
            self._cs_lbl.config(text=f"y = {val:.2f} km")
        elif view == 'yz':
            coords = self._ds.xh
            val = coords[0] + frac * (coords[-1] - coords[0])
            self._cs_lbl.config(text=f"x = {val:.2f} km")
        else:
            self._cs_lbl.config(text="(used in cross sections)")

    # ── time controls ────────────────────────────────────────────────────────

    def _t_first(self):
        self._set_time(0)

    def _t_last(self):
        if self._ds:
            self._set_time(self._ds.ntimes - 1)

    def _t_prev(self):
        self._set_time(max(0, self._t_idx - 1))

    def _t_next(self):
        if self._ds:
            self._set_time(min(self._ds.ntimes - 1, self._t_idx + 1))

    def _set_time(self, idx):
        self._t_idx = idx
        self._t_slider.set(idx)
        self._plot()

    def _toggle_play(self):
        if self._playing:
            self._playing = False
            if self._play_id is not None:
                self.after_cancel(self._play_id)
                self._play_id = None
            self._play_btn.config(text="▶")
        else:
            self._playing = True
            self._play_btn.config(text="⏸")
            self._advance_play()

    def _advance_play(self):
        if not self._playing or self._ds is None:
            return
        self._t_next()
        if self._t_idx >= self._ds.ntimes - 1:
            self._playing = False
            self._play_id = None
            self._play_btn.config(text="▶")
            return
        fps = max(1, self.v_fps.get())
        self._play_id = self.after(int(1000 / fps), self._advance_play)

    # ── colorbar range ───────────────────────────────────────────────────────

    def _reset_range(self):
        self.v_vmin.set('')
        self.v_vmax.set('')
        self._plot()

    def _reset_lims(self):
        for v in (self.v_xmin, self.v_xmax, self.v_ymin, self.v_ymax):
            v.set('')
        self.v_xlim_sym.set(False)
        self.v_ylim_sym.set(False)
        self._plot()

    def _on_xlim_sym(self):
        if self.v_xlim_sym.get():
            # Mirror whichever bound is already set
            try:
                self.v_xmin.set(f'{-float(self.v_xmax.get()):.6g}')
            except ValueError:
                try:
                    self.v_xmax.set(f'{-float(self.v_xmin.get()):.6g}')
                except ValueError:
                    pass
        self._plot()

    def _on_ylim_sym(self):
        if self.v_ylim_sym.get():
            try:
                self.v_ymin.set(f'{-float(self.v_ymax.get()):.6g}')
            except ValueError:
                try:
                    self.v_ymax.set(f'{-float(self.v_ymin.get()):.6g}')
                except ValueError:
                    pass
        self._plot()

    def _get_lims(self, v_min, v_max, v_sym, auto_data):
        try:
            lo = float(v_min.get())
        except ValueError:
            lo = None
        try:
            hi = float(v_max.get())
        except ValueError:
            hi = None
        if lo is None and hi is None:
            return None, None   # fully auto
        if lo is None:
            lo = -hi if v_sym.get() else float(np.nanmin(auto_data))
        if hi is None:
            hi = -lo if v_sym.get() else float(np.nanmax(auto_data))
        return lo, hi

    def _get_range(self, data):
        try:
            vmin = float(self.v_vmin.get())
        except ValueError:
            vmin = None
        try:
            vmax = float(self.v_vmax.get())
        except ValueError:
            vmax = None
        if vmin is None or vmax is None:
            finite = data[np.isfinite(data)]
            dmin = float(finite.min()) if len(finite) else 0.0
            dmax = float(finite.max()) if len(finite) else 1.0
            if vmin is None:
                vmin = dmin
            if vmax is None:
                vmax = dmax
        if self.v_symcb.get():
            amax = max(abs(vmin), abs(vmax))
            vmin, vmax = -amax, amax
        return vmin, vmax

    # ── plot ─────────────────────────────────────────────────────────────────

    def _plot(self):
        if self._ds is None:
            return
        field = self.v_field.get()
        if not field:
            return

        try:
            data = self._ds.get_field(field, self._t_idx)
        except Exception as e:
            self._ax.clear()
            self._ax.text(0.5, 0.5, f"Error reading field:\n{e}",
                          ha='center', va='center',
                          transform=self._ax.transAxes, color='red')
            self._canvas.draw_idle()
            return

        t_sec  = self._ds.times[self._t_idx]
        t_str  = _sec_label(t_sec)
        self._t_lbl.config(text=f"t = {t_str}")
        self._t_slider.set(self._t_idx)
        self._update_z_label()
        self._update_cs_label()

        view   = self.v_view.get()
        cmap   = self.v_cmap.get()
        xh     = self._ds.xh
        yh     = self._ds.yh
        zh     = self._ds.zh

        ax = self._ax
        ax.clear()

        is_3d = data.ndim == 3

        if not is_3d:
            # 2-D surface field: data is (ny, nx)
            plot_data = data
            vmin, vmax = self._get_range(plot_data)
            im = ax.pcolormesh(xh, yh, plot_data,
                               cmap=cmap, norm=Normalize(vmin, vmax),
                               shading='nearest')
            self._add_colorbar(im, field)
            ax.set_xlabel('x  (km)')
            ax.set_ylabel('y  (km)')
            ax.set_title(
                f"{field}  —  t = {t_str}  [{self._ds.get_units(field)}]",
                fontsize=10)
            self._canvas.draw_idle()
            return

        # 3-D field
        ki  = int(np.clip(self._z_slider.get(), 0, len(zh) - 1))
        frac = float(self._cs_slider.get())

        if view == 'plan':
            # data is (nz, ny, nx); select level ki → (ny, nx)
            plot_data = data[ki, :, :]
            vmin, vmax = self._get_range(plot_data)
            im = ax.pcolormesh(xh, yh, plot_data,
                               cmap=cmap, norm=Normalize(vmin, vmax),
                               shading='nearest')
            self._add_colorbar(im, field)
            if self.v_winds.get():
                self._overlay_winds_plan(ax, ki, xh, yh, t_sec)
            ax.set_xlabel('x  (km)')
            ax.set_ylabel('y  (km)')
            ax.set_title(
                f"{field}  —  z = {zh[ki]:.2f} km,  t = {t_str}"
                f"  [{self._ds.get_units(field)}]",
                fontsize=10)

        elif view == 'xz':
            ji = int(np.clip(frac * (len(yh) - 1), 0, len(yh) - 1))
            # data is (nz, ny, nx); select y=ji → (nz, nx)
            plot_data = data[:, ji, :]
            vmin, vmax = self._get_range(plot_data)
            im = ax.pcolormesh(xh, zh, plot_data,
                               cmap=cmap, norm=Normalize(vmin, vmax),
                               shading='nearest')
            self._add_colorbar(im, field)
            if self.v_winds.get():
                self._overlay_winds_xz(ax, ji, xh, zh, t_sec)
            ax.set_xlabel('x  (km)')
            ax.set_ylabel('z  (km)')
            ax.set_title(
                f"{field}  —  y = {yh[ji]:.2f} km,  t = {t_str}"
                f"  [{self._ds.get_units(field)}]",
                fontsize=10)

        elif view == 'yz':
            ii = int(np.clip(frac * (len(xh) - 1), 0, len(xh) - 1))
            # data is (nz, ny, nx); select x=ii → (nz, ny)
            plot_data = data[:, :, ii]
            vmin, vmax = self._get_range(plot_data)
            im = ax.pcolormesh(yh, zh, plot_data,
                               cmap=cmap, norm=Normalize(vmin, vmax),
                               shading='nearest')
            self._add_colorbar(im, field)
            if self.v_winds.get():
                self._overlay_winds_yz(ax, ii, yh, zh, t_sec)
            ax.set_xlabel('y  (km)')
            ax.set_ylabel('z  (km)')
            ax.set_title(
                f"{field}  —  x = {xh[ii]:.2f} km,  t = {t_str}"
                f"  [{self._ds.get_units(field)}]",
                fontsize=10)

        ax.set_aspect('auto')
        self._draw_contour(ax, view, ki, frac, xh, yh, zh)

        # Apply user axis limits (x = horizontal axis, y = vertical axis)
        xlim = self._get_lims(self.v_xmin, self.v_xmax, self.v_xlim_sym,
                               np.array(ax.get_xlim()))
        ylim = self._get_lims(self.v_ymin, self.v_ymax, self.v_ylim_sym,
                               np.array(ax.get_ylim()))
        if xlim[0] is not None:
            ax.set_xlim(xlim)
        if ylim[0] is not None:
            ax.set_ylim(ylim)

        self._canvas.draw_idle()

    def _draw_contour(self, ax, view, ki, frac, xh, yh, zh):
        ctr_field = self.v_ctr_field.get()
        if not ctr_field:
            return
        try:
            cdata = self._ds.get_field(ctr_field, self._t_idx)
        except Exception:
            return

        import matplotlib.cm as mpl_cm
        color_val = self.v_ctr_color.get()
        nlevs     = max(2, self.v_ctr_levels.get())
        is_cmap   = color_val in plt.colormaps()

        # slice same way as pcolormesh
        if cdata.ndim == 2:
            if view == 'plan':
                cslice, X, Y = cdata, xh, yh
            elif view == 'xz':
                ji = int(np.clip(frac * (len(yh) - 1), 0, len(yh) - 1))
                cslice, X, Y = cdata, xh, zh
            else:
                ii = int(np.clip(frac * (len(xh) - 1), 0, len(xh) - 1))
                cslice, X, Y = cdata, yh, zh
        else:
            if view == 'plan':
                cslice, X, Y = cdata[ki, :, :], xh, yh
            elif view == 'xz':
                ji = int(np.clip(frac * (len(yh) - 1), 0, len(yh) - 1))
                cslice, X, Y = cdata[:, ji, :], xh, zh
            else:
                ii = int(np.clip(frac * (len(xh) - 1), 0, len(xh) - 1))
                cslice, X, Y = cdata[:, :, ii], yh, zh

        try:
            style    = self.v_ctr_style.get()
            label    = self.v_ctr_labels.get()
            lw       = max(0.1, self.v_ctr_lw.get())
            color_kw = {'cmap': color_val} if is_cmap else {'colors': color_val}

            # resolve explicit min/max/sym
            try:    clo = float(self.v_ctr_min.get())
            except ValueError: clo = None
            try:    chi = float(self.v_ctr_max.get())
            except ValueError: chi = None
            sym = self.v_ctr_sym.get()
            if clo is None and chi is None:
                clo = float(np.nanmin(cslice))
                chi = float(np.nanmax(cslice))
            else:
                if sym:
                    if chi is not None and clo is None:
                        clo = -chi
                    elif clo is not None and chi is None:
                        chi = -clo
                clo = clo if clo is not None else float(np.nanmin(cslice))
                chi = chi if chi is not None else float(np.nanmax(cslice))
            levels = np.linspace(clo, chi, nlevs)

            if style == 'pn':
                pos = [l for l in levels if l > 0]
                neg = [l for l in levels if l < 0]
                for levs, ls in [(pos, 'solid'), (neg, 'dashed')]:
                    if not levs:
                        continue
                    cs = ax.contour(X, Y, cslice, levels=levs,
                                    linewidths=lw, linestyles=ls, **color_kw)
                    if label:
                        ax.clabel(cs, inline=True, fontsize=7, fmt='%g')
            else:
                cs = ax.contour(X, Y, cslice, levels=levels, linewidths=lw,
                                linestyles=style, **color_kw)
                if label:
                    ax.clabel(cs, inline=True, fontsize=7, fmt='%g')
        except Exception:
            pass

    def _add_colorbar(self, im, field):
        if self._cbar is None:
            self._cbar = self._fig.colorbar(im, cax=self._cax)
        else:
            self._cbar.update_normal(im)
        units = self._ds.get_units(field) or ''
        self._cbar.set_label(units, fontsize=9)

    # ── wind overlays ────────────────────────────────────────────────────────

    def _wind_vars(self, view):
        """Return (u_name, v_name) appropriate for the view."""
        if view == 'plan':
            return 'uinterp', 'vinterp'
        elif view == 'xz':
            return 'uinterp', 'winterp'
        else:
            return 'vinterp', 'winterp'

    def _draw_wind(self, ax, x1d, y1d, u, v):
        """Draw quiver arrows or barbs depending on user selection."""
        if self.v_wind_type.get() == 'barbs':
            ax.barbs(x1d, y1d, u, v, color='k', alpha=0.7,
                     length=6, linewidth=0.8)
        else:
            ax.quiver(x1d, y1d, u, v,
                      scale=None, color='k', alpha=0.6, width=0.002)

    def _overlay_winds_plan(self, ax, ki, xh, yh, t_sec):
        # wind data is (nz, ny, nx); select level ki → (ny, nx)
        sk = max(1, self.v_wind_skip.get())
        for u_n, v_n in [('uinterp', 'vinterp'), ('u', 'v')]:
            if u_n in self._ds.fields_3d and v_n in self._ds.fields_3d:
                u = self._ds.get_field(u_n, self._t_idx)[ki, ::sk, ::sk]
                v = self._ds.get_field(v_n, self._t_idx)[ki, ::sk, ::sk]
                self._draw_wind(ax, xh[::sk], yh[::sk], u, v)
                return

    def _overlay_winds_xz(self, ax, ji, xh, zh, t_sec):
        # wind data is (nz, ny, nx); select y=ji → (nz, nx)
        sk = max(1, self.v_wind_skip.get())
        for u_n in ['uinterp', 'u']:
            for w_n in ['winterp', 'w']:
                if u_n in self._ds.fields_3d and w_n in self._ds.fields_3d:
                    u = self._ds.get_field(u_n, self._t_idx)[::sk, ji, ::sk]
                    w = self._ds.get_field(w_n, self._t_idx)[::sk, ji, ::sk]
                    self._draw_wind(ax, xh[::sk], zh[::sk], u, w)
                    return

    def _overlay_winds_yz(self, ax, ii, yh, zh, t_sec):
        # wind data is (nz, ny, nx); select x=ii → (nz, ny)
        sk = max(1, self.v_wind_skip.get())
        for v_n in ['vinterp', 'v']:
            for w_n in ['winterp', 'w']:
                if v_n in self._ds.fields_3d and w_n in self._ds.fields_3d:
                    v = self._ds.get_field(v_n, self._t_idx)[::sk, ::sk, ii]
                    w = self._ds.get_field(w_n, self._t_idx)[::sk, ::sk, ii]
                    self._draw_wind(ax, yh[::sk], zh[::sk], v, w)
                    return

    # ── save PNG ─────────────────────────────────────────────────────────────

    def _save_png(self):
        if self._ds is None:
            messagebox.showwarning("No data", "Open a file first.")
            return
        field = self.v_field.get()
        t_sec = self._ds.times[self._t_idx]
        default = f"{field}_t{int(t_sec):06d}.png"
        path = filedialog.asksaveasfilename(
            initialfile=default,
            defaultextension='.png',
            filetypes=[("PNG", "*.png"), ("All", "*.*")],
            title="Save frame as PNG")
        if path:
            self._fig.savefig(path, dpi=150, bbox_inches='tight')
            self._gif_progress.config(text=f"Saved: {os.path.basename(path)}")

    # ── save GIF ─────────────────────────────────────────────────────────────

    def _save_gif(self):
        if self._ds is None:
            messagebox.showwarning("No data", "Open a file first.")
            return
        path = filedialog.asksaveasfilename(
            initialfile=f"{self.v_field.get()}.gif",
            defaultextension='.gif',
            filetypes=[("GIF", "*.gif"), ("All", "*.*")],
            title="Save animation as GIF")
        if not path:
            return
        threading.Thread(target=self._gif_worker, args=(path,),
                         daemon=True).start()

    def _gif_worker(self, path):
        try:
            t0 = float(self.v_gif_t0.get())
        except ValueError:
            t0 = self._ds.times[0]
        try:
            t1 = float(self.v_gif_t1.get())
        except ValueError:
            t1 = self._ds.times[-1]

        times = self._ds.times
        idx_range = [i for i, t in enumerate(times) if t0 <= t <= t1]
        if not idx_range:
            self.after(0, lambda: messagebox.showwarning(
                "GIF", "No time steps in the selected range."))
            return

        fps  = max(1, self.v_fps.get())
        n    = len(idx_range)
        self.after(0, lambda: self._gif_progress.config(
            text=f"Saving {n} frames…"))

        writer = PillowWriter(fps=fps)
        with writer.saving(self._fig, path, dpi=120):
            for k, ti in enumerate(idx_range):
                self._t_idx = ti
                self._plot()
                self._fig.canvas.draw()
                writer.grab_frame()
                msg = f"Frame {k+1}/{n}…"
                self.after(0, lambda m=msg: self._gif_progress.config(text=m))

        base = os.path.basename(path)
        self.after(0, lambda: self._gif_progress.config(
            text=f"Saved {n} frames → {base}"))


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    app = CM1Viewer()
    app.protocol('WM_DELETE_WINDOW', lambda: (app._stop_watching(), app.destroy()))
    app.mainloop()
