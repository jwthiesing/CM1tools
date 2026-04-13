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

        self._ds       = None
        self._t_idx    = 0
        self._play_id  = None
        self._last_field = None

        self._build_vars()
        self._build_ui()

    # ── variables ────────────────────────────────────────────────────────────

    def _build_vars(self):
        self.v_field   = tk.StringVar()
        self.v_cmap    = tk.StringVar(value='RdBu_r')
        self.v_view    = tk.StringVar(value='plan')      # plan | xz | yz
        self.v_symcb   = tk.BooleanVar(value=False)
        self.v_vmin    = tk.StringVar(value='')
        self.v_vmax    = tk.StringVar(value='')
        self.v_winds   = tk.BooleanVar(value=False)
        self.v_wind_skip = tk.IntVar(value=4)
        self.v_gif_t0  = tk.StringVar(value='0')
        self.v_gif_t1  = tk.StringVar(value='')

    # ── UI ───────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── top bar: file + field ────────────────────────────────────────
        top = ttk.Frame(self)
        top.pack(fill='x', padx=6, pady=4)

        ttk.Button(top, text="Open file(s)…", command=self._open).pack(side='left', padx=4)
        self._file_lbl = ttk.Label(top, text="No file loaded.", foreground='gray')
        self._file_lbl.pack(side='left', padx=8)

        ttk.Label(top, text="Field:").pack(side='left')
        self._field_cb = ttk.Combobox(top, textvariable=self.v_field, width=18,
                                      state='readonly')
        self._field_cb.pack(side='left', padx=4)
        self._field_cb.bind('<<ComboboxSelected>>', lambda _: self._plot())

        ttk.Label(top, text="Colormap:").pack(side='left', padx=(12, 2))
        ttk.Combobox(top, textvariable=self.v_cmap, values=CMAPS,
                     width=14, state='readonly').pack(side='left', padx=4)
        self.v_cmap.trace_add('write', lambda *_: self._plot())

        # ── main paned window ────────────────────────────────────────────
        pane = tk.PanedWindow(self, orient='horizontal', sashwidth=5,
                              sashrelief='ridge')
        pane.pack(fill='both', expand=True, padx=6, pady=2)

        # --- Left controls ---
        left = ttk.Frame(pane, width=210)
        pane.add(left, minsize=200)
        self._build_controls(left)

        # --- Right: plot ---
        right = ttk.Frame(pane)
        pane.add(right, minsize=600)

        self._fig  = Figure(figsize=(7, 5.5), dpi=100)
        self._ax   = self._fig.add_subplot(111)
        self._canvas = FigureCanvasTkAgg(self._fig, master=right)
        self._canvas.get_tk_widget().pack(fill='both', expand=True)
        tb = NavigationToolbar2Tk(self._canvas, right, pack_toolbar=False)
        tb.pack(fill='x')

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

        # Wind overlay
        wf = ttk.LabelFrame(parent, text="Wind overlay")
        wf.pack(fill='x', padx=6, pady=4)
        ttk.Checkbutton(wf, text="Show vectors",
                        variable=self.v_winds,
                        command=self._plot).pack(anchor='w', padx=6, pady=2)
        row = ttk.Frame(wf)
        row.pack(fill='x', padx=6, pady=2)
        ttk.Label(row, text="Skip N:").pack(side='left')
        ttk.Spinbox(row, from_=1, to=20, textvariable=self.v_wind_skip,
                    width=4, command=self._plot).pack(side='left', padx=4)

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

    # ── open ─────────────────────────────────────────────────────────────────

    def _open(self):
        paths = filedialog.askopenfilenames(
            title="Open CM1 netCDF file(s)",
            filetypes=[("NetCDF files", "*.nc *.nc4"), ("All files", "*.*")])
        if not paths:
            return
        try:
            if self._ds:
                self._ds.close()
            self._ds = CM1Dataset(list(paths))
        except Exception as e:
            messagebox.showerror("Load error", str(e))
            return

        names = [os.path.basename(p) for p in paths]
        self._file_lbl.config(
            text=f"{len(names)} file(s): {', '.join(names[:3])}"
                 + ("…" if len(names) > 3 else ""),
            foreground='black')

        all_fields = self._ds.fields_3d + self._ds.fields_2d
        self._field_cb['values'] = all_fields
        if all_fields:
            self.v_field.set(all_fields[0])

        nt = self._ds.ntimes
        self._t_slider.config(to=max(nt - 1, 1))
        self._t_slider.set(0)
        self._t_idx = 0

        # z slider
        nk = len(self._ds.zh)
        self._z_slider.config(to=max(nk - 1, 1))
        self._z_slider.set(nk // 2)

        self.v_gif_t1.set(str(int(self._ds.times[-1])) if nt > 0 else '')

        self._plot()

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
        if self._play_id is not None:
            self.after_cancel(self._play_id)
            self._play_id = None
            self._play_btn.config(text="▶")
        else:
            self._play_btn.config(text="⏸")
            self._advance_play()

    def _advance_play(self):
        if self._ds is None or self._play_id is None and self._play_btn['text'] == '▶':
            return
        self._t_next()
        if self._t_idx >= self._ds.ntimes - 1:
            self._toggle_play()
            return
        fps = max(1, self.v_fps.get())
        self._play_id = self.after(int(1000 / fps), self._advance_play)

    # ── colorbar range ───────────────────────────────────────────────────────

    def _reset_range(self):
        self.v_vmin.set('')
        self.v_vmax.set('')
        self._plot()

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
            # 2-D surface field
            plot_data = data
            vmin, vmax = self._get_range(plot_data)
            im = ax.pcolormesh(xh, yh, plot_data.T,
                               cmap=cmap, norm=Normalize(vmin, vmax),
                               shading='auto')
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
            plot_data = data[:, :, ki]
            vmin, vmax = self._get_range(plot_data)
            im = ax.pcolormesh(xh, yh, plot_data.T,
                               cmap=cmap, norm=Normalize(vmin, vmax),
                               shading='auto')
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
            plot_data = data[:, ji, :]
            vmin, vmax = self._get_range(plot_data)
            im = ax.pcolormesh(xh, zh, plot_data.T,
                               cmap=cmap, norm=Normalize(vmin, vmax),
                               shading='auto')
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
            plot_data = data[ii, :, :]
            vmin, vmax = self._get_range(plot_data)
            im = ax.pcolormesh(yh, zh, plot_data.T,
                               cmap=cmap, norm=Normalize(vmin, vmax),
                               shading='auto')
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
        self._canvas.draw_idle()

    def _add_colorbar(self, im, field):
        try:
            self._cbar.remove()
        except Exception:
            pass
        self._cbar = self._fig.colorbar(im, ax=self._ax, fraction=0.046, pad=0.04)
        units = self._ds.get_units(field)
        if units:
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

    def _overlay_winds_plan(self, ax, ki, xh, yh, t_sec):
        sk = max(1, self.v_wind_skip.get())
        for u_n, v_n in [('uinterp', 'vinterp'), ('u', 'v')]:
            if u_n in self._ds.fields_3d and v_n in self._ds.fields_3d:
                u = self._ds.get_field(u_n, self._t_idx)[:, :, ki][::sk, ::sk]
                v = self._ds.get_field(v_n, self._t_idx)[:, :, ki][::sk, ::sk]
                ax.quiver(xh[::sk], yh[::sk], u.T, v.T,
                          scale=None, color='k', alpha=0.6, width=0.002)
                return

    def _overlay_winds_xz(self, ax, ji, xh, zh, t_sec):
        sk = max(1, self.v_wind_skip.get())
        for u_n in ['uinterp', 'u']:
            for w_n in ['winterp', 'w']:
                if u_n in self._ds.fields_3d and w_n in self._ds.fields_3d:
                    u = self._ds.get_field(u_n, self._t_idx)[:, ji, :][::sk, ::sk]
                    w = self._ds.get_field(w_n, self._t_idx)[:, ji, :][::sk, ::sk]
                    ax.quiver(xh[::sk], zh[::sk], u.T, w.T,
                              scale=None, color='k', alpha=0.6, width=0.002)
                    return

    def _overlay_winds_yz(self, ax, ii, yh, zh, t_sec):
        sk = max(1, self.v_wind_skip.get())
        for v_n in ['vinterp', 'v']:
            for w_n in ['winterp', 'w']:
                if v_n in self._ds.fields_3d and w_n in self._ds.fields_3d:
                    v = self._ds.get_field(v_n, self._t_idx)[ii, :, :][::sk, ::sk]
                    w = self._ds.get_field(w_n, self._t_idx)[ii, :, :][::sk, ::sk]
                    ax.quiver(yh[::sk], zh[::sk], v.T, w.T,
                              scale=None, color='k', alpha=0.6, width=0.002)
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
    app.mainloop()
