#!/opt/miniconda3/envs/met/bin/python3
"""soundingtool.py — Fetch sounderpy sounding data, compute Bunkers storm motion,
and export to CM1 input_sounding format."""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import numpy as np

# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def _mag(arr, unit):
    """Strip pint units and return plain numpy array in the requested unit."""
    try:
        return np.asarray(arr.to(unit).magnitude, dtype=float)
    except AttributeError:
        return np.asarray(arr, dtype=float)


def _potential_temperature(T_C, p_hPa):
    return (T_C + 273.15) * (1000.0 / p_hPa) ** 0.2854


def _mixing_ratio(Td_C, p_hPa):
    """Bolton (1980) water-vapor mixing ratio in g/kg."""
    e = 6.112 * np.exp(17.67 * Td_C / (Td_C + 243.5))
    return np.clip(621.97 * e / np.maximum(p_hPa - e, 1e-6), 0.0, None)


# ---------------------------------------------------------------------------
# Bunkers right-mover storm motion
# ---------------------------------------------------------------------------

def compute_bunkers(clean_data):
    """Return (u_rm, v_rm) Bunkers right-mover in m/s using MetPy.

    Returns (None, None) if MetPy is unavailable or insufficient data.
    """
    try:
        from metpy.calc import bunkers_storm_motion
    except ImportError:
        return None, None

    try:
        p = clean_data['p']   # pint hPa, surface-first (descending)
        u = clean_data['u'].to('m/s')
        v = clean_data['v'].to('m/s')
        z = clean_data['z']   # pint m

        rm, _lm, _mean = bunkers_storm_motion(p, u, v, z)
        u_rm = float(rm[0].to('m/s').magnitude)
        v_rm = float(rm[1].to('m/s').magnitude)
        return u_rm, v_rm
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# CM1 input_sounding builder
# ---------------------------------------------------------------------------

def build_cm1_sounding(clean_data):
    """Convert sounderpy clean_data to CM1 input_sounding lines.

    Format
    ------
    Line 1 (header):  sfc_pres(mb)  sfc_theta(K)  sfc_qv(g/kg)
    Lines 2+:         z(m)  theta(K)  qv(g/kg)  u(m/s)  v(m/s)

    The last z must exceed the model top.
    """
    p   = _mag(clean_data['p'],  'hPa')
    z   = _mag(clean_data['z'],  'm')
    T   = _mag(clean_data['T'],  'degC')
    Td  = _mag(clean_data['Td'], 'degC')
    u   = _mag(clean_data['u'],  'm/s')
    v   = _mag(clean_data['v'],  'm/s')

    # Sort ascending by height; remove duplicate z values
    order  = np.argsort(z)
    p, z, T, Td, u, v = p[order], z[order], T[order], Td[order], u[order], v[order]
    _, unique = np.unique(z, return_index=True)
    p, z, T, Td, u, v = p[unique], z[unique], T[unique], Td[unique], u[unique], v[unique]

    theta = _potential_temperature(T, p)
    qv    = _mixing_ratio(Td, p)

    lines = []
    lines.append(f"  {p[0]:10.4f}  {theta[0]:12.6f}  {qv[0]:12.6f}")
    for i in range(len(z)):
        lines.append(
            f"  {z[i]:12.4f}  {theta[i]:12.6f}  {qv[i]:12.6f}"
            f"  {u[i]:12.6f}  {v[i]:12.6f}"
        )
    return lines


def build_info_text(clean_data, u_rm, v_rm):
    """Build a human-readable summary file content."""
    si    = clean_data.get('site_info', {}) or {}
    lines = [
        "CM1 Sounding Summary",
        "=" * 40,
        f"Station   : {si.get('site-id','?')} — {si.get('site-name','')}",
        f"Location  : {si.get('site-lctn','')}",
        f"Lat/Lon   : {si.get('site-latlon','?')}",
        f"Valid time: {si.get('valid-time','?')}",
        f"Source    : {si.get('source','?')}",
        "",
        "Storm Motion (Bunkers Right Mover)",
        "-" * 40,
    ]
    if u_rm is not None:
        lines += [
            f"  umove = {u_rm:7.3f}  m/s",
            f"  vmove = {v_rm:7.3f}  m/s",
            "",
            "  (paste these into CM1 namelist.input &param1)",
        ]
    else:
        lines.append("  Could not compute (insufficient sounding depth?)")
    lines += ["", f"Levels in input_sounding: {len(clean_data['p'])}"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class SoundingTool(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("CM1 Sounding Tool")
        self.minsize(860, 580)
        self._clean_data = None
        self._cm1_lines  = None
        self._u_rm       = None
        self._v_rm       = None
        self._build_ui()

    # ── layout ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        P = {"padx": 6, "pady": 4}

        # Source notebook
        nb = ttk.Notebook(self)
        nb.pack(fill="x", padx=10, pady=(10, 4))
        self._nb = nb

        self._tab_obs   = ttk.Frame(nb)
        self._tab_model = ttk.Frame(nb)
        nb.add(self._tab_obs,   text="Observed (RAOB / IGRA)")
        nb.add(self._tab_model, text="Model reanalysis (RAP-RUC / ERA5 / NCEP)")
        self._build_obs_tab()
        self._build_model_tab()

        # Output directory
        out = ttk.LabelFrame(self, text="Output  →  {base_dir}/output/")
        out.pack(fill="x", padx=10, pady=4)

        ttk.Label(out, text="Base directory:").grid(row=0, column=0, sticky="e", **P)
        self._outdir = tk.StringVar(
            value=os.path.expanduser("~/Documents/Code/cm1/run"))
        ttk.Entry(out, textvariable=self._outdir, width=55).grid(
            row=0, column=1, sticky="w", **P)
        ttk.Button(out, text="Browse…", command=self._browse_dir).grid(
            row=0, column=2, sticky="w", **P)

        # Bunkers result bar
        bunk = ttk.LabelFrame(self, text="Bunkers Right Mover  (CM1 storm motion)")
        bunk.pack(fill="x", padx=10, pady=4)

        ttk.Label(bunk, text="umove =").grid(row=0, column=0, sticky="e", **P)
        self._umove_var = tk.StringVar(value="—")
        ttk.Label(bunk, textvariable=self._umove_var, width=10,
                  font=("Courier", 10, "bold")).grid(row=0, column=1, sticky="w", **P)
        ttk.Label(bunk, text="m/s").grid(row=0, column=2, sticky="w")

        ttk.Label(bunk, text="vmove =").grid(row=0, column=3, sticky="e", **P)
        self._vmove_var = tk.StringVar(value="—")
        ttk.Label(bunk, textvariable=self._vmove_var, width=10,
                  font=("Courier", 10, "bold")).grid(row=0, column=4, sticky="w", **P)
        ttk.Label(bunk, text="m/s").grid(row=0, column=5, sticky="w")

        # Buttons
        btn = ttk.Frame(self)
        btn.pack(fill="x", padx=10, pady=4)
        self._fetch_btn = ttk.Button(btn, text="Fetch sounding", command=self._fetch)
        self._fetch_btn.pack(side="left", **P)
        self._save_btn = ttk.Button(btn, text="Save outputs to subfolder",
                                    command=self._save, state="disabled")
        self._save_btn.pack(side="left", **P)

        # Preview
        prev = ttk.LabelFrame(self, text="input_sounding preview")
        prev.pack(fill="both", expand=True, padx=10, pady=4)
        self._preview = scrolledtext.ScrolledText(
            prev, height=14, font=("Courier", 10), state="disabled")
        self._preview.pack(fill="both", expand=True, **P)

        # Status bar
        self._status = tk.StringVar(value="Ready.")
        ttk.Label(self, textvariable=self._status, anchor="w",
                  relief="sunken").pack(fill="x", padx=10, pady=(0, 8))

    # ── observed tab ────────────────────────────────────────────────────────

    def _build_obs_tab(self):
        f, P = self._tab_obs, {"padx": 6, "pady": 6}

        ttk.Label(f, text="Station ID:").grid(row=0, column=0, sticky="e", **P)
        self._obs_station = tk.StringVar(value="OUN")
        ttk.Entry(f, textvariable=self._obs_station, width=12).grid(
            row=0, column=1, sticky="w", **P)
        ttk.Label(f, text="(3-letter RAOB  or  11-digit IGRAv2)",
                  foreground="gray").grid(row=0, column=2, columnspan=4, sticky="w", **P)

        ttk.Label(f, text="Year:").grid(row=1, column=0, sticky="e", **P)
        self._obs_year = tk.StringVar(value="2026")
        ttk.Entry(f, textvariable=self._obs_year, width=6).grid(
            row=1, column=1, sticky="w", **P)

        ttk.Label(f, text="Month:").grid(row=1, column=2, sticky="e", **P)
        self._obs_month = tk.StringVar(value="04")
        ttk.Combobox(f, textvariable=self._obs_month, width=5,
                     values=[f"{m:02d}" for m in range(1, 13)],
                     state="readonly").grid(row=1, column=3, sticky="w", **P)

        ttk.Label(f, text="Day:").grid(row=1, column=4, sticky="e", **P)
        self._obs_day = tk.StringVar(value="13")
        ttk.Combobox(f, textvariable=self._obs_day, width=5,
                     values=[f"{d:02d}" for d in range(1, 32)],
                     state="readonly").grid(row=1, column=5, sticky="w", **P)

        ttk.Label(f, text="Hour (UTC):").grid(row=1, column=6, sticky="e", **P)
        self._obs_hour = tk.StringVar(value="00")
        ttk.Combobox(f, textvariable=self._obs_hour, width=5,
                     values=["00", "06", "12", "18"],
                     state="readonly").grid(row=1, column=7, sticky="w", **P)

    # ── model tab ───────────────────────────────────────────────────────────

    def _build_model_tab(self):
        f, P = self._tab_model, {"padx": 6, "pady": 6}

        ttk.Label(f, text="Model:").grid(row=0, column=0, sticky="e", **P)
        self._model_name = tk.StringVar(value="rap-ruc")
        ttk.Combobox(f, textvariable=self._model_name, width=10,
                     values=["rap-ruc", "era5", "ncep"],
                     state="readonly").grid(row=0, column=1, sticky="w", **P)

        ttk.Label(f, text="Dataset (optional):").grid(row=0, column=2, sticky="e", **P)
        self._model_dataset = tk.StringVar(value="")
        ttk.Entry(f, textvariable=self._model_dataset, width=10).grid(
            row=0, column=3, sticky="w", **P)
        ttk.Label(f, text="(blank = auto)", foreground="gray").grid(
            row=0, column=4, sticky="w", **P)

        ttk.Label(f, text="Latitude:").grid(row=1, column=0, sticky="e", **P)
        self._model_lat = tk.StringVar(value="35.23")
        ttk.Entry(f, textvariable=self._model_lat, width=10).grid(
            row=1, column=1, sticky="w", **P)

        ttk.Label(f, text="Longitude:").grid(row=1, column=2, sticky="e", **P)
        self._model_lon = tk.StringVar(value="-97.46")
        ttk.Entry(f, textvariable=self._model_lon, width=10).grid(
            row=1, column=3, sticky="w", **P)

        ttk.Label(f, text="Box avg (deg):").grid(row=1, column=4, sticky="e", **P)
        self._model_box = tk.StringVar(value="0.10")
        ttk.Entry(f, textvariable=self._model_box, width=6).grid(
            row=1, column=5, sticky="w", **P)

        ttk.Label(f, text="Year:").grid(row=2, column=0, sticky="e", **P)
        self._model_year = tk.StringVar(value="2026")
        ttk.Entry(f, textvariable=self._model_year, width=6).grid(
            row=2, column=1, sticky="w", **P)

        ttk.Label(f, text="Month:").grid(row=2, column=2, sticky="e", **P)
        self._model_month = tk.StringVar(value="04")
        ttk.Combobox(f, textvariable=self._model_month, width=5,
                     values=[f"{m:02d}" for m in range(1, 13)],
                     state="readonly").grid(row=2, column=3, sticky="w", **P)

        ttk.Label(f, text="Day:").grid(row=2, column=4, sticky="e", **P)
        self._model_day = tk.StringVar(value="13")
        ttk.Combobox(f, textvariable=self._model_day, width=5,
                     values=[f"{d:02d}" for d in range(1, 32)],
                     state="readonly").grid(row=2, column=5, sticky="w", **P)

        ttk.Label(f, text="Hour (UTC):").grid(row=2, column=6, sticky="e", **P)
        self._model_hour = tk.StringVar(value="00")
        ttk.Combobox(f, textvariable=self._model_hour, width=5,
                     values=["00", "03", "06", "09", "12", "15", "18", "21"],
                     state="readonly").grid(row=2, column=7, sticky="w", **P)

    # ── helpers ─────────────────────────────────────────────────────────────

    def _browse_dir(self):
        path = filedialog.askdirectory(title="Select base output directory")
        if path:
            self._outdir.set(path)

    def _set_status(self, msg):
        self._status.set(msg)
        self.update_idletasks()

    def _set_preview(self, text):
        self._preview.config(state="normal")
        self._preview.delete("1.0", "end")
        self._preview.insert("end", text)
        self._preview.config(state="disabled")

    def _file_suffix(self):
        """Generate a location+datetime suffix for output filenames."""
        tab = self._nb.index(self._nb.select())
        if tab == 0:
            sta = self._obs_station.get().upper()
            return (f"_{sta}_{self._obs_year.get()}"
                    f"{self._obs_month.get()}{self._obs_day.get()}"
                    f"_{self._obs_hour.get()}Z")
        else:
            lat = self._model_lat.get().replace("-", "S").replace(".", "p")
            lon = self._model_lon.get().replace("-", "W").replace(".", "p")
            return (f"_{self._model_name.get()}"
                    f"_{self._model_year.get()}"
                    f"{self._model_month.get()}{self._model_day.get()}"
                    f"_{self._model_hour.get()}Z"
                    f"_{lat}_{lon}")

    # ── fetch ────────────────────────────────────────────────────────────────

    def _fetch(self):
        self._fetch_btn.config(state="disabled")
        self._save_btn.config(state="disabled")
        self._clean_data = self._cm1_lines = None
        self._u_rm = self._v_rm = None
        self._umove_var.set("—")
        self._vmove_var.set("—")
        self._set_status("Fetching…")
        threading.Thread(target=self._fetch_worker, daemon=True).start()

    def _fetch_worker(self):
        try:
            import sounderpy as spy
        except ImportError:
            self.after(0, lambda: self._fetch_error(
                "sounderpy is not installed.\nRun:  pip install sounderpy"))
            return

        tab = self._nb.index(self._nb.select())
        try:
            if tab == 0:
                clean_data = spy.get_obs_data(
                    self._obs_station.get().upper(),
                    self._obs_year.get(),
                    self._obs_month.get(),
                    self._obs_day.get(),
                    self._obs_hour.get(),
                    hush=True,
                )
            else:
                ds = self._model_dataset.get().strip() or None
                clean_data = spy.get_model_data(
                    self._model_name.get(),
                    [float(self._model_lat.get()), float(self._model_lon.get())],
                    self._model_year.get(),
                    self._model_month.get(),
                    self._model_day.get(),
                    self._model_hour.get(),
                    dataset=ds,
                    box_avg_size=float(self._model_box.get()),
                    hush=True,
                )
        except Exception as exc:
            self.after(0, lambda e=exc: self._fetch_error(str(e)))
            return

        try:
            lines = build_cm1_sounding(clean_data)
        except Exception as exc:
            self.after(0, lambda e=exc: self._fetch_error(f"Conversion error: {e}"))
            return

        u_rm, v_rm = compute_bunkers(clean_data)
        self.after(0, lambda: self._fetch_done(clean_data, lines, u_rm, v_rm))

    def _fetch_error(self, msg):
        self._fetch_btn.config(state="normal")
        self._set_status("Error — see dialog")
        messagebox.showerror("Fetch error", msg)

    def _fetch_done(self, clean_data, lines, u_rm, v_rm):
        self._fetch_btn.config(state="normal")
        self._clean_data = clean_data
        self._cm1_lines  = lines
        self._u_rm       = u_rm
        self._v_rm       = v_rm

        # Populate Bunkers display
        if u_rm is not None:
            self._umove_var.set(f"{u_rm:+.3f}")
            self._vmove_var.set(f"{v_rm:+.3f}")
        else:
            self._umove_var.set("n/a")
            self._vmove_var.set("n/a")

        self._file_suffix_val = self._file_suffix()
        self._set_preview("\n".join(lines))
        self._save_btn.config(state="normal")

        si   = clean_data.get("site_info", {}) or {}
        sid  = si.get("site-id",   "?")
        snam = si.get("site-name", "")
        self._set_status(
            f"Fetched {len(lines)-1} levels  |  {sid} {snam}  |  "
            f"umove = {u_rm:+.2f} m/s,  vmove = {v_rm:+.2f} m/s  —  ready to save."
            if u_rm is not None else
            f"Fetched {len(lines)-1} levels  |  {sid} {snam}  —  ready to save.")

    # ── save ─────────────────────────────────────────────────────────────────

    def _save(self):
        if not self._cm1_lines:
            messagebox.showwarning("Nothing to save", "Fetch a sounding first.")
            return

        base = self._outdir.get().strip()
        if not base:
            messagebox.showwarning("No directory", "Please set the base output directory.")
            return

        out_dir = os.path.join(base, "output")
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as exc:
            messagebox.showerror("Directory error", str(exc))
            return

        suffix = getattr(self, "_file_suffix_val", self._file_suffix())
        errors = []

        # 1. input_sounding_{suffix}
        snd_path = os.path.join(out_dir, f"input_sounding{suffix}")
        try:
            with open(snd_path, "w") as fh:
                fh.write("\n".join(self._cm1_lines) + "\n")
        except Exception as exc:
            errors.append(f"input_sounding: {exc}")

        # 2. sounding_info_{suffix}.txt
        info_path = os.path.join(out_dir, f"sounding_info{suffix}.txt")
        try:
            with open(info_path, "w") as fh:
                fh.write(build_info_text(self._clean_data, self._u_rm, self._v_rm))
        except Exception as exc:
            errors.append(f"sounding_info.txt: {exc}")

        if errors:
            messagebox.showerror("Save errors", "\n".join(errors))
        else:
            self._set_status(f"Saved  →  {out_dir}/")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = SoundingTool()
    app.mainloop()
