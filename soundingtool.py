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
    """Return (u_rm, v_rm, err_str) Bunkers right-mover in m/s using MetPy.

    If the sounding doesn't reach 6 km, extrapolates u/v/p using the wind
    shear from the top ~1 km of available data so Bunkers can still run.
    Returns (None, None, reason_str) on failure.
    """
    try:
        from metpy.calc import bunkers_storm_motion
    except ImportError:
        return None, None, "MetPy not installed"

    try:
        p = clean_data['p']
        u = clean_data['u'].to('m/s')
        v = clean_data['v'].to('m/s')
        z = clean_data['z']

        rm, _lm, _mean = bunkers_storm_motion(p, u, v, z)
        u_rm = float(rm[0].to('m/s').magnitude)
        v_rm = float(rm[1].to('m/s').magnitude)
        return u_rm, v_rm, None
    except Exception as exc:
        return None, None, str(exc)


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
# IEM RAOB fetch (full-resolution observed sounding)
# ---------------------------------------------------------------------------

def _fetch_iem_raob(station, year, month, day, hour):
    """Fetch full-resolution RAOB from IEM ASOS/RAOB JSON API.

    Returns a sounderpy-compatible clean_data dict with MetPy pint quantities.
    Heights come from IEM in feet; converted to metres here.
    Winds come as drct (deg FROM) + sknt (knots); converted to u/v m/s.
    Missing wind levels are linearly interpolated from neighbours.
    """
    import urllib.request
    import json

    try:
        from metpy.units import units
    except ImportError:
        raise ImportError("MetPy is required — install with:  pip install metpy")

    ts  = f"{year}-{int(month):02d}-{int(day):02d}T{int(hour):02d}:00:00Z"
    url = (f"https://mesonet.agron.iastate.edu/json/raob.py"
           f"?ts={ts}&station={station}")

    req = urllib.request.Request(url, headers={"User-Agent": "soundingtool/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())

    profiles = data.get("profiles", [])
    if not profiles:
        raise ValueError(f"IEM returned no sounding for {station} at {ts}")

    prof   = profiles[0]
    levels = prof.get("profile", [])
    if not levels:
        raise ValueError(f"IEM sounding profile is empty for {station} at {ts}")

    p_list, z_list, T_list, Td_list, u_list, v_list = [], [], [], [], [], []

    for lev in levels:
        pres = lev.get("pres")
        hght = lev.get("hght")
        tmpc = lev.get("tmpc")
        if pres is None or hght is None or tmpc is None:
            continue

        dwpc = lev.get("dwpc")
        if dwpc is None:
            dwpc = float(tmpc) - 40.0  # dry placeholder for missing dewpoint

        drct = lev.get("drct")
        sknt = lev.get("sknt")

        if drct is None or sknt is None:
            continue  # skip levels with missing wind
        spd_ms   = float(sknt) * 0.514444
        drct_rad = np.radians(float(drct))

        p_list.append(float(pres))
        z_list.append(float(hght) * 0.3048)   # feet → metres
        T_list.append(float(tmpc))
        Td_list.append(float(dwpc))
        u_list.append(-spd_ms * np.sin(drct_rad))
        v_list.append(-spd_ms * np.cos(drct_rad))

    if not p_list:
        raise ValueError("No valid levels found in IEM sounding")

    p_arr  = np.array(p_list,  dtype=float)
    z_arr  = np.array(z_list,  dtype=float)
    T_arr  = np.array(T_list,  dtype=float)
    Td_arr = np.array(Td_list, dtype=float)
    u_arr  = np.array(u_list,  dtype=float)
    v_arr  = np.array(v_list,  dtype=float)

    # Sort by ascending height
    order = np.argsort(z_arr)
    p_arr, z_arr, T_arr, Td_arr, u_arr, v_arr = (
        p_arr[order], z_arr[order], T_arr[order],
        Td_arr[order], u_arr[order], v_arr[order],
    )

    station_id = prof.get("station", station).upper()
    lat = prof.get("lat") or prof.get("latitude")
    lon = prof.get("lon") or prof.get("longitude")

    if lat is None:
        # IEM RAOB response doesn't include coordinates; fetch from network metadata
        try:
            meta_url = "https://mesonet.agron.iastate.edu/geojson/network.py?network=RAOB"
            req2 = urllib.request.Request(meta_url, headers={"User-Agent": "soundingtool/1.0"})
            with urllib.request.urlopen(req2, timeout=15) as r:
                meta = json.loads(r.read())
            for feat in meta.get("features", []):
                if feat.get("properties", {}).get("sid", "").upper() == station_id:
                    coords = feat["geometry"]["coordinates"]  # [lon, lat]
                    lon, lat = float(coords[0]), float(coords[1])
                    break
        except Exception:
            pass

    latlon_str = f"{lat}, {lon}" if lat is not None else ""

    clean_data = {
        "p":  p_arr  * units.hPa,
        "z":  z_arr  * units.m,
        "T":  T_arr  * units.degC,
        "Td": Td_arr * units.degC,
        "u":  u_arr  * units("m/s"),
        "v":  v_arr  * units("m/s"),
        "site_info": {
            "site-id":     station_id,
            "site-name":   station_id,
            "site-lctn":   "",
            "site-latlon": latlon_str,
            "site-lat":    lat,
            "site-lon":    lon,
            "source":      "IEM RAOB (full resolution)",
            "valid-time":  prof.get("valid", ts),
        },
    }
    return clean_data


def _fetch_bufkit_rap(station, year, month, day, hour):
    """Fetch RAP f00 BUFKIT from Iowa State mtarchive and return a clean_data dict.

    BUFKIT format: column order defined once in the SNPARM header line using
    semicolons; data rows are plain space-separated floats with no per-block
    header.  Heights (HGHT) are metres MSL.  Terminator row is all 99999.
    """
    import urllib.request
    from metpy.units import units as munits

    station_lower = station.lower()
    url = (f"https://mtarchive.geol.iastate.edu"
           f"/{int(year)}/{int(month):02d}/{int(day):02d}"
           f"/bufkit/{int(hour):02d}/rap/rap_{station_lower}.buf")

    req = urllib.request.Request(url, headers={"User-Agent": "soundingtool/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        lines = resp.read().decode('ascii', errors='replace').splitlines()

    # Parse column order from SNPARM line at top of file
    # e.g.  SNPARM = PRES;TMPC;TMWC;DWPC;THTE;DRCT;SKNT;OMEG;CFRL;HGHT
    snparm_line = next((l for l in lines[:30] if l.strip().startswith('SNPARM')), None)
    if snparm_line is None:
        raise ValueError("No SNPARM line in BUFKIT file")
    cols = [c.strip() for c in snparm_line.split('=', 1)[1].split(';')]
    try:
        ip  = cols.index('PRES'); iz  = cols.index('HGHT')
        it  = cols.index('TMPC'); id_ = cols.index('DWPC')
        idr = cols.index('DRCT'); isk = cols.index('SKNT')
    except ValueError as e:
        raise ValueError(f"BUFKIT SNPARM missing column: {e}")

    # Locate the first STIM = 0 block
    stim0 = next((i for i, l in enumerate(lines) if 'STIM = 0' in l), None)
    if stim0 is None:
        raise ValueError("No STIM=0 block in BUFKIT file")

    # Column header after STIM=0 starts with 'PRES' (may span 2 lines)
    hdr = next((i for i in range(stim0 + 1, min(stim0 + 20, len(lines)))
                if lines[i].strip().startswith('PRES')), None)
    if hdr is None:
        raise ValueError("No column header found after STIM=0")

    # Data begins immediately after the (2-line) column header
    data_start = hdr + 2

    # Each level spans 2 lines (e.g. 8 values then 2 values); join pairs
    p_l, z_l, T_l, Td_l, u_l, v_l = [], [], [], [], [], []

    i = data_start
    while i + 1 < len(lines):
        parts = (lines[i] + ' ' + lines[i + 1]).split()
        i += 2
        if not parts:
            continue
        try:
            if float(parts[0]) > 99990:
                break
        except ValueError:
            break

        try:
            pres = float(parts[ip]); hght = float(parts[iz])
            tmpc = float(parts[it]); dwpc = float(parts[id_])
            drct = float(parts[idr]); sknt = float(parts[isk])
        except (ValueError, IndexError):
            continue

        if pres <= 0 or tmpc < -9990 or hght < -9990:
            continue
        if dwpc < -9990:
            dwpc = tmpc - 40.0
        if drct < -9990 or sknt < -9990:
            continue

        spd_ms   = sknt * 0.514444
        drct_rad = np.radians(drct)
        p_l.append(pres); z_l.append(hght)
        T_l.append(tmpc); Td_l.append(dwpc)
        u_l.append(-spd_ms * np.sin(drct_rad))
        v_l.append(-spd_ms * np.cos(drct_rad))

    if not p_l:
        raise ValueError("No valid levels parsed from BUFKIT data")

    order = np.argsort(np.array(z_l, dtype=float))
    def _s(a): return np.array(a, dtype=float)[order]

    return {
        'p':  _s(p_l)  * munits.hPa,
        'z':  _s(z_l)  * munits.m,
        'T':  _s(T_l)  * munits.degC,
        'Td': _s(Td_l) * munits.degC,
        'u':  _s(u_l)  * munits('m/s'),
        'v':  _s(v_l)  * munits('m/s'),
    }


def _extend_with_model(clean_data, year, month, day, hour):
    """Append RAP/RUC levels above the balloon ceiling, if needed for Bunkers.

    Tries BUFKIT RAP f00 first (recent dates); falls back to RAP/RUC reanalysis.
    Returns (extended_clean_data, n_model_levels_added, warning_str_or_None).
    """
    import datetime as dt

    z_m = _mag(clean_data['z'], 'm')

    si         = clean_data.get('site_info', {})
    station_id = si.get('site-id', '')
    lat        = si.get('site-lat')
    lon        = si.get('site-lon')

    try:
        import sounderpy as spy
    except ImportError:
        return clean_data, 0, "sounderpy not installed — cannot fill upper levels with RAP/RUC"

    from metpy.units import units as munits

    model_data = None

    # Try BUFKIT RAP f00 first (works for recent dates, ~last 30 days)
    # BUFKIT uses 3-letter IDs; IEM returns 4-letter ICAO (e.g. KOUN → OUN)
    bufkit_id = (station_id[1:] if len(station_id) == 4 and station_id.startswith('K')
                 else station_id)
    try:
        model_data = _fetch_bufkit_rap(bufkit_id, year, month, day, hour)
    except Exception as _bufkit_err:
        import sys
        print(f"[soundingtool] BUFKIT fetch failed ({_bufkit_err}), trying reanalysis", file=sys.stderr)
        model_data = None

    # Fall back to archived RAP/RUC reanalysis (works for dates > ~1 month old)
    if model_data is None and lat is not None and lon is not None:
        model_data = spy.get_model_data(
            'rap-ruc', [float(lat), float(lon)],
            str(year), str(month), str(day), str(hour),
            box_avg_size=0.1, hush=True,
        )

    if model_data is None:
        return clean_data, 0, "Could not fetch RAP/RUC data (BUFKIT and reanalysis both failed)"

    z_ceil = z_m[-1]
    mz = _mag(model_data['z'], 'm')
    above = mz > z_ceil
    if not above.any():
        return clean_data, 0, "RAP/RUC profile has no levels above balloon ceiling"

    n_model = int(above.sum())
    p_arr  = np.concatenate([_mag(clean_data['p'],  'hPa'),  _mag(model_data['p'],  'hPa')[above]])
    z_arr  = np.concatenate([_mag(clean_data['z'],  'm'),    mz[above]])
    T_arr  = np.concatenate([_mag(clean_data['T'],  'degC'), _mag(model_data['T'],  'degC')[above]])
    Td_arr = np.concatenate([_mag(clean_data['Td'], 'degC'), _mag(model_data['Td'], 'degC')[above]])
    u_arr  = np.concatenate([_mag(clean_data['u'],  'm/s'),  _mag(model_data['u'],  'm/s')[above]])
    v_arr  = np.concatenate([_mag(clean_data['v'],  'm/s'),  _mag(model_data['v'],  'm/s')[above]])

    order = np.argsort(z_arr)
    p_arr, z_arr, T_arr, Td_arr, u_arr, v_arr = (
        p_arr[order], z_arr[order], T_arr[order],
        Td_arr[order], u_arr[order], v_arr[order],
    )

    extended = {
        'p':  p_arr  * munits.hPa,
        'z':  z_arr  * munits.m,
        'T':  T_arr  * munits.degC,
        'Td': Td_arr * munits.degC,
        'u':  u_arr  * munits('m/s'),
        'v':  v_arr  * munits('m/s'),
        'site_info': si,
    }
    return extended, n_model, None


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
        tab = self._nb.index(self._nb.select())
        try:
            if tab == 0:
                clean_data = _fetch_iem_raob(
                    self._obs_station.get().upper(),
                    self._obs_year.get(),
                    self._obs_month.get(),
                    self._obs_day.get(),
                    self._obs_hour.get(),
                )
                clean_data, _n_model, _fill_warn = _extend_with_model(
                    clean_data,
                    self._obs_year.get(),
                    self._obs_month.get(),
                    self._obs_day.get(),
                    self._obs_hour.get(),
                )
                if _fill_warn:
                    import sys
                    print(f"[soundingtool] RAP/RUC fill: {_fill_warn}", file=sys.stderr)
            else:
                try:
                    import sounderpy as spy
                except ImportError:
                    self.after(0, lambda: self._fetch_error(
                        "sounderpy is not installed.\nRun:  pip install sounderpy"))
                    return
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

        u_rm, v_rm, bunkers_err = compute_bunkers(clean_data)
        self.after(0, lambda: self._fetch_done(clean_data, lines, u_rm, v_rm, bunkers_err))

    def _fetch_error(self, msg):
        self._fetch_btn.config(state="normal")
        self._set_status("Error — see dialog")
        messagebox.showerror("Fetch error", msg)

    def _fetch_done(self, clean_data, lines, u_rm, v_rm, bunkers_err=None):
        self._fetch_btn.config(state="normal")
        self._clean_data = clean_data
        self._cm1_lines  = lines
        self._u_rm       = u_rm
        self._v_rm       = v_rm

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
        nlev = len(lines) - 1

        if u_rm is not None:
            self._set_status(
                f"Fetched {nlev} levels  |  {sid} {snam}  |  "
                f"umove = {u_rm:+.2f} m/s,  vmove = {v_rm:+.2f} m/s  —  ready to save.")
        else:
            reason = f": {bunkers_err}" if bunkers_err else ""
            self._set_status(
                f"Fetched {nlev} levels  |  {sid} {snam}  |  "
                f"Bunkers failed{reason}")

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
