#!/opt/miniconda3/envs/met/bin/python3
"""namelisttool.py — Comprehensive CM1 namelist.input editor."""

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import numpy as np
import re, os

try:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ═══════════════════════════════════════════════════════════════════════════════
# Grid math  (ported from CM1 param.F)
# ═══════════════════════════════════════════════════════════════════════════════

def zf_uniform(nz, dz):
    return np.arange(nz + 1, dtype=float) * dz

def zf_stretch1(nz, ztop, str_bot, str_top, dz_bot, dz_top):
    nk1 = round(str_bot / dz_bot) if dz_bot > 0 else 0
    nk3 = round((ztop - str_top) / dz_top) if dz_top > 0 else 0
    nk2 = nz - (nk1 + nk3)
    errs = []
    if nk1 <= 0 and str_bot > 0:
        errs.append(f"nk1={nk1}: str_bot/dz_bot must be ≥ 1")
    elif dz_bot > 0 and abs(nk1 * dz_bot - str_bot) > 0.5:
        errs.append("str_bot not exactly divisible by dz_bot")
    if nk3 <= 0:
        errs.append(f"nk3={nk3} ≤ 0: increase ztop or reduce str_top")
    elif dz_top > 0 and abs(nk3 * dz_top - (ztop - str_top)) > 0.5:
        errs.append("(ztop−str_top) not exactly divisible by dz_top")
    if nk2 <= 1:
        errs.append(f"nk2={nk2} ≤ 1: increase nz or adjust str params")
    if errs:
        return None, errs
    nom = (str_top - str_bot) / nk2
    c2 = (nom - dz_bot) / (nom * nom * (nk2 - 1))
    c1 = (dz_bot / nom) - c2 * nom
    zf = np.zeros(nz + 1)
    for k in range(nk1 + 1):
        zf[k] = k * dz_bot
    for k in range(nk1, nk1 + nk2 + 1):
        kp = k - nk1
        zf[k] = zf[nk1] + (c1 + c2 * kp * nom) * kp * nom
    for k in range(nk1 + nk2 + 1, nz + 1):
        zf[k] = zf[k - 1] + dz_top
    return zf, []

def _zheight(dz_bot, r, n, dz_max, n1):
    total, n2, dzm = 0.0, 0, dz_bot
    for k in range(1, n + 1):
        if k <= n1:
            dznew = dz_bot
        else:
            dznew = min(dz_bot * r ** (k - n1 - 1), float(dz_max))
        total += dznew
    return total

def zf_stretch2(nz, dz, dz_bot, str_bot, dz_top):
    if dz_bot <= 0: return None, ["dz_bot must be > 0"], 1.0
    if dz_top <= 0: return None, ["dz_top must be > 0"], 1.0
    if dz_bot > dz:  return None, [f"dz_bot ({dz_bot:.0f}) > dz ({dz:.0f})"], 1.0
    nbndlyr = max(0, int(str_bot / dz_bot + 0.01) - 1)
    target = dz * nz
    zx, r = 1.0, 1.0
    if dz > dz_bot:
        for _ in range(50):
            if abs(zx) > 1e-12:
                zx *= 0.5
                xmid = r + zx
                fmid = _zheight(dz_bot, xmid, nz, dz_top, nbndlyr) - target
                if fmid <= 0.0:
                    r = xmid
                if abs(fmid) < 1e-3:
                    break
    if r > 1.1:
        return None, [f"Stretch factor r={r:.5f} > 1.1 — increase nz or dz_bot"], r
    zf = np.zeros(nz + 1)
    for k in range(1, nz + 1):
        zf[k] = _zheight(dz_bot, r, k, dz_top, nbndlyr)
    return zf, [], r

def suggest_nz_s1(str_bot, str_top, dz_bot, dz_top, ztop):
    nom = 0.5 * (dz_bot + dz_top)
    nk1 = round(str_bot / dz_bot) if dz_bot > 0 else 0
    nk2 = round((str_top - str_bot) / nom) if nom > 0 else 0
    nk3 = round((ztop - str_top) / dz_top) if dz_top > 0 else 0
    return nk1, nk2, nk3

def _horiz_xf(sx, n, d_inner, d_outer, nos_len, tot_len):
    if sx == 0:
        return np.arange(n + 1, dtype=float) * d_inner, 0, n, 0, []
    nominal = 0.5 * (d_inner + d_outer)
    if nominal <= 0:
        return None, 0, 0, 0, ["d_inner/d_outer must be > 0"]
    if sx == 1:
        ni1 = round((tot_len - nos_len) * 0.5 / nominal)
        ni2 = round(nos_len / d_inner) if d_inner > 0 else 0
        ni3 = ni1
    else:
        ni1 = 0
        ni2 = round(nos_len / d_inner) if d_inner > 0 else 0
        ni3 = round((tot_len - nos_len) / nominal)
    n_total = ni1 + ni2 + ni3
    if n_total <= 0:
        return None, ni1, ni2, ni3, ["Grid count 0; check lengths"]
    c2 = (nominal - d_inner) / (nominal ** 2 * (ni3 - 1)) if ni3 > 1 else 0.0
    c1 = (d_inner / nominal) - c2 * nominal
    mult = 0.5
    xf = np.zeros(n_total + 1)
    for i in range(ni1, ni1 + ni2 + 1):
        xf[i] = ni1 * nominal + (i - ni1) * d_inner - mult * tot_len
    for i in range(ni1 + ni2 + 1, n_total + 1):
        kp = i - ni1 - ni2
        xf[i] = (ni1 * nominal + ni2 * d_inner
                 + (c1 + c2 * kp * nominal) * kp * nominal - mult * tot_len)
    for i in range(0, ni1):
        kp = ni1 - i
        xf[i] = (ni1 * nominal
                 - (c1 + c2 * kp * nominal) * kp * nominal - mult * tot_len)
    return xf, ni1, ni2, ni3, []

# ═══════════════════════════════════════════════════════════════════════════════
# Namelist parser
# ═══════════════════════════════════════════════════════════════════════════════

def _nl_val(s):
    s = s.strip()
    if s.lower() in ('.true.', '.t.'): return True
    if s.lower() in ('.false.', '.f.'): return False
    try: return int(s)
    except ValueError: pass
    try: return float(s.replace('d','e').replace('D','e'))
    except ValueError: return s

def _nl_parse_kv(line, dest):
    for m in re.finditer(r'(\w+)\s*=\s*([^,=/]+)', line):
        dest[m.group(1).lower()] = _nl_val(m.group(2))

def parse_namelist(text):
    result, cur = {}, None
    for raw in text.split('\n'):
        line = raw.split('!')[0].strip()
        if not line: continue
        m = re.match(r'&(\w+)', line)
        if m:
            cur = m.group(1).lower()
            result[cur] = {}
            line = line[m.end():].strip()
        if cur is None: continue
        if '/' in line:
            _nl_parse_kv(line[:line.index('/')], result[cur])
            cur = None
        else:
            _nl_parse_kv(line, result[cur])
    return result

# ═══════════════════════════════════════════════════════════════════════════════
# Parameter schema
# ═══════════════════════════════════════════════════════════════════════════════
# Tuple: (section, key, default, widget, label, hint[, choices])
# widget: 'e'=entry  'c'=combo  'cb'=check(.true./.false.)  'ci'=check(0/1 int)
# choices for 'c': [(int_val, description), ...]

def _e(sec, key, dflt, lbl, hint):
    return (sec, key, dflt, 'e', lbl, hint, [])
def _c(sec, key, dflt, lbl, hint, choices):
    return (sec, key, dflt, 'c', lbl, hint, choices)
def _cb(sec, key, dflt, lbl, hint):
    return (sec, key, dflt, 'cb', lbl, hint, [])
def _ci(sec, key, dflt, lbl, hint):
    return (sec, key, dflt, 'ci', lbl, hint, [])

_YN   = [(0,'no'),(1,'yes')]
_BC   = [(1,'periodic'),(2,'open-radiative'),(3,'rigid free-slip'),(4,'rigid no-slip')]
_SLIP = [(1,'free slip'),(2,'no slip'),(3,'semi-slip')]

ALL_PARAMS = [
    # ── param0 ──────────────────────────────────────────────────────────────
    _e ('param0','nx',          200,     'nx',          'Total grid points in X'),
    _e ('param0','ny',          200,     'ny',          'Total grid points in Y'),
    _e ('param0','nz',           40,     'nz',          'Total grid points in Z'),
    _e ('param0','ppnode',      128,     'ppnode',       'MPI ranks per node (I/O only)'),
    _c ('param0','timeformat',    2,     'timeformat',   'Time format for text output',
        [(1,'seconds'),(2,'minutes'),(3,'hours'),(4,'days')]),
    _c ('param0','timestats',     1,     'timestats',    'Timing statistics',
        [(0,'off'),(1,'summary at end'),(2,'per timestep')]),
    _cb('param0','terrain_flag', False,  'terrain_flag', 'Enable terrain'),
    _cb('param0','procfiles',   False,   'procfiles',    'Write config file for every MPI process'),
    _c ('param0','outunits',      1,     'outunits',     'Units for x,y,z in output files',
        [(1,'km'),(2,'meters')]),

    # ── param1 ──────────────────────────────────────────────────────────────
    _e ('param1','dx',         2000.0,  'dx (m)',       'Grid spacing in X'),
    _e ('param1','dy',         2000.0,  'dy (m)',       'Grid spacing in Y'),
    _e ('param1','dz',          500.0,  'dz (m)',       'Vertical grid spacing (or nominal if stretched)'),
    _e ('param1','dtl',           7.5,  'dtl (s)',      'Large time step; ~min(dx,dy,dz)/67 for psolver=2,3'),
    _e ('param1','timax',      7200.0,  'timax (s)',    'Maximum integration time'),
    _e ('param1','run_time',   -999.9,  'run_time (s)', 'Integration time from current time (overrides timax if ≥ 0)'),
    _e ('param1','tapfrq',      900.0,  'tapfrq (s)',   'Output frequency (3D fields)'),
    _e ('param1','rstfrq',    -3600.0,  'rstfrq (s)',   'Restart file frequency (negative = no restart)'),
    _e ('param1','statfrq',      60.0,  'statfrq (s)',  'Statistics output frequency (negative = every timestep)'),
    _e ('param1','prclfrq',      60.0,  'prclfrq (s)', 'Parcel output frequency (negative = every timestep)'),

    # ── param2 — run control ─────────────────────────────────────────────────
    _c ('param2','cm1setup',      1,    'cm1setup',     'Overall model setup / turbulence framework',
        [(0,'Euler (no SGS)'),(1,'LES'),(2,'mesoscale+PBL (RANS)'),(3,'DNS'),(4,'LES-within-mesoscale')]),
    _e ('param2','testcase',      0,    'testcase',     'Idealized test case (0=default; see docs for 1-15)'),
    _c ('param2','adapt_dt',      0,    'adapt_dt',     'Adaptive time step', _YN),
    _c ('param2','irst',          0,    'irst',         'Restart run', _YN),
    _e ('param2','rstnum',        1,    'rstnum',       'Restart file number (for irst=1)'),
    _c ('param2','iconly',        0,    'iconly',       'Initialize only, do not integrate', _YN),

    # ── param2 — advection ───────────────────────────────────────────────────
    _e ('param2','hadvordrs',     5,    'hadvordrs',    'Horizontal advection order for scalars (2-10)'),
    _e ('param2','vadvordrs',     5,    'vadvordrs',    'Vertical advection order for scalars (2-10)'),
    _e ('param2','hadvordrv',     5,    'hadvordrv',    'Horizontal advection order for velocities (2-10)'),
    _e ('param2','vadvordrv',     5,    'vadvordrv',    'Vertical advection order for velocities (2-10)'),
    _c ('param2','advwenos',      2,    'advwenos',     'WENO scheme for scalars',
        [(0,'off'),(1,'every RK step'),(2,'final RK step')]),
    _c ('param2','advwenov',      0,    'advwenov',     'WENO scheme for velocities',
        [(0,'off'),(1,'every RK step'),(2,'final RK step')]),
    _c ('param2','weno_order',    5,    'weno_order',   'WENO scheme order',
        [(3,'3rd'),(5,'5th (Borges)'),(7,'7th'),(9,'9th')]),

    # ── param2 — diffusion ───────────────────────────────────────────────────
    _c ('param2','idiff',         0,    'idiff',        'Artificial diffusion',
        [(0,'off'),(1,'all variables'),(2,'winds only')]),
    _c ('param2','mdiff',         0,    'mdiff',        'Monotonic diffusion (requires idiff=1, difforder=6)', _YN),
    _c ('param2','difforder',     6,    'difforder',    'Diffusion order',
        [(2,'2nd order'),(6,'6th order')]),

    # ── param2 — pressure / mass ─────────────────────────────────────────────
    _c ('param2','psolver',       3,    'psolver',      'Pressure solver',
        [(1,'compressible explicit'),(2,'compressible KW explicit'),(3,'compressible KW vert-implicit'),
         (4,'anelastic'),(5,'incompressible'),(6,'compressible-Boussinesq'),(7,'modified compressible')]),
    _c ('param2','apmasscon',     1,    'apmasscon',    'Adjust avg pressure to conserve dry-air mass', _YN),

    # ── param2 — moisture / thermodynamics ───────────────────────────────────
    _c ('param2','imoist',        1,    'imoist',       'Include moisture', _YN),
    _c ('param2','eqtset',        2,    'eqtset',       'Moist equation set',
        [(1,'traditional approx'),(2,'energy-conserving (Bryan-Fritsch)')]),
    _c ('param2','idiss',         1,    'idiss',        'Dissipative heating', _YN),
    _c ('param2','efall',         0,    'efall',        'Energy fallout term', _YN),
    _c ('param2','rterm',         0,    'rterm',        'Simple radiation relaxation term', _YN),

    # ── param2 — coriolis ────────────────────────────────────────────────────
    _c ('param2','icor',          0,    'icor',         'Coriolis (f-plane; set fcor in param3)', _YN),
    _c ('param2','betaplane',     0,    'betaplane',    'Beta plane (Coriolis varies with y)', _YN),
    _c ('param2','lspgrad',       0,    'lspgrad',      'Large-scale pressure gradient acceleration',
        [(0,'none'),(1,'geostrophic from base-state winds'),(2,'geostrophic from ug,vg arrays'),
         (3,'gradient-wind balance'),(4,'specified ulspg/vlspg')]),

    # ── param2 — microphysics ────────────────────────────────────────────────
    _c ('param2','ptype',         5,    'ptype',        'Microphysics scheme',
        [(0,'none (vapor only)'),(1,'Kessler'),(2,'NASA-Goddard LFO'),(3,'Thompson'),
         (4,'GRS LFO'),(5,'Morrison'),(6,'Rotunno-Emanuel water-only'),(7,'WSM6'),
         (26,'NSSL 2-mom (graupel)'),(27,'NSSL 2-mom (graupel+hail)'),
         (28,'NSSL 1-mom'),(50,'P3 1-ice 1-mom cloud'),(51,'P3 1-ice 2-mom cloud'),
         (52,'P3 2-ice 2-mom cloud'),(53,'P3 1-ice 3-mom ice'),(55,'Jensen ISHMAEL')]),
    _cb('param2','nssl_3moment', False, 'nssl_3moment', 'NSSL: enable 3-moment for rain/graupel/hail'),
    _cb('param2','nssl_density_on',True,'nssl_density_on','NSSL: predict graupel/hail density'),
    _c ('param2','ihail',         1,    'ihail',        'Large ice category (Goddard-LFO & Morrison)',
        [(0,'graupel'),(1,'hail')]),
    _c ('param2','iautoc',        1,    'iautoc',       'Autoconversion qc→qr (Goddard-LFO only)', _YN),
    _c ('param2','cuparam',       0,    'cuparam',      'Convection parameterization',
        [(0,'none'),(1,'new Tiedtke')]),

    # ── param2 — turbulence / PBL ────────────────────────────────────────────
    _c ('param2','sgsmodel',      1,    'sgsmodel',     'SGS turbulence model (cm1setup=1)',
        [(1,'TKE / Deardorff'),(2,'Smagorinsky'),(3,'TKE + Sullivan two-part'),
         (4,'TKE + Bryan two-part'),(5,'NBA-TKE'),(6,'NBA-Smagorinsky')]),
    _c ('param2','ipbl',          0,    'ipbl',         'PBL parameterization (cm1setup=2)',
        [(0,'none'),(1,'YSU'),(2,'simple Louis-type'),(3,'GFS-EDMF'),
         (4,'MYNN 2.5'),(5,'MYNN 3'),(6,'MYJ')]),
    _c ('param2','tconfig',       1,    'tconfig',      'Turbulence coefficient config',
        [(1,'isotropic (dx≈dz)'),(2,'anisotropic (dx>>dz)')]),
    _c ('param2','bcturbs',       1,    'bcturbs',      'Lower/upper BC for scalar turbulent diffusion',
        [(1,'zero flux'),(2,'zero gradient')]),
    _c ('param2','horizturb',     0,    'horizturb',    'Horizontal Smagorinsky (cm1setup=2)', _YN),
    _c ('param2','doimpl',        1,    'doimpl',       'Implicit vertical turbulence',
        [(0,'explicit'),(1,'implicit Crank-Nicholson')]),

    # ── param2 — damping ─────────────────────────────────────────────────────
    _c ('param2','irdamp',        1,    'irdamp',       'Upper Rayleigh damping (set rdalpha,zd in param3)',
        [(0,'off'),(1,'damp toward base state'),(2,'damp toward horiz avg')]),
    _c ('param2','hrdamp',        0,    'hrdamp',       'Lateral Rayleigh damping (set xhd in param3)', _YN),

    # ── param2 — boundary conditions ─────────────────────────────────────────
    _c ('param2','wbc',           2,    'wbc',          'West lateral BC',  _BC),
    _c ('param2','ebc',           2,    'ebc',          'East lateral BC',  _BC),
    _c ('param2','sbc',           2,    'sbc',          'South lateral BC', _BC),
    _c ('param2','nbc',           2,    'nbc',          'North lateral BC', _BC),
    _c ('param2','bbc',           1,    'bbc',          'Bottom BC (winds)', _SLIP),
    _c ('param2','tbc',           1,    'tbc',          'Top BC (winds)',    _SLIP),
    _c ('param2','irbc',          4,    'irbc',         'Radiative BC scheme (for bc=2)',
        [(1,'Klemp-Wilhelmson large steps'),(2,'Klemp-Wilhelmson small steps'),(4,'Durran-Klemp 1983')]),
    _c ('param2','roflux',        0,    'roflux',       'Restrict outward mass flux at open BCs', _YN),
    _c ('param2','nudgeobc',      0,    'nudgeobc',     'Nudge winds toward base state at inflow BCs', _YN),

    # ── param2 — initialization ──────────────────────────────────────────────
    _c ('param2','isnd',          7,    'isnd',         'Base-state sounding',
        [(1,'dry adiabatic'),(2,'dry isothermal'),(3,'dry const dT/dz'),
         (4,'saturated neutral BF02'),(5,'Weisman-Klemp analytic'),(7,'external input_sounding file'),
         (8,'dry const dθ/dz'),(9,'dry const BV freq'),(10,'saturated const BV freq'),
         (11,'saturated const θe'),(12,'dry adiabatic + const lapse rate'),
         (13,'dry 3-layer const N²'),(14,'dry CBL profile'),(15,'moist DYCOMS-II strCu'),
         (17,'external file (winds ignored)'),(18,'dry sharp inversion SBL'),
         (19,'moist BOMEX shallow Cu'),(20,'moist RICO shallow Cu'),
         (22,'stable BL test case'),(23,'shallow Cu over land')]),
    _c ('param2','iwnd',          2,    'iwnd',         'Base-state wind profile (ignored for isnd=7)',
        [(0,'zero winds'),(1,'RKW-type'),(2,'Weisman-Klemp supercell'),(3,'multicell'),
         (4,'W-K multicell'),(5,'Dornbrack analytic'),(6,'constant wind'),
         (8,'constant/linear hurricane BL'),(9,'linear shallow Cu'),
         (10,'linear drizzling strCu'),(11,'RICO shallow Cu')]),
    _c ('param2','itern',         0,    'itern',        'Initial topography',
        [(0,'none'),(1,'bell-shaped hill'),(2,'Schaer test'),(3,'Lane-Doyle'),(4,'from GrADS file')]),
    _c ('param2','iinit',         1,    'iinit',        '3D perturbation initialization',
        [(0,'none'),(1,'warm bubble'),(2,'cold pool'),(3,'line of warm bubbles'),
         (4,'moist benchmark'),(5,'cold blob'),(7,'tropical cyclone'),
         (8,'line thermal + random'),(9,'forced convergence'),(10,'momentum forcing'),
         (11,'Skamarock-Klemp IG wave'),(12,'updraft nudging')]),
    _c ('param2','irandp',        0,    'irandp',       'Random θ perturbations in initial conditions', _YN),
    _c ('param2','ibalance',      0,    'ibalance',     'Initial 3D pressure balance',
        [(0,'none (p\'=0)'),(1,'hydrostatic'),(2,'anelastic')]),
    _c ('param2','iorigin',       2,    'iorigin',      'Origin location in horizontal space',
        [(1,'bottom-left corner'),(2,'domain center')]),
    _c ('param2','axisymm',       0,    'axisymm',      'Axisymmetric mode (ny=1 required)', _YN),
    _c ('param2','imove',         1,    'imove',        'Moving domain (set umove/vmove in param3)', _YN),

    # ── param2 — tracers & parcels ───────────────────────────────────────────
    _c ('param2','iptra',         0,    'iptra',        'Passive fluid tracers', _YN),
    _e ('param2','npt',           1,    'npt',          'Number of passive tracers'),
    _c ('param2','pdtra',         1,    'pdtra',        'Positive-definite tracers', _YN),
    _c ('param2','iprcl',         0,    'iprcl',        'Passive parcels', _YN),
    _e ('param2','nparcels',      1,    'nparcels',     'Number of parcels'),

    # ── param3 — diffusion coefficients ─────────────────────────────────────
    _e ('param3','kdiff2',       75.0,  'kdiff2 (m²/s)','Diffusion coeff for difforder=2'),
    _e ('param3','kdiff6',        0.04, 'kdiff6',       'Diffusion coeff for difforder=6 (fraction of 1D stability; 0.02-0.24)'),
    _e ('param3','kdiv',          0.10, 'kdiv',         'Divergence damper coefficient (~0.1; for psolver=2,3)'),
    _e ('param3','alph',          0.60, 'alph',         'Off-centering coeff for vert-implicit acoustic solver (psolver=3)'),
    _e ('param3','cstar',        30.0,  'cstar (m/s)',  'Outward wave speed at open BCs (for irbc=1,2)'),
    _e ('param3','csound',      300.0,  'csound (m/s)', 'Speed of sound for psolver=6,7 (~5-10× max flow speed)'),

    # ── param3 — Rayleigh damping ─────────────────────────────────────────────
    _e ('param3','rdalpha',  3.3333e-3, 'rdalpha (1/s)','Rayleigh damping e-folding rate (~1/300)'),
    _e ('param3','zd',       15000.0,   'zd (m)',       'Height above which upper Rayleigh damping is applied'),
    _e ('param3','xhd',     100000.0,   'xhd (m)',      'Distance from lateral BCs where lateral Rayleigh damping is applied'),
    _e ('param3','alphobc',     60.0,   'alphobc (s)',  'Nudging timescale at inflow BCs (for nudgeobc=1)'),

    # ── param3 — storm motion ────────────────────────────────────────────────
    _e ('param3','umove',         7.318, 'umove (m/s)', 'Domain translation speed in X (for imove=1)'),
    _e ('param3','vmove',         0.458, 'vmove (m/s)', 'Domain translation speed in Y (for imove=1)'),
    _e ('param3','fcor',       0.00005,  'fcor (1/s)',  'Coriolis parameter (for icor=1)'),

    # ── param3 — turbulence scales ───────────────────────────────────────────
    _e ('param3','l_h',         100.0,  'l_h (m)',      'Horizontal turbulence length scale (horizturb=1, over land)'),
    _e ('param3','lhref1',      100.0,  'lhref1 (m)',   'l_h at psurf=1015 mb (ocean, horizturb=1)'),
    _e ('param3','lhref2',     1000.0,  'lhref2 (m)',   'l_h at psurf=900 mb (ocean, horizturb=1)'),
    _e ('param3','l_inf',        75.0,  'l_inf (m)',    'Asymptotic vertical turbulence length scale (ipbl=2)'),

    # ── param3 — microphysics coefficients ───────────────────────────────────
    _e ('param3','ndcnst',      250.0,  'ndcnst (cm⁻³)','Cloud droplet concentration (Morrison microphysics)'),
    _e ('param3','nt_c',        250.0,  'nt_c (cm⁻³)',  'Cloud droplet concentration (Thompson microphysics)'),
    _e ('param3','v_t',           7.0,  'v_t (m/s)',    'Terminal fall velocity (ptype=6; negative = pseudoadiabatic)'),

    # ── param4 — X stretching ────────────────────────────────────────────────
    _c ('param4','stretch_x',     0,    'stretch_x',    'Horizontal stretching in X',
        [(0,'none'),(1,'both sides'),(2,'east side only'),(3,'from input_grid_x file')]),
    _e ('param4','dx_inner',   1000.0,  'dx_inner (m)', 'Smallest X grid spacing'),
    _e ('param4','dx_outer',   7000.0,  'dx_outer (m)', 'Largest X grid spacing (at edges)'),
    _e ('param4','nos_x_len',  40000.0, 'nos_x_len (m)','Length of unstretched inner X region'),
    _e ('param4','tot_x_len', 120000.0, 'tot_x_len (m)','Total X domain length'),

    # ── param5 — Y stretching ────────────────────────────────────────────────
    _c ('param5','stretch_y',     0,    'stretch_y',    'Horizontal stretching in Y',
        [(0,'none'),(1,'both sides'),(2,'north side only'),(3,'from input_grid_y file')]),
    _e ('param5','dy_inner',   1000.0,  'dy_inner (m)', 'Smallest Y grid spacing'),
    _e ('param5','dy_outer',   7000.0,  'dy_outer (m)', 'Largest Y grid spacing (at edges)'),
    _e ('param5','nos_y_len',  40000.0, 'nos_y_len (m)','Length of unstretched inner Y region'),
    _e ('param5','tot_y_len', 120000.0, 'tot_y_len (m)','Total Y domain length'),

    # ── param6 — Z stretching ────────────────────────────────────────────────
    _c ('param6','stretch_z',     0,    'stretch_z',    'Vertical stretching',
        [(0,'none'),(1,'Wilhelmson-Chen 3-layer'),(2,'geometric Wicker'),
         (3,'from input_grid_z (scalar levels)'),(4,'from input_grid_z (w-levels)')]),
    _e ('param6','ztop',      18000.0,  'ztop (m)',     'Total domain depth'),
    _e ('param6','str_bot',       0.0,  'str_bot (m)',  'Base of stretching zone (bottom constant layer depth)'),
    _e ('param6','str_top',    2000.0,  'str_top (m)',  'Top of stretching zone (bottom of upper constant layer)'),
    _e ('param6','dz_bot',      125.0,  'dz_bot (m)',   'Grid spacing at/below str_bot'),
    _e ('param6','dz_top',      500.0,  'dz_top (m)',   'Grid spacing at/above str_top (max dz for stretch_z=2)'),

    # ── param7 — DNS ─────────────────────────────────────────────────────────
    _c ('param7','bc_temp',       1,    'bc_temp',      'θ BC at top/bottom (DNS only)',
        [(1,'constant θ specified'),(2,'constant flux specified')]),
    _e ('param7','ptc_top',     250.0,  'ptc_top',      'θ (K) or flux (K m/s) at top'),
    _e ('param7','ptc_bot',     300.0,  'ptc_bot',      'θ (K) or flux (K m/s) at bottom'),
    _e ('param7','viscosity',    25.0,  'viscosity (m²/s)','Kinematic viscosity (DNS only)'),
    _e ('param7','pr_num',        0.72, 'pr_num',       'Prandtl number (DNS only)'),

    # ── param8 — user variables ──────────────────────────────────────────────
    *[_e('param8', f'var{i}', 0.0, f'var{i}', f'User flex variable {i} (set in source code)') for i in range(1, 21)],

    # ── param9 — output fields ───────────────────────────────────────────────
    _c ('param9','output_format',    1, 'output_format',   'Output file format',
        [(1,'GrADS binary'),(2,'netCDF')]),
    _c ('param9','output_filetype',  1, 'output_filetype', 'Output file splitting',
        [(1,'one file (all times)'),(2,'one file per output time'),(3,'one file per time per MPI process')]),
    _c ('param9','output_interp',    0, 'output_interp',   'Interpolate to nominal levels (terrain sims)', _YN),
    _ci('param9','output_rain',      1, 'rain',            'Surface rainfall / swath'),
    _ci('param9','output_sws',       1, 'sws',             'Max surface wind speed swath'),
    _ci('param9','output_svs',       1, 'svs',             'Max vert vorticity at lowest level'),
    _ci('param9','output_sps',       1, 'sps',             'Min pressure perturbation at lowest level'),
    _ci('param9','output_srs',       1, 'srs',             'Max rainwater at lowest level'),
    _ci('param9','output_sgs',       1, 'sgs',             'Max graupel/hail at lowest level'),
    _ci('param9','output_sus',       1, 'sus',             'Max w at 5 km AGL (updraft swath)'),
    _ci('param9','output_shs',       1, 'shs',             'Max updraft helicity swath'),
    _ci('param9','output_coldpool',  0, 'coldpool',        'Cold pool properties (intensity, depth)'),
    _ci('param9','output_sfcflx',    0, 'sfcflx',          'Surface heat/moisture fluxes'),
    _ci('param9','output_sfcparams', 0, 'sfcparams',       'Surface/soil/ocean model parameters'),
    _ci('param9','output_sfcdiags',  0, 'sfcdiags',        'Surface-layer diagnostics (10-m winds, 2-m T)'),
    _ci('param9','output_psfc',      0, 'psfc',            'Surface pressure (z=0)'),
    _ci('param9','output_zs',        0, 'zs',              'Terrain height'),
    _ci('param9','output_zh',        0, 'zh',              'Height on model levels'),
    _ci('param9','output_basestate', 0, 'basestate',       'Base-state arrays'),
    _ci('param9','output_th',        1, 'th',              'Potential temperature'),
    _ci('param9','output_thpert',    0, 'thpert',          'θ perturbation'),
    _ci('param9','output_prs',       1, 'prs',             'Pressure'),
    _ci('param9','output_prspert',   0, 'prspert',         'Pressure perturbation'),
    _ci('param9','output_pi',        0, 'pi',              'Exner function'),
    _ci('param9','output_pipert',    0, 'pipert',          'Exner perturbation'),
    _ci('param9','output_rho',       0, 'rho',             'Dry air density'),
    _ci('param9','output_rhopert',   0, 'rhopert',         'Density perturbation'),
    _ci('param9','output_tke',       1, 'tke',             'Subgrid TKE'),
    _ci('param9','output_km',        1, 'km',              'Eddy viscosity'),
    _ci('param9','output_kh',        1, 'kh',              'Eddy diffusivity'),
    _ci('param9','output_qv',        1, 'qv',              'Water vapor mixing ratio'),
    _ci('param9','output_qvpert',    0, 'qvpert',          'qv perturbation'),
    _ci('param9','output_q',         1, 'q',               'Liquid/solid hydrometeors'),
    _ci('param9','output_dbz',       1, 'dbz',             'Reflectivity (dBZ) + composite'),
    _ci('param9','output_buoyancy',  0, 'buoyancy',        'Buoyancy'),
    _ci('param9','output_u',         1, 'u',               'u-velocity'),
    _ci('param9','output_upert',     0, 'upert',           'u perturbation'),
    _ci('param9','output_uinterp',   1, 'uinterp',         'u interpolated to scalar pts'),
    _ci('param9','output_v',         1, 'v',               'v-velocity'),
    _ci('param9','output_vpert',     0, 'vpert',           'v perturbation'),
    _ci('param9','output_vinterp',   1, 'vinterp',         'v interpolated to scalar pts'),
    _ci('param9','output_w',         1, 'w',               'w-velocity'),
    _ci('param9','output_winterp',   1, 'winterp',         'w interpolated to scalar pts'),
    _ci('param9','output_vort',      0, 'vort',            'Vorticity (3 components)'),
    _ci('param9','output_pv',        0, 'pv',              'Potential vorticity'),
    _ci('param9','output_uh',        0, 'uh',              'Updraft helicity (2-5 km AGL)'),
    _ci('param9','output_pblten',    0, 'pblten',          'PBL scheme tendencies'),
    _ci('param9','output_dissten',   0, 'dissten',         'Dissipation rate'),
    _ci('param9','output_fallvel',   0, 'fallvel',         'Hydrometeor fall velocities'),
    _ci('param9','output_nm',        0, 'nm',              'Brunt-Väisälä frequency squared'),
    _ci('param9','output_def',       0, 'def',             'Deformation'),
    _ci('param9','output_radten',    0, 'radten',          'Radiation tendencies + surface/TOA fluxes'),
    _ci('param9','output_cape',      0, 'cape',            'CAPE'),
    _ci('param9','output_cin',       0, 'cin',             'CIN'),
    _ci('param9','output_lcl',       0, 'lcl',             'LCL'),
    _ci('param9','output_lfc',       0, 'lfc',             'LFC'),
    _ci('param9','output_pwat',      0, 'pwat',            'Precipitable water'),
    _ci('param9','output_lwp',       0, 'lwp',             'Liquid water path + cloud water path'),
    _ci('param9','output_thbudget',  0, 'thbudget',        'θ budget terms'),
    _ci('param9','output_qvbudget',  0, 'qvbudget',        'qv budget terms'),
    _ci('param9','output_ubudget',   0, 'ubudget',         'u budget terms'),
    _ci('param9','output_vbudget',   0, 'vbudget',         'v budget terms'),
    _ci('param9','output_wbudget',   0, 'wbudget',         'w budget terms'),
    _ci('param9','output_pdcomp',    0, 'pdcomp',          'Pressure decomposition (no MPI)'),

    # ── param10 — statistics ─────────────────────────────────────────────────
    _ci('param10','stat_w',       1, 'w',               'Max/min vertical velocity'),
    _ci('param10','stat_wlevs',   1, 'wlevs',           'Max/min w at 5 levels (0.5,1,2.5,5,10 km)'),
    _ci('param10','stat_u',       1, 'u',               'Max/min u-velocity'),
    _ci('param10','stat_v',       1, 'v',               'Max/min v-velocity'),
    _ci('param10','stat_rmw',     1, 'rmw',             'Radius of max wind (axisymm only)'),
    _ci('param10','stat_pipert',  1, 'pipert',          'Max/min Exner perturbation'),
    _ci('param10','stat_prspert', 1, 'prspert',         'Max/min pressure perturbation'),
    _ci('param10','stat_thpert',  1, 'thpert',          'Max/min θ perturbation'),
    _ci('param10','stat_q',       1, 'q',               'Max/min moisture variables'),
    _ci('param10','stat_tke',     1, 'tke',             'Max/min subgrid TKE'),
    _ci('param10','stat_km',      1, 'km',              'Max/min eddy viscosity'),
    _ci('param10','stat_kh',      1, 'kh',              'Max/min eddy diffusivity'),
    _ci('param10','stat_div',     1, 'div',             'Max/min divergence'),
    _ci('param10','stat_rh',      1, 'rh',              'Max/min RH (liquid)'),
    _ci('param10','stat_rhi',     1, 'rhi',             'Max/min RH (ice)'),
    _ci('param10','stat_the',     1, 'the',             'Max/min equivalent potential temperature'),
    _ci('param10','stat_cloud',   1, 'cloud',           'Max/min cloud top/bottom'),
    _ci('param10','stat_sfcprs',  1, 'sfcprs',          'Max/min surface pressure'),
    _ci('param10','stat_wsp',     1, 'wsp',             'Max/min wind speed at surface'),
    _ci('param10','stat_cfl',     1, 'cfl',             'Max Courant number'),
    _ci('param10','stat_vort',    1, 'vort',            'Max vertical vorticity at several levels'),
    _ci('param10','stat_tmass',   1, 'tmass',           'Total dry-air mass'),
    _ci('param10','stat_tmois',   1, 'tmois',           'Total moisture'),
    _ci('param10','stat_qmass',   1, 'qmass',           'Total mass of each moisture variable'),
    _ci('param10','stat_tenerg',  1, 'tenerg',          'Total energy'),
    _ci('param10','stat_mo',      1, 'mo',              'Total momentum'),
    _ci('param10','stat_tmf',     1, 'tmf',             'Total up/down mass flux'),
    _ci('param10','stat_pcn',     1, 'pcn',             'Precipitation/moisture statistics'),
    _ci('param10','stat_qsrc',    1, 'qsrc',            'Moisture source statistics'),

    # ── param11 — radiation ──────────────────────────────────────────────────
    _c ('param11','radopt',       0,    'radopt',       'Atmospheric radiation scheme',
        [(0,'none'),(1,'NASA-Goddard'),(2,'RRTMG')]),
    _e ('param11','dtrad',      300.0,  'dtrad (s)',    'Radiation call interval'),
    _e ('param11','ctrlat',      36.68, 'ctrlat (°N)',  'Domain latitude'),
    _e ('param11','ctrlon',     -98.35, 'ctrlon (°E)',  'Domain longitude'),
    _e ('param11','year',        2009,  'year',         'Start year'),
    _e ('param11','month',          5,  'month',        'Start month'),
    _e ('param11','day',           15,  'day',          'Start day'),
    _e ('param11','hour',          21,  'hour',         'Start hour (UTC)'),
    _e ('param11','minute',        38,  'minute',       'Start minute'),
    _e ('param11','second',         0,  'second',       'Start second'),

    # ── param12 — surface ────────────────────────────────────────────────────
    _c ('param12','isfcflx',      0,    'isfcflx',      'Surface fluxes of heat and moisture', _YN),
    _c ('param12','sfcmodel',     0,    'sfcmodel',     'Surface model',
        [(0,'none'),(1,'original CM1'),(2,'WRF MM5/Monin-Obukhov'),(3,'revised WRF MO'),
         (4,'GFDL/HWRF'),(5,'MOST for LES'),(6,'MYNN surface layer'),(7,'MYJ surface layer')]),
    _c ('param12','oceanmodel',   0,    'oceanmodel',   'Ocean/water surface model',
        [(0,'none'),(1,'fixed SST'),(2,'ocean mixed layer')]),
    _c ('param12','initsfc',      1,    'initsfc',      'Initial surface conditions',
        [(1,'constant (set tsk0,xland0,lu0)'),(2,'sea breeze test case'),
         (3,'rough west / smooth east'),(4,'coastline (land W, ocean E)')]),
    _e ('param12','tsk0',       299.28, 'tsk0 (K)',     'Default skin temperature (soil/water ~1 cm)'),
    _e ('param12','tmn0',       297.28, 'tmn0 (K)',     'Default deep soil temperature (sfcmodel=2 only)'),
    _e ('param12','xland0',       2.0,  'xland0',       'Default land/water flag (1=land, 2=water)'),
    _e ('param12','lu0',           16,  'lu0',          'Default land-use index (16=water; see LANDUSE.TBL)'),
    _c ('param12','season',        1,   'season',       'Land-use season',
        [(1,'summer'),(2,'winter')]),
    _c ('param12','cecd',          3,   'cecd',         'Ce/Cd formulation (sfcmodel=1)',
        [(1,'constant (set cnstce/cnstcd)'),(2,"Deacon's formula (water)"),(3,'Fairall/Donelan (water)')]),
    _c ('param12','pertflx',       0,   'pertflx',      'Use only perturbation winds for surface fluxes (sfcmodel=1)', _YN),
    _e ('param12','cnstce',      0.001, 'cnstce',       'Constant Ce (cecd=1)'),
    _e ('param12','cnstcd',      0.001, 'cnstcd',       'Constant Cd (cecd=1)'),
    _c ('param12','isftcflx',      0,   'isftcflx',     'Alt Ck/Cd for tropical storms (sfcmodel=2,3,6)',
        [(0,'off'),(1,'Donelan Cd + const Z0q Ce'),(2,'Donelan Cd + Garratt Ce')]),
    _c ('param12','iz0tlnd',       0,   'iz0tlnd',      'Thermal roughness length (sfcmodel=2)',
        [(0,'Carlson-Boland'),(1,'Czil_new')]),
    _e ('param12','oml_hml0',     50.0, 'oml_hml0 (m)', 'Initial ocean mixed layer depth (oceanmodel=2)'),
    _e ('param12','oml_gamma',     0.14,'oml_gamma (K/m)','Deep water lapse rate (oceanmodel=2)'),
    _c ('param12','set_flx',       0,   'set_flx',      'Impose constant surface heat fluxes (sfcmodel=1)', _YN),
    _e ('param12','cnst_shflx',    0.24,'cnst_shflx (K m/s)','Sensible heat flux (set_flx=1)'),
    _e ('param12','cnst_lhflx',  5.2e-5,'cnst_lhflx (g/g m/s)','Latent heat flux (set_flx=1)'),
    _c ('param12','set_znt',       0,   'set_znt',      'Impose constant roughness length (sfcmodel=1)', _YN),
    _e ('param12','cnst_znt',      0.16,'cnst_znt (m)', 'Roughness length z0 (set_znt=1)'),
    _c ('param12','set_ust',       0,   'set_ust',      'Impose constant friction velocity (sfcmodel=1)', _YN),
    _e ('param12','cnst_ust',      0.25,'cnst_ust (m/s)','Friction velocity u* (set_ust=1)'),
    _c ('param12','ramp_sgs',      1,   'ramp_sgs',     'Gradually ramp up LES SGS model', _YN),
    _e ('param12','ramp_time',  1800.0, 'ramp_time (s)','Ramp-up duration for SGS model'),
    _c ('param12','t2p_avg',       1,   't2p_avg',      'Two-part model averaging (sgsmodel=3,4)',
        [(1,'spatial average'),(2,'time average per grid point')]),

    # ── param13 — parcel variables ───────────────────────────────────────────
    _ci('param13','prcl_th',    1, 'th',    'Potential temperature'),
    _ci('param13','prcl_t',     1, 't',     'Temperature'),
    _ci('param13','prcl_prs',   1, 'prs',   'Pressure'),
    _ci('param13','prcl_ptra',  1, 'ptra',  'Passive tracer (if iptra=1)'),
    _ci('param13','prcl_q',     1, 'q',     'Moisture mixing ratios'),
    _ci('param13','prcl_nc',    1, 'nc',    'Number concentrations (double-moment schemes)'),
    _ci('param13','prcl_km',    1, 'km',    'Eddy viscosity'),
    _ci('param13','prcl_kh',    1, 'kh',    'Eddy diffusivity'),
    _ci('param13','prcl_tke',   1, 'tke',   'Subgrid TKE'),
    _ci('param13','prcl_dbz',   1, 'dbz',   'Reflectivity'),
    _ci('param13','prcl_b',     1, 'b',     'Buoyancy'),
    _ci('param13','prcl_vpg',   1, 'vpg',   'Vertical perturbation pressure gradient'),
    _ci('param13','prcl_vort',  1, 'vort',  'Vertical vorticity'),
    _ci('param13','prcl_rho',   1, 'rho',   'Dry-air density'),
    _ci('param13','prcl_qsat',  1, 'qsat',  'Saturation vapor pressure'),
    _ci('param13','prcl_sfc',   1, 'sfc',   'Surface variables (z0, u*)'),

    # ── param14 — domain diagnostics ─────────────────────────────────────────
    _cb('param14','dodomaindiag', False, 'dodomaindiag','Write domain-wide averaged diagnostics'),
    _e ('param14','diagfrq',       60.0, 'diagfrq (s)', 'Frequency for domain diagnostics'),

    # ── param15 — azimuthal averaging ────────────────────────────────────────
    _cb('param15','doazimavg',    False, 'doazimavg',   'Write azimuthally averaged cross-sections'),
    _e ('param15','azimavgfrq', 3600.0,  'azimavgfrq (s)','Frequency for azimuthal output'),
    _e ('param15','rlen',      300000.0, 'rlen (m)',    'Radial extent of analysis grid'),
    _cb('param15','do_adapt_move',False, 'do_adapt_move','Adaptively adjust umove/vmove to track storm'),
    _e ('param15','adapt_move_frq',3600.0,'adapt_move_frq (s)','Frequency to update umove/vmove'),

    # ── param16 — restart ────────────────────────────────────────────────────
    _c ('param16','restart_format',    1, 'restart_format',   'Restart file format',
        [(1,'binary (recommended)'),(2,'netCDF')]),
    _c ('param16','restart_filetype',  2, 'restart_filetype', 'Restart file splitting',
        [(1,'one file'),(2,'one per restart time'),(3,'one per time per node (MPI)')]),
    _cb('param16','restart_reset_frqtim',True,'restart_reset_frqtim','Reset output times from namelist on restart'),
    _cb('param16','restart_file_theta',  False,'restart_file_theta','Include total θ in restart file'),
    _cb('param16','restart_file_dbz',    False,'restart_file_dbz',  'Include reflectivity in restart file'),
    _cb('param16','restart_file_th0',    False,'restart_file_th0',  'Include base-state θ in restart file'),
    _cb('param16','restart_file_prs0',   False,'restart_file_prs0', 'Include base-state pressure in restart file'),
    _cb('param16','restart_file_pi0',    False,'restart_file_pi0',  'Include base-state π in restart file'),
    _cb('param16','restart_file_rho0',   False,'restart_file_rho0', 'Include base-state density in restart file'),
    _cb('param16','restart_file_qv0',    False,'restart_file_qv0',  'Include base-state qv in restart file'),
    _cb('param16','restart_file_u0',     False,'restart_file_u0',   'Include base-state u in restart file'),
    _cb('param16','restart_file_v0',     False,'restart_file_v0',   'Include base-state v in restart file'),
    _cb('param16','restart_file_zs',     False,'restart_file_zs',   'Include terrain height in restart file'),
    _cb('param16','restart_file_zh',     False,'restart_file_zh',   'Include zh 3D array in restart file'),
    _cb('param16','restart_file_zf',     False,'restart_file_zf',   'Include zf 3D array in restart file'),
    _cb('param16','restart_file_diags',  False,'restart_file_diags','Include θ/qv diagnostics in restart file'),
    _cb('param16','restart_use_theta',   False,'restart_use_theta', 'Read total θ (not perturbation) on restart'),

    # ── param17 — LES subdomain ──────────────────────────────────────────────
    _c ('param17','les_subdomain_shape',  1,'les_subdomain_shape','LES subdomain shape (cm1setup=4)',
        [(1,'square'),(2,'circular (not yet implemented)')]),
    _e ('param17','les_subdomain_xlen',200000.0,'les_subdomain_xlen (m)','LES subdomain X size'),
    _e ('param17','les_subdomain_ylen',200000.0,'les_subdomain_ylen (m)','LES subdomain Y size'),
    _e ('param17','les_subdomain_dlen',200000.0,'les_subdomain_dlen (m)','LES subdomain diameter (circular only)'),
    _e ('param17','les_subdomain_trnslen',5000.0,'les_subdomain_trnslen (m)','Transition scale into LES subdomain'),

    # ── param18 — eddy recycling ─────────────────────────────────────────────
    _cb('param18','do_recycle_w', False,'do_recycle_w','Eddy recycling at west boundary'),
    _cb('param18','do_recycle_s', False,'do_recycle_s','Eddy recycling at south boundary'),
    _cb('param18','do_recycle_e', False,'do_recycle_e','Eddy recycling at east boundary'),
    _cb('param18','do_recycle_n', False,'do_recycle_n','Eddy recycling at north boundary'),
    _e ('param18','recycle_width_dx',   6.0,'recycle_width_dx (pts)','Width of recycled region (grid points)'),
    _e ('param18','recycle_depth_m', 1500.0,'recycle_depth_m (m)',   'Depth of recycled region from surface'),
    _e ('param18','recycle_cap_loc_m',4000.0,'recycle_cap_loc_m (m)','Capture location from LES domain edge'),
    _e ('param18','recycle_inj_loc_m',   0.0,'recycle_inj_loc_m (m)','Injection location from LES domain edge'),

    # ── param19 — large-scale nudging ────────────────────────────────────────
    _cb('param19','do_lsnudge',     False,'do_lsnudge',   'Enable large-scale nudging'),
    _cb('param19','do_lsnudge_u',   False,'do_lsnudge_u', 'Nudge domain-avg u profile'),
    _cb('param19','do_lsnudge_v',   False,'do_lsnudge_v', 'Nudge domain-avg v profile'),
    _cb('param19','do_lsnudge_th',  False,'do_lsnudge_th','Nudge domain-avg θ profile'),
    _cb('param19','do_lsnudge_qv',  False,'do_lsnudge_qv','Nudge domain-avg qv profile'),
    _e ('param19','lsnudge_tau',   1800.0,'lsnudge_tau (s)',        'Nudging damping timescale'),
    _e ('param19','lsnudge_start', 3600.0,'lsnudge_start (s)',      'Nudging start time'),
    _e ('param19','lsnudge_end',   7200.0,'lsnudge_end (s)',        'Nudging end time'),
    _e ('param19','lsnudge_ramp_time',600.0,'lsnudge_ramp_time (s)','Nudging ramp-up duration (0=instant)'),

    # ── param20 — immersed boundaries ────────────────────────────────────────
    _cb('param20','do_ib',   False,'do_ib',   'Enable immersed boundary method'),
    _e ('param20','ib_init',     4,'ib_init', 'IB distribution type (see ib_module.F)'),
    _e ('param20','top_cd',    0.4,'top_cd',  'Drag coefficient on top of IB surfaces'),
    _e ('param20','side_cd',   0.4,'side_cd', 'Drag coefficient on sides of IB surfaces'),

    # ── param21 — hurricane vortex ───────────────────────────────────────────
    _e ('param21','hurr_vg',      40.0,  'hurr_vg (m/s)',  'Gradient wind / initial wind speed'),
    _e ('param21','hurr_rad',  40000.0,  'hurr_rad (m)',   'Radius from TC center'),
    _e ('param21','hurr_vgpl',    -0.70, 'hurr_vgpl',      'Radial decay power law (negative)'),
    _e ('param21','hurr_rotate',   0.0,  'hurr_rotate (°)','Wind rotation angle (+ clockwise)'),

    # ── nssl2mom_params ──────────────────────────────────────────────────────
    _e ('nssl2mom_params','alphah',   0.0,  'alphah',  'Shape parameter for graupel'),
    _e ('nssl2mom_params','alphahl',  0.5,  'alphahl', 'Shape parameter for hail'),
    _e ('nssl2mom_params','ccn',      0.6e9,'ccn',     'CCN concentration (0.25e9 maritime, 0.6e9 continental)'),
    _e ('nssl2mom_params','cnor',     8.0e6,'cnor',    'Rain intercept (1-moment only)'),
    _e ('nssl2mom_params','cnoh',     4.0e4,'cnoh',    'Graupel/hail intercept (1-moment only)'),
]

# Build lookup dict: (section, key) → param tuple
PARAM_LOOKUP = {(p[0], p[1]): p for p in ALL_PARAMS}

# Section output order (matches real namelist.input)
SECTION_ORDER = [
    'param0','param1','param2','param3','param11','param12',
    'param4','param5','param6','param7','param8','param9','param16',
    'param10','param13','param14','param15','param17','param18',
    'param19','param20','param21','nssl2mom_params',
]

# ═══════════════════════════════════════════════════════════════════════════════
# Tab layout  (section, group_title, [keys...])
# ═══════════════════════════════════════════════════════════════════════════════

TAB_LAYOUT = [
    ('Grid', None),   # built specially
    ('Timing & Run', [
        ('param1', 'Grid Spacing & Time Step', ['dx','dy','dz','dtl']),
        ('param1', 'Simulation Time',          ['timax','run_time']),
        ('param1', 'Output Frequencies',       ['tapfrq','rstfrq','statfrq','prclfrq']),
        ('param2', 'Run Control',              ['adapt_dt','irst','rstnum','iconly']),
    ]),
    ('Physics', [
        ('param2', 'Model Setup',      ['cm1setup','testcase','psolver','apmasscon']),
        ('param2', 'Moisture & Thermo',['imoist','eqtset','idiss','efall','rterm']),
        ('param2', 'Coriolis',         ['icor','betaplane','lspgrad']),
        ('param3', 'Coriolis / Motion',['fcor','umove','vmove']),
    ]),
    ('Advection & Diffusion', [
        ('param2', 'Advection Scheme',          ['hadvordrs','vadvordrs','hadvordrv','vadvordrv',
                                                  'advwenos','advwenov','weno_order']),
        ('param2', 'Artificial Diffusion',      ['idiff','mdiff','difforder']),
        ('param3', 'Diffusion Coefficients',    ['kdiff2','kdiff6','kdiv']),
        ('param3', 'Pressure Solver Coefficients',['alph','cstar','csound']),
    ]),
    ('Turbulence & PBL', [
        ('param2', 'SGS Turbulence (LES)',      ['sgsmodel','tconfig','bcturbs','horizturb','doimpl']),
        ('param2', 'PBL Parameterization',      ['ipbl']),
        ('param3', 'Turbulence Length Scales',  ['l_h','lhref1','lhref2','l_inf']),
        ('param7', 'DNS Settings (cm1setup=3)', ['bc_temp','ptc_top','ptc_bot','viscosity','pr_num']),
    ]),
    ('Microphysics', [
        ('param2', 'Microphysics Scheme',       ['ptype','ihail','iautoc','nssl_3moment',
                                                  'nssl_density_on','cuparam']),
        ('param3', 'Cloud/Micro Coefficients',  ['ndcnst','nt_c','v_t']),
        ('nssl2mom_params','NSSL 2-moment Parameters',['alphah','alphahl','ccn','cnor','cnoh']),
    ]),
    ('Boundaries & Damping', [
        ('param2', 'Lateral Boundary Conditions',['wbc','ebc','sbc','nbc','irbc','roflux','nudgeobc']),
        ('param2', 'Vertical Boundary Conditions',['bbc','tbc']),
        ('param2', 'Upper Rayleigh Damping',     ['irdamp']),
        ('param3', 'Upper Damping Parameters',   ['rdalpha','zd']),
        ('param2', 'Lateral Rayleigh Damping',   ['hrdamp']),
        ('param3', 'Lateral Damping / OBC',      ['xhd','alphobc']),
    ]),
    ('Initialization', [
        ('param2', 'Base State',                 ['isnd','iwnd']),
        ('param2', 'IC Perturbation',            ['iinit','irandp','ibalance']),
        ('param2', 'Domain & Motion',            ['iorigin','axisymm','imove','itern']),
        ('param2', 'Tracers & Parcels',          ['iptra','npt','pdtra','iprcl','nparcels']),
        ('param21','Hurricane Vortex (iinit=7)', ['hurr_vg','hurr_rad','hurr_vgpl','hurr_rotate']),
    ]),
    ('Surface', [
        ('param12','Surface Setup',              ['isfcflx','sfcmodel','oceanmodel','initsfc']),
        ('param12','Surface Initial Conditions', ['tsk0','tmn0','xland0','lu0','season']),
        ('param12','sfcmodel=1 Options',         ['cecd','pertflx','cnstce','cnstcd',
                                                   'set_flx','cnst_shflx','cnst_lhflx',
                                                   'set_znt','cnst_znt','set_ust','cnst_ust',
                                                   'ramp_sgs','ramp_time','t2p_avg']),
        ('param12','sfcmodel=2,3,6 Options',     ['isftcflx','iz0tlnd']),
        ('param12','Ocean Model Options',        ['oml_hml0','oml_gamma']),
    ]),
    ('Radiation', [
        ('param11','Radiation Setup',            ['radopt','dtrad']),
        ('param11','Location & Start Time',      ['ctrlat','ctrlon','year','month','day','hour','minute','second']),
    ]),
    ('Output Fields', None),    # built specially (checkbox grid)
    ('Statistics & Diag', [
        ('param14','Domain Diagnostics',         ['dodomaindiag','diagfrq']),
        ('param15','Azimuthal Averaging',        ['doazimavg','azimavgfrq','rlen',
                                                   'do_adapt_move','adapt_move_frq']),
    ]),
    ('Restart', [
        ('param2', 'Restart Control',            ['irst','rstnum','iconly']),
        ('param16','Restart Files',              ['restart_format','restart_filetype',
                                                   'restart_reset_frqtim']),
        ('param16','Extra Restart File Contents',['restart_file_theta','restart_file_dbz',
                                                   'restart_file_th0','restart_file_prs0',
                                                   'restart_file_pi0','restart_file_rho0',
                                                   'restart_file_qv0','restart_file_u0',
                                                   'restart_file_v0','restart_file_zs',
                                                   'restart_file_zh','restart_file_zf',
                                                   'restart_file_diags','restart_use_theta']),
    ]),
    ('Special', [
        ('param17','LES Subdomain (cm1setup=4)', ['les_subdomain_shape','les_subdomain_xlen',
                                                   'les_subdomain_ylen','les_subdomain_dlen',
                                                   'les_subdomain_trnslen']),
        ('param18','Eddy Recycling',             ['do_recycle_w','do_recycle_s',
                                                   'do_recycle_e','do_recycle_n',
                                                   'recycle_width_dx','recycle_depth_m',
                                                   'recycle_cap_loc_m','recycle_inj_loc_m']),
        ('param19','Large-Scale Nudging',        ['do_lsnudge','do_lsnudge_u','do_lsnudge_v',
                                                   'do_lsnudge_th','do_lsnudge_qv',
                                                   'lsnudge_tau','lsnudge_start',
                                                   'lsnudge_end','lsnudge_ramp_time']),
        ('param20','Immersed Boundaries',        ['do_ib','ib_init','top_cd','side_cd']),
        ('param8', 'User Flex Variables',        [f'var{i}' for i in range(1, 21)]),
    ]),
]

# ═══════════════════════════════════════════════════════════════════════════════
# Main GUI
# ═══════════════════════════════════════════════════════════════════════════════

class NamelistTool(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("CM1 Namelist Editor")
        self.minsize(1100, 750)
        self._vars    = {}   # (section, key) → tk variable
        self._after   = None
        self._grid_nb = None  # sub-notebook in Grid tab
        self._grid_ax = None
        self._grid_fig = None
        self._grid_canvas = None
        self._s1_err_var = None
        self._s2_status_var = None
        self._nz_hint_var = None
        self._ztop_lbl_var = None
        self._build_vars()
        self._build_ui()
        self.after(150, self._refresh)

    # ── variable initialisation ──────────────────────────────────────────────

    def _build_vars(self):
        for p in ALL_PARAMS:
            sec, key, dflt, wtype = p[0], p[1], p[2], p[3]
            k = (sec, key)
            if wtype == 'cb':
                v = tk.BooleanVar(value=bool(dflt))
            elif wtype == 'ci':
                v = tk.IntVar(value=int(dflt))
            elif wtype == 'c':
                choices = p[6]
                display = next((f"{val} — {desc}" for val, desc in choices if val == dflt),
                               str(dflt))
                v = tk.StringVar(value=display)
            else:  # 'e'
                v = tk.StringVar(value=str(dflt))
            v.trace_add('write', self._on_change)
            self._vars[k] = v

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        # toolbar
        tb = ttk.Frame(self)
        tb.pack(fill='x', padx=6, pady=4)
        ttk.Button(tb, text="Open namelist…", command=self._load).pack(side='left', padx=3)
        ttk.Button(tb, text="Save namelist…", command=self._save).pack(side='left', padx=3)
        ttk.Button(tb, text="Copy to clipboard", command=self._copy).pack(side='left', padx=3)
        ttk.Button(tb, text="Reset defaults",    command=self._reset).pack(side='left', padx=3)

        # main pane: tabs on top, preview on bottom
        main = tk.PanedWindow(self, orient='vertical', sashwidth=5, sashrelief='ridge')
        main.pack(fill='both', expand=True, padx=6, pady=(0, 6))

        self._nb = ttk.Notebook(main)
        main.add(self._nb, minsize=420)

        # preview
        bot = ttk.LabelFrame(main, text="namelist.input  preview")
        main.add(bot, minsize=160)
        self._preview = scrolledtext.ScrolledText(
            bot, height=9, font=('Courier', 9), state='disabled', wrap='none')
        self._preview.pack(fill='both', expand=True, padx=4, pady=4)

        # build each tab
        for tab_name, layout in TAB_LAYOUT:
            frame = ttk.Frame(self._nb)
            self._nb.add(frame, text=tab_name)
            if tab_name == 'Grid':
                self._build_grid_tab(frame)
            elif tab_name == 'Output Fields':
                self._build_checkbox_tab(frame, 'param9',
                    [k for k in [p[1] for p in ALL_PARAMS if p[0]=='param9']
                     if k not in ('output_format','output_filetype','output_interp')],
                    header_keys=['output_format','output_filetype','output_interp'],
                    label_prefix='output_')
            elif tab_name == 'Statistics & Diag':
                self._build_stats_tab(frame, layout)
            else:
                self._build_form_tab(frame, layout)

        self._nb.bind("<<NotebookTabChanged>>", self._on_change)

    # ── grid tab ─────────────────────────────────────────────────────────────

    def _build_grid_tab(self, parent):
        pane = tk.PanedWindow(parent, orient='horizontal', sashwidth=5, sashrelief='ridge')
        pane.pack(fill='both', expand=True)

        left = ttk.Frame(pane)
        pane.add(left, minsize=360)

        self._grid_nb = ttk.Notebook(left)
        self._grid_nb.pack(fill='both', expand=True)
        self._grid_nb.bind("<<NotebookTabChanged>>", self._on_change)

        # sub-tabs
        th = ttk.Frame(self._grid_nb); self._grid_nb.add(th, text="Horizontal")
        tv = ttk.Frame(self._grid_nb); self._grid_nb.add(tv, text="Vertical")
        self._build_grid_horiz(th)
        self._build_grid_vert(tv)

        # right: plot + info
        right = ttk.Frame(pane)
        pane.add(right, minsize=400)

        if HAS_MPL:
            self._grid_fig    = Figure(figsize=(5, 5), dpi=96, tight_layout=True)
            self._grid_ax     = self._grid_fig.add_subplot(111)
            self._grid_canvas = FigureCanvasTkAgg(self._grid_fig, master=right)
            self._grid_canvas.get_tk_widget().pack(fill='both', expand=True)
        else:
            ttk.Label(right, text="Install matplotlib for grid preview.",
                      foreground='gray').pack(pady=60)

        self._grid_info = tk.StringVar(value="")
        ttk.Label(right, textvariable=self._grid_info,
                  font=('Courier', 9), foreground='#444',
                  justify='left').pack(fill='x', padx=8, pady=2)

    def _build_grid_horiz(self, f):
        P = dict(padx=6, pady=3)
        inner = self._make_scrollframe(f)

        gf = ttk.LabelFrame(inner, text="Domain Size  (param0)")
        gf.pack(fill='x', padx=8, pady=6)
        for i, key in enumerate(['nx','ny','nz','ppnode','timeformat','timestats',
                                   'terrain_flag','procfiles','outunits']):
            self._add_row(gf, i, 'param0', key)

        xf = ttk.LabelFrame(inner, text="Horizontal Spacing  (param1)")
        xf.pack(fill='x', padx=8, pady=4)
        self._add_row(xf, 0, 'param1', 'dx')
        self._add_row(xf, 1, 'param1', 'dy')
        self._domain_lbl = tk.StringVar(value="")
        ttk.Label(xf, textvariable=self._domain_lbl,
                  font=('Courier', 9), foreground='#226').grid(
            row=2, column=0, columnspan=3, sticky='w', padx=8)

        sxf = ttk.LabelFrame(inner, text="Stretch X  (param4)")
        sxf.pack(fill='x', padx=8, pady=4)
        for i, key in enumerate(['stretch_x','dx_inner','dx_outer','nos_x_len','tot_x_len']):
            self._add_row(sxf, i, 'param4', key)

        syf = ttk.LabelFrame(inner, text="Stretch Y  (param5)")
        syf.pack(fill='x', padx=8, pady=4)
        for i, key in enumerate(['stretch_y','dy_inner','dy_outer','nos_y_len','tot_y_len']):
            self._add_row(syf, i, 'param5', key)

    def _build_grid_vert(self, f):
        inner = self._make_scrollframe(f)

        nf = ttk.LabelFrame(inner, text="Vertical Size  (param0 / param1)")
        nf.pack(fill='x', padx=8, pady=6)
        self._add_row(nf, 0, 'param0', 'nz')
        self._add_row(nf, 1, 'param1', 'dz')
        self._add_row(nf, 2, 'param1', 'dtl')
        self._dtl_hint_var = tk.StringVar(value="")
        ttk.Label(nf, textvariable=self._dtl_hint_var,
                  foreground='gray', font=('Courier', 9)).grid(
            row=3, column=0, columnspan=3, sticky='w', padx=8)

        zf = ttk.LabelFrame(inner, text="Vertical Stretching  (param6)")
        zf.pack(fill='x', padx=8, pady=4)
        for i, key in enumerate(['stretch_z','ztop','str_bot','str_top','dz_bot','dz_top']):
            self._add_row(zf, i, 'param6', key)

        # sz=0 label
        self._ztop_lbl_var = tk.StringVar(value="")
        ttk.Label(zf, textvariable=self._ztop_lbl_var,
                  font=('Courier', 9), foreground='#226').grid(
            row=6, column=0, columnspan=3, sticky='w', padx=8, pady=2)

        # sz=1 hints
        nz_row = ttk.Frame(zf)
        nz_row.grid(row=7, column=0, columnspan=3, sticky='ew', padx=6)
        self._nz_hint_var = tk.StringVar(value="")
        ttk.Label(nz_row, textvariable=self._nz_hint_var,
                  font=('Courier', 9), foreground='#226').pack(side='left')
        ttk.Button(nz_row, text="Apply nz →",
                   command=self._apply_nz).pack(side='left', padx=6)

        self._s1_err_var = tk.StringVar(value="")
        ttk.Label(zf, textvariable=self._s1_err_var,
                  foreground='red', wraplength=340, justify='left',
                  font=('Courier', 9)).grid(
            row=8, column=0, columnspan=3, sticky='w', padx=6)

        # sz=2 status
        self._s2_status_var = tk.StringVar(value="")
        ttk.Label(zf, textvariable=self._s2_status_var,
                  foreground='#226', font=('Courier', 9)).grid(
            row=9, column=0, columnspan=3, sticky='w', padx=6, pady=2)

        # Sponge
        sf = ttk.LabelFrame(inner, text="Rayleigh Damping  (param2/3)")
        sf.pack(fill='x', padx=8, pady=4)
        for i, (sec, key) in enumerate([('param2','irdamp'),('param3','rdalpha'),
                                         ('param3','zd'),('param2','hrdamp'),('param3','xhd')]):
            self._add_row(sf, i, sec, key)

    # ── generic scrollable form tab ──────────────────────────────────────────

    def _build_form_tab(self, parent, layout):
        inner = self._make_scrollframe(parent)
        for (section, group_title, keys) in layout:
            lf = ttk.LabelFrame(inner, text=f"{group_title}  ({section})")
            lf.pack(fill='x', padx=8, pady=5)
            for row, key in enumerate(keys):
                self._add_row(lf, row, section, key)

    # ── output fields tab ────────────────────────────────────────────────────

    def _build_checkbox_tab(self, parent, section, keys, header_keys=None, label_prefix=''):
        inner = self._make_scrollframe(parent)

        if header_keys:
            hf = ttk.LabelFrame(inner, text="File Format")
            hf.pack(fill='x', padx=8, pady=6)
            for row, key in enumerate(header_keys):
                self._add_row(hf, row, section, key)

        cf = ttk.LabelFrame(inner, text="Fields  (0=off, 1=on)")
        cf.pack(fill='both', expand=True, padx=8, pady=4)
        NCOLS = 4
        for idx, key in enumerate(keys):
            p = PARAM_LOOKUP.get((section, key))
            if p is None: continue
            lbl = p[4]  # short label
            var = self._vars[(section, key)]
            cb = ttk.Checkbutton(cf, text=lbl, variable=var)
            cb.grid(row=idx // NCOLS, column=idx % NCOLS, sticky='w', padx=8, pady=1)

    # ── statistics + parcel tab ──────────────────────────────────────────────

    def _build_stats_tab(self, parent, layout):
        inner = self._make_scrollframe(parent)

        # param10 stat flags
        sf = ttk.LabelFrame(inner, text="Statistical Output  (param10)")
        sf.pack(fill='x', padx=8, pady=5)
        stat_keys = [p[1] for p in ALL_PARAMS if p[0] == 'param10']
        NCOLS = 4
        for idx, key in enumerate(stat_keys):
            p = PARAM_LOOKUP[('param10', key)]
            ttk.Checkbutton(sf, text=p[4], variable=self._vars[('param10', key)]).grid(
                row=idx // NCOLS, column=idx % NCOLS, sticky='w', padx=8, pady=1)

        # param13 parcel flags
        pf = ttk.LabelFrame(inner, text="Parcel Output  (param13 — requires iprcl=1)")
        pf.pack(fill='x', padx=8, pady=5)
        prcl_keys = [p[1] for p in ALL_PARAMS if p[0] == 'param13']
        for idx, key in enumerate(prcl_keys):
            p = PARAM_LOOKUP[('param13', key)]
            ttk.Checkbutton(pf, text=p[4], variable=self._vars[('param13', key)]).grid(
                row=idx // NCOLS, column=idx % NCOLS, sticky='w', padx=8, pady=1)

        # domain diag + azimavg
        for (section, group_title, keys) in layout:
            lf = ttk.LabelFrame(inner, text=f"{group_title}  ({section})")
            lf.pack(fill='x', padx=8, pady=5)
            for row, key in enumerate(keys):
                self._add_row(lf, row, section, key)

    # ── widget helpers ───────────────────────────────────────────────────────

    def _make_scrollframe(self, parent):
        """Return an inner frame inside a canvas+scrollbar."""
        canvas = tk.Canvas(parent, borderwidth=0, highlightthickness=0)
        vsb    = ttk.Scrollbar(parent, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        inner  = ttk.Frame(canvas)
        win_id = canvas.create_window((0, 0), window=inner, anchor='nw')

        def _resize(evt):
            canvas.configure(scrollregion=canvas.bbox('all'))
        def _width(evt):
            canvas.itemconfigure(win_id, width=evt.width)

        inner.bind('<Configure>', _resize)
        canvas.bind('<Configure>', _width)
        canvas.bind_all('<MouseWheel>',
                        lambda e: canvas.yview_scroll(-1 * (e.delta // 120), 'units'))

        vsb.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)
        return inner

    def _add_row(self, parent, row, section, key):
        p = PARAM_LOOKUP.get((section, key))
        if p is None:
            return
        sec, k, dflt, wtype, lbl, hint, choices = p
        var = self._vars[(sec, k)]

        ttk.Label(parent, text=lbl + ':', anchor='e').grid(
            row=row, column=0, sticky='e', padx=6, pady=2)

        if wtype == 'e':
            ttk.Entry(parent, textvariable=var, width=14).grid(
                row=row, column=1, sticky='w', padx=4, pady=2)
        elif wtype == 'c':
            values = [f"{v} — {d}" for v, d in choices]
            ttk.Combobox(parent, textvariable=var, values=values,
                         width=36, state='readonly').grid(
                row=row, column=1, sticky='w', padx=4, pady=2)
        elif wtype == 'cb':
            ttk.Checkbutton(parent, variable=var, text="").grid(
                row=row, column=1, sticky='w', padx=4, pady=2)
        elif wtype == 'ci':
            ttk.Checkbutton(parent, variable=var, text="").grid(
                row=row, column=1, sticky='w', padx=4, pady=2)

        if hint:
            ttk.Label(parent, text=hint, foreground='gray',
                      font=('TkDefaultFont', 8)).grid(
                row=row, column=2, sticky='w', padx=4)

    # ── value accessors ──────────────────────────────────────────────────────

    def _getv(self, section, key, fallback=None):
        v = self._vars.get((section, key))
        if v is None:
            return fallback
        try:
            raw = v.get()
            if isinstance(raw, bool):
                return raw
            if isinstance(raw, int):
                return raw
            if isinstance(raw, str):
                tok = raw.split()[0]
                # try int first, then float
                try:   return int(tok)
                except ValueError: pass
                try:   return float(tok)
                except ValueError: return tok
            return raw
        except Exception:
            return fallback

    def _getf(self, section, key, fallback=0.0):
        v = self._getv(section, key, fallback)
        try:
            return float(v)
        except (TypeError, ValueError):
            return fallback

    def _geti(self, section, key, fallback=0):
        v = self._getv(section, key, fallback)
        try:
            return int(v)
        except (TypeError, ValueError):
            return fallback

    # ── apply nz helper ──────────────────────────────────────────────────────

    def _apply_nz(self):
        try:
            nk1, nk2, nk3 = suggest_nz_s1(
                self._getf('param6','str_bot'), self._getf('param6','str_top'),
                self._getf('param6','dz_bot'),  self._getf('param6','dz_top'),
                self._getf('param6','ztop'))
            self._vars[('param0','nz')].set(str(nk1 + nk2 + nk3))
        except Exception:
            pass

    # ── change / refresh ─────────────────────────────────────────────────────

    def _on_change(self, *_):
        if self._after:
            self.after_cancel(self._after)
        self._after = self.after(280, self._refresh)

    def _refresh(self):
        self._after = None
        try:
            self._update_grid_hints()
            self._draw_grid()
            self._update_preview()
        except Exception:
            pass

    # ── grid hints ───────────────────────────────────────────────────────────

    def _update_grid_hints(self):
        nx = self._geti('param0','nx', 200)
        ny = self._geti('param0','ny', 200)
        nz = self._geti('param0','nz', 40)
        dx = self._getf('param1','dx', 2000.0)
        dy = self._getf('param1','dy', 2000.0)
        dz = self._getf('param1','dz', 500.0)
        dtl = self._getf('param1','dtl', 7.5)
        sx  = self._geti('param4','stretch_x', 0)
        sy  = self._geti('param5','stretch_y', 0)
        sz  = self._geti('param6','stretch_z', 0)
        ztop    = self._getf('param6','ztop',    18000.0)
        str_bot = self._getf('param6','str_bot',     0.0)
        str_top = self._getf('param6','str_top',  2000.0)
        dz_bot  = self._getf('param6','dz_bot',   125.0)
        dz_top  = self._getf('param6','dz_top',   500.0)

        # domain label
        dom_x = self._getf('param4','tot_x_len', nx*dx) if sx > 0 else nx * dx
        dom_y = self._getf('param5','tot_y_len', ny*dy) if sy > 0 else ny * dy
        if hasattr(self, '_domain_lbl'):
            self._domain_lbl.set(f"Domain: {dom_x/1000:.1f} km × {dom_y/1000:.1f} km")

        # dtl suggestion
        min_d = min(dx, dy, dz_bot if sz > 0 else dz)
        if hasattr(self, '_dtl_hint_var') and self._dtl_hint_var:
            self._dtl_hint_var.set(f"min(dx,dy,dz)/{min_d:.0f} / 67 ≈ {min_d/67:.2f} s suggested")

        # sz=0 ztop
        if hasattr(self, '_ztop_lbl_var') and self._ztop_lbl_var:
            if sz == 0:
                self._ztop_lbl_var.set(f"→ ztop = {nz} × {dz:.0f} = {nz*dz/1000:.2f} km")
            else:
                self._ztop_lbl_var.set("")

        # sz=1 hints
        if hasattr(self, '_nz_hint_var') and self._nz_hint_var:
            if sz == 1:
                nk1, nk2, nk3 = suggest_nz_s1(str_bot, str_top, dz_bot, dz_top, ztop)
                self._nz_hint_var.set(
                    f"nk1={nk1}  nk2={nk2}  nk3={nk3}  →  nz = {nk1+nk2+nk3}")
            else:
                self._nz_hint_var.set("")

        if hasattr(self, '_s1_err_var') and self._s1_err_var:
            if sz == 1:
                _, errs = zf_stretch1(nz, ztop, str_bot, str_top, dz_bot, dz_top)
                self._s1_err_var.set(
                    "\n".join(errs) if errs else "✓  stretch_z=1 parameters are valid")
            else:
                self._s1_err_var.set("")

        if hasattr(self, '_s2_status_var') and self._s2_status_var:
            if sz == 2:
                _, errs, r = zf_stretch2(nz, dz, dz_bot, str_bot, dz_top)
                self._s2_status_var.set(
                    "\n".join(errs) if errs else f"✓  r = {r:.6f}")
            else:
                self._s2_status_var.set("")

    # ── grid plot ────────────────────────────────────────────────────────────

    def _draw_grid(self):
        if not HAS_MPL or self._grid_fig is None:
            return

        # which sub-tab?
        try:
            sub = self._grid_nb.tab(self._grid_nb.select(), 'text')
        except Exception:
            sub = 'Vertical'

        nx = self._geti('param0','nx', 200)
        ny = self._geti('param0','ny', 200)
        nz = self._geti('param0','nz', 40)
        dx = self._getf('param1','dx', 2000.0)
        dy = self._getf('param1','dy', 2000.0)
        dz = self._getf('param1','dz', 500.0)
        sx = self._geti('param4','stretch_x', 0)
        sy = self._geti('param5','stretch_y', 0)
        sz = self._geti('param6','stretch_z', 0)
        ztop    = self._getf('param6','ztop',    18000.0)
        str_bot = self._getf('param6','str_bot',     0.0)
        str_top = self._getf('param6','str_top',  2000.0)
        dz_bot  = self._getf('param6','dz_bot',   125.0)
        dz_top  = self._getf('param6','dz_top',   500.0)
        zd      = self._getf('param3','zd',      15000.0)
        irdamp  = self._geti('param2','irdamp',       1)

        self._grid_fig.clear()

        if sub == 'Horizontal':
            ax1 = self._grid_fig.add_subplot(211)
            ax2 = self._grid_fig.add_subplot(212)
            for ax, s, n, d_in, d_out, nos, tot, lbl in [
                (ax1, sx, nx, self._getf('param4','dx_inner',1000.), self._getf('param4','dx_outer',7000.),
                 self._getf('param4','nos_x_len',40000.), self._getf('param4','tot_x_len',120000.), 'X'),
                (ax2, sy, ny, self._getf('param5','dy_inner',1000.), self._getf('param5','dy_outer',7000.),
                 self._getf('param5','nos_y_len',40000.), self._getf('param5','tot_y_len',120000.), 'Y'),
            ]:
                xf, ni1, ni2, ni3, errs = _horiz_xf(s, n, d_in, d_out, nos, tot)
                if xf is None or errs:
                    ax.text(0.5, 0.5, f"{lbl}: {errs[0] if errs else 'invalid'}",
                            ha='center', va='center', transform=ax.transAxes, color='red')
                    continue
                xh = 0.5 * (xf[:-1] + xf[1:]) / 1000.0
                dv = np.diff(xf)
                ax.step(xh, dv, where='mid', color='royalblue', lw=1.6)
                ax.scatter(xh, dv, s=14, color='royalblue', zorder=4)
                if s > 0 and ni2 > 0:
                    ax.axvspan(xh[ni1], xh[ni1+ni2-1], alpha=0.12, color='green',
                               label=f'inner ({d_in:.0f} m)')
                    ax.legend(fontsize=8, loc='upper center')
                ax.set_xlabel(f'{lbl} (km)', fontsize=9)
                ax.set_ylabel(f'd{lbl.lower()} (m)', fontsize=9)
                ax.set_title(f'{lbl}: n={len(dv)}, domain={xf[-1]/1000-xf[0]/1000:.1f} km, '
                             f'dmin={dv.min():.0f}, dmax={dv.max():.0f} m', fontsize=9)
                ax.set_ylim(bottom=0)
                ax.grid(True, alpha=0.25, lw=0.6)
        else:
            ax = self._grid_fig.add_subplot(111)
            # compute zf
            zf = None
            if sz == 0:
                zf = zf_uniform(nz, dz)
            elif sz == 1:
                zf, errs = zf_stretch1(nz, ztop, str_bot, str_top, dz_bot, dz_top)
            elif sz == 2:
                result = zf_stretch2(nz, dz, dz_bot, str_bot, dz_top)
                zf = result[0]

            if zf is None:
                ax.text(0.5, 0.5, "Invalid grid — check Vertical tab",
                        ha='center', va='center', transform=ax.transAxes, color='red')
            else:
                zh   = 0.5 * (zf[:-1] + zf[1:])
                dz_k = np.diff(zf)
                z_km = zh / 1000.0
                ztop_km = zf[-1] / 1000.0
                if irdamp > 0 and 0 < zd < zf[-1]:
                    ax.axhspan(zd/1000., ztop_km, alpha=0.12, color='tomato',
                               label=f'Rayleigh (zd={zd/1000:.0f} km)')
                ax.step(dz_k, z_km, where='mid', color='royalblue', lw=1.8)
                ax.scatter(dz_k, z_km, s=18, color='royalblue', zorder=4)
                if sz == 1:
                    for h, lbl2, col in [(str_bot/1000., f'str_bot={str_bot/1000.:.1f}km','#2a7a2a'),
                                          (str_top/1000., f'str_top={str_top/1000.:.1f}km','#b07000')]:
                        if 0 < h < ztop_km:
                            ax.axhline(h, ls='--', lw=1, color=col, alpha=0.8)
                            ax.text(dz_k.max()*0.98, h+ztop_km*0.005, lbl2,
                                    ha='right', va='bottom', fontsize=8, color=col)
                ax.set_xlabel('dz (m)', fontsize=10)
                ax.set_ylabel('Height (km)', fontsize=10)
                ax.set_title(f'Vertical grid — nz={nz}, ztop={ztop_km:.2f} km '
                             f'(stretch_z={sz})', fontsize=10)
                ax.set_ylim(0, ztop_km * 1.02)
                ax.set_xlim(left=0)
                ax.grid(True, alpha=0.25, lw=0.6)
                if irdamp > 0 and 0 < zd < zf[-1]:
                    ax.legend(fontsize=8, loc='lower right')
                self._grid_info.set(
                    f"ztop={ztop_km:.3f} km  dz_min={dz_k.min():.1f} m  dz_max={dz_k.max():.1f} m")

        self._grid_fig.tight_layout(pad=1.0)
        self._grid_canvas.draw_idle()

    # ── namelist generation ──────────────────────────────────────────────────

    def _fmt_val(self, section, key):
        """Format the value for namelist output."""
        p = PARAM_LOOKUP.get((section, key))
        if p is None:
            return ''
        wtype = p[3]
        var   = self._vars[(section, key)]
        try:
            raw = var.get()
        except Exception:
            return str(p[2])

        if wtype == 'cb':
            return '.true.' if raw else '.false.'
        if wtype == 'ci':
            return '1' if raw else '0'
        if wtype == 'c':
            tok = str(raw).split()[0]
            try:   return str(int(tok))
            except ValueError:
                try: return str(float(tok))
                except ValueError: return tok
        # entry — detect int vs float from default
        dflt = p[2]
        s = str(raw).strip()
        if isinstance(dflt, bool):
            return s
        if isinstance(dflt, int):
            try:   return str(int(float(s)))
            except ValueError: return s
        # float
        try:
            f = float(s)
            # use scientific if very small or very large
            if f != 0 and (abs(f) < 1e-3 or abs(f) >= 1e9):
                return f'{f:.10e}'
            return f'{f:g}'
        except ValueError:
            return s

    def _generate_nl(self):
        lines = []
        sections_seen = set()

        # collect all keys per section in schema order
        by_sec = {}
        for p in ALL_PARAMS:
            sec = p[0]
            if sec not in by_sec:
                by_sec[sec] = []
            by_sec[sec].append(p[1])

        for sec in SECTION_ORDER:
            if sec not in by_sec:
                continue
            lines.append(f"\n &{sec}")
            for key in by_sec[sec]:
                val = self._fmt_val(sec, key)
                lines.append(f" {key:<24s} = {val},")
            lines.append(" /")

        return '\n'.join(lines).lstrip('\n')

    def _update_preview(self):
        txt = self._generate_nl()
        self._preview.config(state='normal')
        self._preview.delete('1.0', 'end')
        self._preview.insert('end', txt)
        self._preview.config(state='disabled')

    # ── load / save ──────────────────────────────────────────────────────────

    def _load(self):
        path = filedialog.askopenfilename(
            title="Open namelist.input",
            filetypes=[('Namelist / input files', '*.input'),
                       ('All files', '*.*')])
        if not path:
            return
        try:
            with open(path) as fh:
                text = fh.read()
        except Exception as e:
            messagebox.showerror("Load error", str(e))
            return
        parsed = parse_namelist(text)
        for (sec, key), var in self._vars.items():
            val = parsed.get(sec, {}).get(key)
            if val is None:
                continue
            p = PARAM_LOOKUP[(sec, key)]
            wtype = p[3]
            try:
                if wtype == 'cb':
                    var.set(bool(val))
                elif wtype == 'ci':
                    var.set(1 if val else 0)
                elif wtype == 'c':
                    choices = p[6]
                    int_val = int(val) if not isinstance(val, bool) else int(val)
                    display = next((f"{v} — {d}" for v, d in choices if v == int_val),
                                   str(int_val))
                    var.set(display)
                else:
                    var.set(str(val))
            except Exception:
                pass
        self.title(f"CM1 Namelist Editor — {os.path.basename(path)}")

    def _save(self):
        path = filedialog.asksaveasfilename(
            initialfile='namelist.input',
            defaultextension='.input',
            filetypes=[('Namelist / input files', '*.input'), ('All files', '*.*')],
            title="Save namelist.input")
        if not path:
            return
        try:
            with open(path, 'w') as fh:
                fh.write(self._generate_nl() + '\n')
            self.title(f"CM1 Namelist Editor — {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Save error", str(e))

    def _copy(self):
        self.clipboard_clear()
        self.clipboard_append(self._generate_nl())

    def _reset(self):
        if not messagebox.askyesno("Reset", "Reset all values to defaults?"):
            return
        for p in ALL_PARAMS:
            sec, key, dflt, wtype = p[0], p[1], p[2], p[3]
            var = self._vars[(sec, key)]
            if wtype == 'cb':
                var.set(bool(dflt))
            elif wtype == 'ci':
                var.set(int(dflt))
            elif wtype == 'c':
                choices = p[6]
                display = next((f"{v} — {d}" for v, d in choices if v == dflt), str(dflt))
                var.set(display)
            else:
                var.set(str(dflt))


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    app = NamelistTool()
    app.mainloop()
