# CM1tools

GUI tools for setting up and analyzing [CM1](https://www2.mmm.ucar.edu/people/bryan/cm1/) (Cloud Model 1) simulations.

## Tools

### cm1view.py — Output Viewer

Interactive viewer for CM1 netCDF output files.

**Views**
- Plan view (x–y at a fixed height level)
- X–Z cross section (at fixed y index)
- Y–Z cross section (at fixed x index)

**Features**
- Open individual files or an entire output directory
- Live mode: watches a directory and auto-advances to new files as they are written
- Time slider, back/forward buttons, and auto-play
- Wind overlay: arrows or barbs (u/v on plan view; u/w or v/w on cross sections)
- Contour overlay with adjustable min/max, levels, linewidth, and ±symmetric option
- Extract a sounding at any clicked point (requires `sounderpy`)
- Save current frame as PNG
- Export a time range as an animated GIF

**Usage**
```
python cm1view.py [file_or_directory ...]
```

---

### namelisttool.py — Namelist Editor

GUI editor for `namelist.input` with real-time grid validation.

**Features**
- Tabbed interface covering all CM1 namelist sections
- Horizontal and vertical grid visualizers with stretch-grid math ported from CM1's `param.F`
- Validation of vertical grid constraints (nk1/nk2/nk3 divisibility checks) with suggestions for valid `nz` values
- Load and save `namelist.input` files

**Usage**
```
python namelisttool.py [namelist.input]
```

---

### soundingtool.py — Sounding Tool

Fetch observed or model soundings, compute Bunkers storm motion, and export to CM1 `input_sounding` format.

**Features**
- Fetch soundings from observed data (SPC archive) or model analysis via `sounderpy`
- Compute Bunkers right-mover storm motion using MetPy
- Preview sounding data and storm-motion parameters
- Export directly to CM1 `input_sounding` format (z, θ, qv, u, v)

**Usage**
```
python soundingtool.py
```

---

## Dependencies

```
pip install netCDF4 matplotlib numpy sounderpy metpy
```

Tkinter is included with standard Python distributions.
