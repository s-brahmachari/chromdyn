# chromdyn_pbc Updates

This document describes the major updates and improvements in `chromdyn_pbc` compared to the original `chromdyn` package.

## Overview

The `chromdyn_pbc` module is an enhanced version of the original `chromdyn` package with critical bug fixes and new features for periodic boundary condition (PBC) simulations and advanced trajectory analysis.

---

## Major Updates

### 1. Periodic Boundary Condition (PBC) Support

**Status**: ✅ Fully Implemented

The new version provides complete support for PBC simulations under NVT ensemble conditions.

#### Key Changes:

- **[ChromatinDynamics.py](file:///c:/Users/Leren/Documents/GitHub/Mycode/chromdyn_new/chromdyn_pbc/ChromatinDynamics.py)**
  - Added `PBC` parameter to the `__init__` method
  - Added `box_vectors` parameter to define simulation box dimensions
  - Automatic box setup when `PBC=True`
  
- **[Reporters.py](file:///c:/Users/Leren/Documents/GitHub/Mycode/chromdyn_new/chromdyn_pbc/Reporters.py)**
  - `SaveStructure` class now accepts `PBC` parameter
  - Box vectors are automatically saved as HDF5 attributes for each frame when `PBC=True`
  - `save_pdb()` function updated to write CRYST1 records for PBC simulations
  - Added `enforcePeriodicBox` parameter when getting state

#### Usage Example:

```python
from chromdyn_pbc import ChromatinDynamics

# Define box vectors (in nanometers)
box_vectors = ((50.0, 0.0, 0.0), (0.0, 50.0, 0.0), (0.0, 0.0, 50.0))

# Initialize with PBC
sim = ChromatinDynamics(
    topology=my_topology,
    PBC=True,
    box_vectors=box_vectors,
    # ... other parameters
)
```

---

### 2. Fixed Data Loss Bug: Topology and ChromSeq Preservation

**Status**: ✅ Critical Bug Fix

The original `chromdyn` package had a critical bug where `topology` and `ChromSeq` information were **not saved** to the trajectory file, making it difficult to reload and analyze simulations later.

#### What Was Fixed:

- **[Reporters.py](file:///c:/Users/Leren/Documents/GitHub/Mycode/chromdyn_new/chromdyn_pbc/Reporters.py)** - `SaveStructure` class:
  - Added `_save_full_topology()` method that serializes the complete OpenMM topology to JSON
  - Topology is now saved as `topology_json` dataset in the HDF5 file
  - Includes complete chain/residue/atom hierarchy and bond information
  
- **[Cndbtools.py](file:///c:/Users/Leren/Documents/GitHub/Mycode/chromdyn_new/chromdyn_pbc/Cndbtools.py)** - `ChromatinTrajectory` class:
  - Added `TopologyData` helper class to parse and access topology information
  - `load_trajectory()` now reads and parses the saved topology
  - `ChromSeq` is properly reconstructed from the saved topology

#### Topology Storage Format:

The topology is stored as a JSON string with the following structure:

```json
{
  "chains": [
    {
      "index": 0,
      "id": "C1",
      "residues": [
        {
          "index": 0,
          "name": "L0",
          "id": "0",
          "atoms": [
            {"index": 0, "name": "A0", "type": "A"}
          ]
        }
      ]
    }
  ],
  "bonds": [[0, 1], [1, 2], ...]
}
```

---

### 3. Enhanced Trajectory Analysis Tools

**Status**: ✅ New Features Added

#### 3.1 Standardized Trajectory Loading

Both [Cndbtools.py](file:///c:/Users/Leren/Documents/GitHub/Mycode/chromdyn_new/chromdyn_pbc/Cndbtools.py) and [tools.py](file:///c:/Users/Leren/Documents/GitHub/Mycode/chromdyn_new/chromdyn_pbc/tools.py) now provide a consistent `ChromatinTrajectory` class for loading trajectories:

```python
from chromdyn_pbc.Cndbtools import ChromatinTrajectory

# Load trajectory
traj = ChromatinTrajectory("simulation.cndb")

# Access data
print(f"Number of beads: {traj.Nbeads}")
print(f"Number of frames: {traj.Nframes}")
print(f"Bead types: {traj.Types}")
print(f"Topology: {traj.Topology}")

# Get coordinates
coords = traj.xyz(frames=[0, 100, 1])  # frames 0-99
```

#### 3.2 PBC Visualization Tools

New functions for visualizing PBC simulations:

- **`visualize_pbc()`** - Visualize a single frame with PBC box
- **`visualize_animation_pbc()`** - Create animations of PBC trajectories
- **`recenter_coordinates_v3()`** - Recenter coordinates based on box vectors
- **`_draw_generic_box()`** - Draw parallelepiped box for non-cubic boxes

**Features:**
- Automatic recentering of polymer chains
- Color-coding by chain or bead type
- Support for both cubic and non-cubic (parallelepiped) boxes
- Export to image files or animated GIFs/MP4s

#### 3.3 GPU-Accelerated Velocity Analysis (CuPy)

**Location**: [Cndbtools.py](file:///c:/Users/Leren/Documents/GitHub/Mycode/chromdyn_new/chromdyn_pbc/Cndbtools.py)

New GPU-accelerated functions for velocity correlation analysis using CuPy:

- **`calculate_vacf_gpu()`** - Velocity autocorrelation function (VACF)
  - Computes C(t) = ⟨v(0)·v(t)⟩ for all bead types
  - Uses FFT for efficient computation
  - Returns results for general and per-type analysis

- **`calculate_spatial_vel_corr_gpu()`** - Spatial velocity correlation
  - Computes C(r) = ⟨v_i·v_j⟩ as a function of distance
  - Vectorized GPU implementation for O(N²) calculations
  - Returns binned correlation data

**Requirements:**
- CuPy must be installed (`pip install cupy-cuda11x` or `cupy-cuda12x`)
- If CuPy is not available, functions will not be loaded (graceful degradation)

**Usage Example:**

```python
from chromdyn_pbc.Cndbtools import calculate_vacf_gpu

# Load velocities from trajectory
velocities = ...  # shape: (n_frames, n_beads, 3)
bead_types = traj.Types

# Calculate VACF
vacf_results = calculate_vacf_gpu(
    velocities=velocities,
    bead_types=bead_types,
    sampling_step=10
)

# Results contain 'general' and per-type VACF
import matplotlib.pyplot as plt
plt.plot(vacf_results['general'])
plt.xlabel('Time lag')
plt.ylabel('VACF')
plt.show()
```

---

### 4. Tutorial Notebook

**Location**: [chromdyn_new/notebooks/PBC_sim_test.ipynb](file:///c:/Users/Leren/Documents/GitHub/Mycode/chromdyn_new/notebooks/PBC_sim_test.ipynb)

A comprehensive Jupyter notebook demonstrating:
- Setting up PBC simulations
- Running NVT simulations with periodic boundaries
- Loading and analyzing PBC trajectories
- Visualizing PBC systems
- Computing velocity correlations with GPU acceleration

---

## File-by-File Summary

### Modified Core Files

| File | Key Changes |
|------|-------------|
| [ChromatinDynamics.py](file:///c:/Users/Leren/Documents/GitHub/Mycode/chromdyn_new/chromdyn_pbc/ChromatinDynamics.py) | Added `PBC` and `box_vectors` parameters |
| [Reporters.py](file:///c:/Users/Leren/Documents/GitHub/Mycode/chromdyn_new/chromdyn_pbc/Reporters.py) | Added topology serialization, PBC box saving |
| [Cndbtools.py](file:///c:/Users/Leren/Documents/GitHub/Mycode/chromdyn_new/chromdyn_pbc/Cndbtools.py) | Added `TopologyData` class, PBC visualization, GPU velocity analysis |
| [tools.py](file:///c:/Users/Leren/Documents/GitHub/Mycode/chromdyn_new/chromdyn_pbc/tools.py) | Added PBC visualization functions |

### New Utility Functions

**Visualization:**
- `visualize_pbc()` - Single frame PBC visualization
- `visualize_animation_pbc()` - PBC trajectory animation
- `recenter_coordinates_v3()` - Coordinate recentering
- `_draw_generic_box()` - Generic box drawing
- `_draw_pbc_box()` - Cubic PBC box drawing

**Analysis (GPU-accelerated):**
- `calculate_vacf_gpu()` - Velocity autocorrelation function
- `calculate_spatial_vel_corr_gpu()` - Spatial velocity correlation
- `_autocorrFFT_gpu()` - FFT-based autocorrelation helper

---

## Migration Guide

### From `chromdyn` to `chromdyn_pbc`

1. **Import changes**: Replace `from chromdyn import ...` with `from chromdyn_pbc import ...`

2. **Enable PBC** (if needed):
   ```python
   # Old
   sim = ChromatinDynamics(topology=topo)
   
   # New (with PBC)
   sim = ChromatinDynamics(
       topology=topo,
       PBC=True,
       box_vectors=((50, 0, 0), (0, 50, 0), (0, 0, 50))
   )
   ```

3. **Trajectory loading**: No changes needed, but now you get topology and ChromSeq automatically!

4. **Visualization**: Use new PBC-aware visualization functions for better results

---

## Backward Compatibility

✅ **Fully backward compatible** - All existing `chromdyn` code will work with `chromdyn_pbc` without modifications. PBC features are opt-in via the `PBC` parameter.

---

## Requirements

- Python 3.8+
- OpenMM 7.7+
- NumPy
- h5py
- matplotlib
- scipy
- **Optional**: CuPy (for GPU-accelerated velocity analysis)

---

## Known Issues & Limitations

1. **CuPy dependency**: GPU velocity analysis requires CuPy installation. If not available, these functions will not be loaded.
2. **Box shape**: Current PBC visualization assumes parallelepiped boxes (triclinic boxes not fully tested).

---

## Future Enhancements

- [ ] Support for NPT ensemble simulations
- [ ] Additional GPU-accelerated analysis functions
- [ ] Improved visualization for large systems
- [ ] Trajectory compression options

---

## Questions or Issues?

For questions about these updates or to report issues, please contact the development team or open an issue in the repository.

---

**Last Updated**: 2025-11-19
