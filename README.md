# chromdyn
*A Python library for chromosome dynamics simulation and analysis with options to integrate Hi-C data.*

## Overview

`chromdyn` provides tools for building, simulating, and analyzing 3D chromosome structures.  
It uses **OpenMM** for fast CPU/GPU simulations and includes:

- Polymer topology generation  
- Customizable force fields  
- Hi-C-based restraints  
- Simulation utilities  
- Analysis tools

## Installation

### From PyPI

```
pip install chromdyn
```

### From source (development)

```
git clone https://github.com/s-brahmachari/chromdyn
cd chromdyn
pip install -e .
```

## Verify Installation

Run the built-in test script:

```
python -m chromdyn.check_install
```

## Quick Example

```python
from chromdyn import ChromatinDynamics, TopologyGenerator

gen = TopologyGenerator()
gen.gen_top([100])

sim = ChromatinDynamics(
    gen.topology,
    name="demo",
    platform_name="CPU",
)

sim.force_field_manager.add_harmonic_bonds()
sim.simulation_setup(
    init_struct="randomwalk",
    integrator="langevin",
    temperature=300.0,
    timestep=0.01,
)

sim.run(1000)
```

## Tutorials

See the `notebooks/` directory for examples on:

- Building polymer models  
- Running simulations  
- Using Hi-C restraints  
- Analyzing trajectories  


## License

MIT License.
