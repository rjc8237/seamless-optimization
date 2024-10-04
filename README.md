# Symmetric Dirichlet Optimization

An implementation of uv-coordinate based optimization for the symmetric Dirichlet energy.

### Overview

This method takes as input a cut mesh with a uv-parametrization and edge tags marking edges where seamless constraints should be imposed. The edge markings are a list of the form
```
#seamless-edges
vi0 vj0 vj1 vi1
...
```
Each line is the vertex pairing for edge $e_{ij}$ in faces $f_{ijk}$ and $f_{jil}$, with vertices `vi0` and `vj0` in face $f_{ijk}$ and vertices `vj1` and `vi1` in face $f_{jil}$. The mesh must be stored as `<indput-dir>/<model>_init.obj` and the edge correspondence stored in `<input-dir>/EE/<model>_EE.txt`.

Note that the current input data structure is not optimal or standard, and it requires a particular bookkeeping scheme that is cumbersome exposes too much implementation. It would be better to define separate face matrices for the 3D embedding and uv-coordinates and then infer seamless edges are those cut in the parametric domain but not in the embedding.

## Installation

To install this project on a Unix-based system, use the following standard CMake build procedure:

```bash
git clone https://github.com/rjc8237/symmetric-dirichlet.git
cd symmetric-dirichlet
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j 4
```

## Usage

The core method is `bin/symmetric_dirichlet`. This executable takes the following arguments:

|flag | description|
| --- | --- |
|`--input` | directory containing meshes and edge markings|
|`--model` | mesh name|
|`--output` | (optional) output directory|
|`--json` | (optional) path to configuration json|
