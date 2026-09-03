# Python user `.pt` force provider

Build the Cython bridge once in this directory:

```bash
python setup_user_pt_force.py build_ext --inplace
```

Use `force_calculation.mode: python` (or combine it with `builtin` and/or
`libtorch`) and set `mlip_config.model_type: user_pt`.  The model is loaded once
from `mlip_config.model_path`.  Its TorchScript forward signature is expected to
be `forward(positions, types)`, returning `[N,3]` forces or an energy tensor.
The bridge receives zero-copy NumPy views of the C++ host particle buffers and
adds returned forces to the existing force buffer.
