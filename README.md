# BEASTGPU

BEASTGPU is the GPU-accelerated backend for the BEAST package.  
It enables running large-scale computations efficiently on compatible GPUs.

## Running

Before running BEASTGPU, activate the project environment:

Run the following steps **in order** to set the number of threads, start Julia, activate the environment, and run BEASTGPU:


# 1. Set the number of threads (Linux/macOS)
```bash
export JULIA_NUM_THREADS=8
```

# On Windows PowerShell:
```bash
# $env:JULIA_NUM_THREADS = 8
```

# 2. Start Julia
```bash
julia 
```

```julia
using Pkg;
Pkg.activate(".");
include("run.jl");
```
