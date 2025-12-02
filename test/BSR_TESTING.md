# Running BSR Tests

The Block Sparse Row (BSR) format tests can be run independently using the BSR test group.

## Running BSR Tests Only

There are several ways to run only the BSR tests:

### Method 1: Using Pkg.test
```julia
using Pkg
Pkg.test("HierarchicalEOM", test_args=["BSR"])
```

### Method 2: Using GROUP environment variable
```bash
GROUP=BSR julia --project test/runtests.jl
```

### Method 3: From Julia REPL
```julia
ENV["GROUP"] = "BSR"
include("test/runtests.jl")
```

## BSR Test Coverage

The BSR test group includes 6 test items:

1. **BlockSparseRowMatrix basics** - Tests the core BSR matrix data structure
   - Block storage and indexing
   - Element access
   - Block retrieval
   - Conversion to sparse matrix

2. **BSRBuilder** - Tests the BSR matrix builder with deduplication
   - Adding blocks
   - Automatic deduplication
   - Block construction

3. **BSROperator** - Tests SciML operator interface
   - Matrix-vector multiplication
   - Operator properties
   - Conversion and caching

4. **M_Boson with BSR** - Tests BSR with bosonic HEOMLS matrices
   - Uses Drude-Lorentz spectral density
   - Compares BSR vs CSC formats
   - Validates numerical accuracy
   - Measures memory savings

5. **M_Fermion with BSR** - Tests BSR with fermionic HEOMLS matrices
   - Uses Lorentz spectral density
   - Compares BSR vs CSC formats
   - Validates numerical accuracy
   - Measures memory savings

6. **M_Boson_Fermion with BSR** - Tests BSR with mixed bath HEOMLS matrices
   - Uses both bosonic and fermionic spectral densities
   - Compares BSR vs CSC formats
   - Validates numerical accuracy
   - Measures memory savings

## Test Parameters

All HEOM tests use realistic physical parameters:
- Coupling strength: λ = 0.1450
- Reorganization energy: W = 0.6464
- Temperature: kT = 0.7414
- Chemical potential: μ = 0.8787 (fermion only)
- Expansion terms: N = 2-3
- Hierarchy tier: tier = 2

## Available Test Groups

- `All` - Runs all tests (default)
- `Core` - Core functionality tests
- `BSR` - Block Sparse Row format tests
- `Code-Quality` - Code quality checks (Aqua, JET)
- `CUDA_Ext` - CUDA extension tests
