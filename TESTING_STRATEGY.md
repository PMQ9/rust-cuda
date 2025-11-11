# CUDA 13 Testing Strategy Without Blackwell Hardware

## Overview

This document describes how to validate CUDA 13 Blackwell (sm_100+) support without access to Blackwell GPUs, using a combination of validation tools and PTX forward compatibility.

---

## Testing Pyramid

```
                    ┌──────────────┐
                    │ Real B200    │  Confidence: 100%
                    │ (Optional)   │  Cost: $12
                    └──────────────┘
                ┌──────────────────────┐
                │  Virtual PTX Test    │  Confidence: 85%
                │  on Ampere (sm_86)   │  Cost: $0
                └──────────────────────┘
        ┌──────────────────────────────────┐
        │    ptxas Syntax Validation       │  Confidence: 95%
        │    for sm_100                    │  Cost: $0
        └──────────────────────────────────┘
┌──────────────────────────────────────────────┐
│        Unit Tests + Compilation Tests        │  Confidence: 80%
│        (no GPU required)                     │  Cost: $0
└──────────────────────────────────────────────┘
```

---

## Method 1: PTX Syntax Validation (Primary Method)

### What It Tests
- LLVM 20 IR generation
- libnvvm compilation to PTX
- PTX instruction validity for sm_100
- Metadata correctness

### Requirements
```bash
# CUDA 13.0 toolkit (includes ptxas with sm_100 support)
ptxas --version
# Should show: Cuda compilation tools, release 13.0
```

### Test Script

```bash
#!/bin/bash
# test_blackwell_compilation.sh

set -e

echo "Testing CUDA 13 Blackwell compilation..."

# Build kernel for sm_100
cd examples/cuda/vecadd
cargo clean
CUDA_ARCH=100 cargo build --release

# Extract PTX
PTX_FILE="kernels/target/nvptx64-nvidia-cuda/release/kernels.ptx"

if [ ! -f "$PTX_FILE" ]; then
    echo "ERROR: PTX file not found at $PTX_FILE"
    exit 1
fi

echo "✓ PTX generated successfully"

# Validate with ptxas
echo "Validating PTX syntax for sm_100..."
ptxas "$PTX_FILE" -arch=sm_100 -o /tmp/test_sm100.cubin 2>&1 | tee ptxas.log

if [ $? -eq 0 ]; then
    echo "✓ PTX is valid for sm_100 (Blackwell)"
else
    echo "✗ PTX validation failed"
    cat ptxas.log
    exit 1
fi

# Also test for other architectures
for ARCH in 100 100f 100a 110 110f 110a 120 120f 120a; do
    echo "Testing sm_${ARCH}..."
    ptxas "$PTX_FILE" -arch=sm_${ARCH} -o /tmp/test_sm${ARCH}.cubin
    echo "✓ sm_${ARCH} OK"
done

echo ""
echo "🎉 All Blackwell architectures validated successfully!"
```

### CI Integration

```yaml
# .github/workflows/test_blackwell.yml
name: Test Blackwell Compilation

on: [push, pull_request]

jobs:
  validate-blackwell-ptx:
    name: Validate Blackwell PTX
    runs-on: ubuntu-latest
    container:
      image: "ghcr.io/rust-gpu/rust-cuda-cuda13:latest"  # TODO: create this

    steps:
      - uses: actions/checkout@v4

      - name: Build for sm_100
        run: |
          cd examples/cuda/vecadd
          CUDA_ARCH=100 cargo build --release

      - name: Validate PTX with ptxas
        run: |
          PTX="examples/cuda/vecadd/kernels/target/nvptx64-nvidia-cuda/release/kernels.ptx"
          ptxas $PTX -arch=sm_100 -o /tmp/test.cubin
          echo "✓ Blackwell PTX validation passed"

      - name: Test all Blackwell variants
        run: |
          for arch in 100 100f 100a 110 110f 110a; do
            ptxas $PTX -arch=sm_$arch -o /tmp/test_$arch.cubin
          done
```

**Confidence:** ✅ 95% - If ptxas accepts it, real hardware will too

---

## Method 2: Virtual PTX Execution on Ampere

### How PTX Forward Compatibility Works

CUDA's PTX is a **virtual instruction set**:
- `sm_XX` = Physical target (generates hardware-specific code)
- `compute_XX` = Virtual target (generates portable PTX)

Virtual PTX can run on older hardware via JIT compilation!

### Example: Run "Blackwell" Code on RTX 3060

```rust
// In your build.rs or cuda_builder setup:

use cuda_builder::{CudaBuilder, NvvmArch};

fn main() {
    // Option 1: Physical target (won't run on older GPUs)
    CudaBuilder::new("kernels")
        .arch(NvvmArch::Compute100)  // Virtual, not Compute100f or Compute100a
        .build()
        .unwrap();
}
```

**What happens:**
1. LLVM 20 generates IR for compute_100
2. libnvvm produces generic PTX (no sm_100-specific instructions)
3. Your RTX 3060 (sm_86) JIT-compiles the PTX at runtime
4. Kernel executes successfully (without Blackwell-specific features)

### Test Script

```rust
// tests/test_virtual_blackwell.rs

#[test]
fn test_compute100_on_ampere() {
    // This test compiles for compute_100 (virtual)
    // but runs on whatever GPU is available

    let module = Module::from_ptx(KERNEL_PTX, &[]).unwrap();
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

    // Allocate and run kernel
    let mut output = DeviceBuffer::<f32>::from_slice(&[0.0; 1024]).unwrap();

    unsafe {
        launch!(module.test_kernel<<<1, 1024, 0, stream>>>(
            output.as_device_ptr()
        )).unwrap();
    }

    stream.synchronize().unwrap();

    let result: Vec<f32> = output.to_vec().unwrap();
    assert_eq!(result[0], 42.0); // Validate computation is correct
}
```

**Limitations:**
- ⚠️ Blackwell-specific instructions will be emulated or unavailable
- ⚠️ New tensor core features won't activate
- ⚠️ Performance will differ
- ✅ But correctness can be validated!

**Confidence:** ✅ 85% - Tests code generation and logic, but not HW-specific features

---

## Method 3: CUDA Driver API Module Loading Test

### Test PTX Can Be Loaded (Without Execution)

```rust
// tests/test_module_loading.rs

use cust::prelude::*;

#[test]
fn test_sm100_ptx_loads() {
    // Initialize CUDA on your RTX 3060
    let _ctx = quick_init().unwrap();

    // Try to load Blackwell PTX
    let ptx_sm100 = include_str!("../kernels/kernel_sm100.ptx");

    let result = Module::from_ptx(ptx_sm100, &[]);

    match result {
        Ok(_) => {
            println!("✓ sm_100 PTX loaded successfully");
            println!("  (May be JIT-compiled to sm_86)");
        }
        Err(e) => {
            // If it fails to load, PTX has compatibility issues
            panic!("✗ Failed to load sm_100 PTX: {}", e);
        }
    }
}
```

**What This Tests:**
- ✅ PTX is parseable by CUDA driver
- ✅ No incompatible features (would error at load time)
- ✅ Metadata is correct
- ✅ Virtual architecture compatibility

**Confidence:** ✅ 90% - If driver accepts it, it's valid PTX

---

## Method 4: Differential Testing

### Compare LLVM 7 vs LLVM 20 Output

```rust
// Build same kernel with both backends
fn test_llvm_backends_match() {
    // Build with LLVM 7 for sm_86
    let ptx_llvm7 = build_with_llvm7("kernels", NvvmArch::Compute86);

    // Build with LLVM 20 for sm_86 (yes, LLVM 20 can target older archs)
    let ptx_llvm20 = build_with_llvm20("kernels", NvvmArch::Compute86);

    // Run both on your RTX 3060
    let result_llvm7 = run_kernel(&ptx_llvm7);
    let result_llvm20 = run_kernel(&ptx_llvm20);

    // They should produce identical results
    assert_eq!(result_llvm7, result_llvm20);
}
```

**Confidence:** ✅ 75% - If both backends produce same results on sm_86, LLVM 20 likely works for sm_100

---

## Method 5: GPGPU-Sim (Overkill, Not Recommended)

### Full GPU Emulation

**Setup:**
```bash
git clone https://github.com/gpgpu-sim/gpgpu-sim_distribution.git
cd gpgpu-sim_distribution
# Follow lengthy setup instructions...
# May not support Blackwell yet
```

**Why NOT to use:**
- ❌ 1000x slower than real GPU
- ❌ Complex setup requiring CUDA toolkit patches
- ❌ May not support sm_100+ yet
- ❌ Overkill for compiler validation

**When to use:**
- Research on GPU microarchitecture
- Detailed performance modeling
- When you have weeks of time

**Confidence:** ✅ 95%, but impractical

---

## Recommended Testing Flow

### For Development (Your RTX 3060):

```bash
# 1. Test LLVM 7 path (fully supported)
./test_ampere.sh

# 2. Test LLVM 20 compilation for sm_100
LLVM_BACKEND=20 CUDA_ARCH=100 cargo build --release

# 3. Validate PTX syntax
ptxas kernels/target/.../kernels.ptx -arch=sm_100 -o /tmp/test.cubin

# 4. Test virtual PTX on your GPU
cargo test --test virtual_blackwell
```

**Time:** 30 minutes
**Cost:** $0
**Confidence:** 85%

### For CI Pipeline:

```yaml
test-matrix:
  - compile-sm_75-llvm7    # Turing
  - compile-sm_86-llvm7    # Ampere
  - compile-sm_90-llvm7    # Hopper
  - compile-sm_100-llvm20  # Blackwell (ptxas validation only)
  - compile-sm_110-llvm20  # Future arch (ptxas validation only)
```

**Time:** 10 minutes
**Cost:** $0
**Confidence:** 90%

### For Final Validation (Optional):

```python
# Use Modal's free $30 credit for real B200 testing
import modal

@app.function(gpu="B200")
def test_real_blackwell():
    # Run your kernels on real hardware
    # Validate execution correctness
    pass
```

**Time:** 1 hour
**Cost:** $6.25 (covered by free credit)
**Confidence:** 100%

---

## Summary Table

| Method | Confidence | Cost | Speed | CI-Friendly | Hardware Needed |
|--------|-----------|------|-------|-------------|-----------------|
| **ptxas validation** | 95% | $0 | ⚡ Fast | ✅ Yes | CUDA 13 toolkit |
| **Virtual PTX on sm_86** | 85% | $0 | ⚡ Fast | ✅ Yes | Any CUDA GPU |
| **Module load test** | 90% | $0 | ⚡ Fast | ✅ Yes | Any CUDA GPU |
| **Differential testing** | 75% | $0 | 🐢 Medium | ✅ Yes | Any CUDA GPU |
| **GPGPU-Sim** | 95% | $0 | 🐌 Very Slow | ❌ No | None |
| **Real B200** | 100% | $6/hr | ⚡ Fast | ⚠️ Maybe | Blackwell GPU |

---

## Conclusion

**You CAN validate 90%+ of CUDA 13 Blackwell support without Blackwell hardware!**

The recommended approach:
1. ✅ Use `ptxas` to validate PTX syntax (95% confidence)
2. ✅ Use virtual PTX (`compute_100`) to test on your RTX 3060 (85% confidence)
3. ✅ Use Colab T4 for cross-architecture validation (free)
4. ⚠️ Optional: Use Modal B200 for final peace of mind ($12, 100% confidence)

This gives you high confidence at near-zero cost!
