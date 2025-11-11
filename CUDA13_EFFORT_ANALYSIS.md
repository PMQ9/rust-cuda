# CUDA 13 Support - Effort Analysis

**Date:** 2025-11-11
**Prepared by:** Senior Software Engineer
**Status:** Technical Feasibility Study

## Executive Summary

Adding CUDA 13.0 support to rust-cuda requires **MAJOR** effort due to fundamental changes in NVIDIA's NVVM IR specification. The project currently targets LLVM 7.x exclusively, but CUDA 13.0 introduces a dual-dialect system requiring both LLVM 7.0.1 (legacy) and LLVM 20.1.0 (modern). This represents a **breaking architectural change** similar to PR #227's LLVM 19 attempt.

**Estimated Effort:** 6-8 weeks for a single experienced developer
**Complexity:** High
**Risk Level:** Medium-High (ABI compatibility, dual toolchain management)

---

## Current State Analysis

### Supported CUDA Versions
- **Officially Supported:** CUDA 11.2 - 11.8
- **Experimental:** CUDA 12.x (see issue #100)
- **Not Supported:** CUDA 13.0+

### Current Architecture
```
Rust Code → rustc_codegen_nvvm → LLVM 7.x IR → libnvvm → PTX → GPU
```

**Key Constraints:**
- **Fixed LLVM Version:** 7.0-7.4 (hardcoded in `build.rs:16`)
- **NVVM IR Version:** 1.6+ (checked at `nvvm.rs:61-66`)
- **Minimum CUDA:** 11.2 (libnvvm 1.6 requirement)
- **No Dynamic Version Selection:** All version checks happen at build time

**Location References:**
- Version detection: `crates/cust_raw/build/cuda_sdk.rs:77-99`
- LLVM requirement: `crates/rustc_codegen_nvvm/build.rs:16`
- NVVM version check: `crates/rustc_codegen_nvvm/src/nvvm.rs:61-66`
- Compute capabilities: `crates/nvvm/src/lib.rs`

---

## CUDA 13.0 Breaking Changes

### 1. NVVM IR Version Incompatibility
- **Old:** NVVM IR 1.x (LLVM 7.0.1 based)
- **New:** NVVM IR 2.0 (incompatible with 1.x)

⚠️ **BREAKING:** Version 2.0 is not backward compatible with version 1.x

### 2. Dual-Dialect LLVM Requirement

CUDA 13.0 requires **TWO** LLVM versions:

| Architecture | LLVM Version | NVVM IR | Compute Capability |
|-------------|--------------|---------|-------------------|
| Legacy (Pre-Blackwell) | LLVM 7.0.1 | 1.11 / 2.0 | sm_75 - sm_90a |
| Modern (Blackwell+) | LLVM 20.1.0 | 2.0 only | sm_100+ |

**Implications:**
- Cannot target Blackwell (sm_100+) with LLVM 7.x
- Blackwell requires LLVM 20.1.0 minimum
- Must maintain parallel compilation paths

### 3. PTX ISA Changes
- **PTX Version:** 9.0 (new in CUDA 13)
- **New Architectures:** sm_110, sm_110f, sm_110a
- **Deprecated:** Maxwell architectures (sm_50-53) - already documented in codebase

### 4. API Removals
- `cudaLaunchCooperativeKernelMultiDevice` removed
- Display driver no longer bundled on Windows

---

## Technical Challenges

### Challenge 1: Dual LLVM Support
**Current:** Single LLVM 7.x installation
**Required:** Both LLVM 7.0.1 AND LLVM 20.1.0

**Approach (from PR #227 discussion):**
> "Ideally both are compiled in statically or as dylibs and runtime chooses based on arch selected"

**Complexity Factors:**
- Build system must link two different LLVM versions
- C++ ABI compatibility between LLVM 7 and LLVM 20
- Distribution size increase (LLVM is ~500MB per version)
- Dynamic library loading complexity

**Files Requiring Major Changes:**
- `crates/rustc_codegen_nvvm/build.rs` (entire LLVM detection/linking logic)
- `crates/rustc_codegen_nvvm/rustc_llvm_wrapper/*.cpp` (may need version-specific wrappers)

### Challenge 2: Runtime Architecture Detection
**Current:** Compile-time only (`CudaBuilder::arch()`)
**Required:** Runtime selection of LLVM backend based on target architecture

**Implementation Strategy:**
```rust
// Pseudocode
fn select_llvm_backend(arch: NvvmArch) -> LlvmBackend {
    match arch {
        Compute75 | Compute80 | Compute86 | Compute87 | Compute89 | Compute90 | Compute90a
            => LlvmBackend::V7,
        Compute100 | Compute100f | Compute100a | Compute101 | Compute103 | Compute120 | Compute121
            => LlvmBackend::V20,
    }
}
```

**Affected Components:**
- `crates/cuda_builder/src/lib.rs:59-200` - Add backend selection
- `crates/rustc_codegen_nvvm/src/nvvm.rs` - Conditional compilation paths
- `crates/nvvm/src/lib.rs` - Extended `NvvmArch` enum (already has sm_100+ definitions)

### Challenge 3: NVVM IR Version Negotiation
**Current:** Simple version check for IR 1.6+
**Required:** Support both IR 1.x and 2.0, with version negotiation

**Code Location:** `crates/rustc_codegen_nvvm/src/nvvm.rs:61-66`

**New Logic Needed:**
```rust
let (major, minor) = nvvm::ir_version();
let nvvm_version = nvvm::nvvm_version();

// CUDA 13+ uses NVVM IR 2.0
if major >= 2 {
    // Use CUDA 13+ codepath
    // Requires LLVM 20 for sm_100+, supports LLVM 7 for legacy
} else if major == 1 && minor >= 6 {
    // Use CUDA 11.2-12.x codepath (current)
} else {
    fatal("Minimum libnvvm 1.6 required");
}
```

### Challenge 4: Test Infrastructure
**Current:** CI tests on CUDA 12.8.1 only
**Required:** Matrix testing across CUDA 11.8, 12.8, and 13.0

**CI Changes Required:** `.github/workflows/ci_linux.yml`
- Add CUDA 13.0 container images
- Test both LLVM 7 and LLVM 20 paths
- Validate cross-architecture compilation

---

## Detailed Implementation Plan

### Phase 1: Preparation (1 week)
**Tasks:**
1. Set up CUDA 13.0 development environment
2. Install both LLVM 7.0.1 and LLVM 20.1.0
3. Research libnvvm 4.0 API changes (CUDA 13 version)
4. Create feature branch with comprehensive tests
5. Document NVVM IR 2.0 differences

**Deliverables:**
- Working CUDA 13.0 + dual LLVM environment
- Compatibility matrix document
- Test plan

### Phase 2: Build System Refactoring (2 weeks)
**Primary Files:**
- `crates/rustc_codegen_nvvm/build.rs`
- `crates/rustc_codegen_nvvm/Cargo.toml`

**Tasks:**
1. Refactor LLVM detection to support multiple versions
   - Add `LLVM7_CONFIG` and `LLVM20_CONFIG` environment variables
   - Modify `find_llvm_config()` to return version-specific paths
   - Update prebuilt LLVM download logic (line 13-124)

2. Dual LLVM linking strategy:
   - Option A: Static linking both versions with name mangling
   - Option B: Dynamic library loading at runtime (recommended)
   - Implement C++ wrapper abstraction layer

3. Update C++ wrappers:
   - `rustc_llvm_wrapper/RustWrapper.cpp`
   - `rustc_llvm_wrapper/PassWrapper.cpp`
   - Add version dispatch logic

**Success Criteria:**
- Can build with LLVM 7.0.1 OR LLVM 20.1.0
- Both versions linkable in single binary
- No symbol conflicts

### Phase 3: Codegen Backend Adaptation (2 weeks)
**Primary Files:**
- `crates/rustc_codegen_nvvm/src/nvvm.rs`
- `crates/rustc_codegen_nvvm/src/ptxgen.rs`
- `crates/rustc_codegen_nvvm/src/back/`

**Tasks:**
1. Implement architecture-based backend selection:
   ```rust
   pub enum LlvmBackend {
       V7,   // For sm_75 through sm_90a
       V20,  // For sm_100 and newer
   }
   ```

2. Modify `codegen_bitcode_modules()` (nvvm.rs:52-67):
   - Add NVVM IR version detection
   - Route to appropriate LLVM backend
   - Handle NVVM IR 2.0 metadata requirements

3. Update LLVM IR generation:
   - Conditionally generate LLVM 7 vs LLVM 20 IR
   - Handle opaque pointer differences (LLVM 15+ feature)
   - Update debug metadata for both versions

4. Test PTX generation for both paths:
   - Validate sm_75-90a targets with LLVM 7
   - Validate sm_100+ targets with LLVM 20

**Success Criteria:**
- Can compile simple kernels for both sm_75 and sm_100
- PTX output validates with ptxas
- No IR version mismatch errors

### Phase 4: API and Library Updates (1 week)
**Primary Files:**
- `crates/cuda_builder/src/lib.rs`
- `crates/nvvm/src/lib.rs`
- `crates/cust_raw/build/cuda_sdk.rs`

**Tasks:**
1. Extend `CudaBuilder`:
   ```rust
   pub struct CudaBuilder {
       // ... existing fields ...
       pub llvm_backend: Option<LlvmBackend>,  // Auto-detect or manual override
       pub cuda_version: CudaVersion,           // 11.x, 12.x, 13.x
   }
   ```

2. Add CUDA 13 version detection:
   - Enhance `cuda_sdk.rs` to detect CUDA 13.x
   - Add version-specific feature flags
   - Update `driver_version()` handling

3. Compute capability validation:
   - Warn if using LLVM 7 for sm_100+ (impossible)
   - Warn if using LLVM 20 for sm_75 (inefficient)

**Success Criteria:**
- `CudaBuilder` auto-selects correct LLVM version
- Clear error messages for invalid configurations
- Backward compatible with CUDA 11.x/12.x

### Phase 5: Testing & Documentation (1-2 weeks)
**Tasks:**
1. **Unit Tests:**
   - Test LLVM version detection logic
   - Test architecture-to-backend mapping
   - Test NVVM IR version negotiation

2. **Integration Tests:**
   - Compile all examples with CUDA 13
   - Test vecadd, gemm, sha2 examples
   - Validate PTX across all compute capabilities

3. **CI/CD:**
   - Create CUDA 13 container images
   - Add CUDA 13 to build matrix
   - Set up LLVM 7 + LLVM 20 CI runners

4. **Documentation:**
   - Update `guide/src/guide/getting_started.md`
   - Document LLVM version requirements
   - Add migration guide for CUDA 13
   - Update README.md

**Success Criteria:**
- All existing tests pass on CUDA 11/12/13
- CI green on all supported CUDA versions
- Documentation complete and accurate

---

## Alternative Approaches

### Option 1: CUDA 13 Only (Simpler)
**Effort:** 3-4 weeks
**Trade-off:** Drop CUDA 11.x/12.x support, require LLVM 20 only

**Pros:**
- Simpler implementation (single LLVM version)
- Cleaner codebase
- Future-focused

**Cons:**
- Breaks existing users on CUDA 11.x/12.x
- Forces hardware upgrade (Blackwell only for LLVM 20 features)
- Community backlash likely

### Option 2: CUDA 13 with LLVM 7 Only (Limited)
**Effort:** 2-3 weeks
**Trade-off:** Support CUDA 13 but not Blackwell (sm_100+) architectures

**Pros:**
- Minimal changes to build system
- Maintains single LLVM version
- Easier to maintain

**Cons:**
- Cannot target latest Blackwell GPUs
- Defeats main purpose of CUDA 13 upgrade
- Partial feature support

### Option 3: Dual LLVM with Runtime Selection (Recommended)
**Effort:** 6-8 weeks
**Trade-off:** Complex implementation, maximum compatibility

**Pros:**
- Full CUDA 11.2 through 13.x support
- All architectures supported (sm_75 through sm_120+)
- Future-proof design
- Matches NVIDIA's recommended approach

**Cons:**
- Highest implementation complexity
- Larger binary size / installation footprint
- Longer development timeline

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| LLVM ABI incompatibility between versions | Medium | High | Use dynamic loading with isolation |
| NVVM IR 2.0 semantic differences | Medium | High | Extensive testing on CUDA 13 libnvvm |
| Performance regression with LLVM 20 | Low | Medium | Benchmark suite before/after |
| Breaking changes to public API | Medium | Medium | Deprecation period, semantic versioning |
| CI infrastructure complexity | High | Low | Incremental rollout, fallback to manual testing |
| Upstream rustc changes | Low | High | Pin to specific nightly version (already done) |
| Community pushback on breaking changes | Medium | Medium | Clear communication, migration guide |

**Critical Risks:**
- **LLVM Symbol Conflicts:** If both LLVM 7 and 20 are statically linked, symbol collisions are likely. **Mitigation:** Use dynamic loading or namespace isolation.
- **Debugging Complexity:** Dual-path codegen makes debugging harder. **Mitigation:** Comprehensive logging and deterministic backend selection.

---

## Resource Requirements

### Development Resources
- **Senior Rust Developer:** 6-8 weeks full-time
- **CUDA/LLVM Expertise:** Required (libnvvm, LLVM IR, PTX)
- **Hardware:** Access to Blackwell GPU for testing (sm_100+)

### Infrastructure
- **LLVM Installations:** Both 7.0.1 and 20.1.0 (~1GB disk space)
- **CUDA Versions:** 11.8, 12.8, 13.0 (~15GB disk space)
- **CI Runners:** GPU-enabled runners for integration tests (expensive)

### External Dependencies
- NVIDIA CUDA 13.0 Toolkit (released August 2025)
- LLVM 20.1.0 (verify compatibility with rustc internals)
- Updated container images for CI

---

## Validation & Testing Strategy

### Test Matrix
```
CUDA Version: [11.8, 12.8, 13.0]
LLVM Version: [7.0.1, 20.1.0]
Architecture: [sm_75, sm_80, sm_86, sm_89, sm_90, sm_100, sm_110]
Platform: [x86_64, ARM64]
OS: [Ubuntu 22.04, Ubuntu 24.04, RockyLinux 9]
```

**Estimated Test Combinations:** ~126 configurations (filtered by validity)

### Success Metrics
1. ✅ All existing examples compile and run on CUDA 13
2. ✅ PTX generated for sm_75 (LLVM 7) matches CUDA 12 baseline
3. ✅ PTX generated for sm_100 (LLVM 20) validates with ptxas
4. ✅ No performance regression >5% on existing benchmarks
5. ✅ Zero compiler crashes or panics in test suite
6. ✅ Documentation coverage >90%

---

## Comparison to PR #227 (LLVM 19 Support)

### Similarities
- Both require multi-version LLVM support
- Both target new GPU architectures (Blackwell)
- Both involve significant build system changes

### Key Differences
| Aspect | PR #227 (LLVM 19) | CUDA 13 Support |
|--------|-------------------|-----------------|
| **LLVM Versions** | 7 → 19 migration | 7 + 20 dual support |
| **NVVM IR Version** | 1.x only | 1.x + 2.0 |
| **Backward Compat** | Unclear | Must maintain 11.2+ |
| **Status** | Stalled (SIGSEGV issues) | Not started |
| **Complexity** | High | Higher (dual path) |

### Lessons from PR #227
1. ⚠️ **ThinLTO Crashes:** LLVM 19 had SIGSEGV during ThinLTO passes. Must validate LLVM 20 stability.
2. ⚠️ **Long Integration Time:** PR open for months with multiple rebases. Budget extra time.
3. ✅ **Community Preference:** Maintainers want dual-version support, not wholesale replacement.
4. ✅ **Architecture-Based Selection:** The approach of selecting LLVM version by target arch is validated.

---

## Recommended Path Forward

### Short Term (Next 2 Weeks)
1. **Prototype Dual LLVM Loading:**
   - Create proof-of-concept with both LLVM 7 and LLVM 20 statically linked
   - Test symbol isolation techniques
   - Validate C++ ABI compatibility

2. **CUDA 13 Environment Setup:**
   - Install CUDA 13.0 in development environment
   - Test libnvvm 4.0 API with simple examples
   - Identify actual breaking changes vs. documented changes

3. **Community Engagement:**
   - Open GitHub issue outlining this plan
   - Request feedback from maintainers
   - Coordinate with PR #227 author if still active

### Medium Term (2-3 Months)
1. **Implement Dual LLVM Support** (Phase 2)
2. **Adapt Codegen Backend** (Phase 3)
3. **Iterative Testing** with community feedback

### Long Term (3-6 Months)
1. **Full CI Integration** (Phase 5)
2. **Production Hardening**
3. **Deprecate CUDA 11.x** (optional, based on community)

---

## Open Questions

1. **LLVM 20 Stability:** Is LLVM 20.1.0 stable enough for production use with rustc internals?
   - **Action:** Test with rustc nightly, check for crashes/bugs

2. **libnvvm 4.0 ABI Changes:** Are there undocumented breaking changes in libnvvm 4.0?
   - **Action:** Diff API headers between CUDA 12.8 and 13.0

3. **Prebuilt LLVM Distribution:** Should we provide prebuilt LLVM 20 binaries?
   - **Current:** Only LLVM 7 prebuilt for Windows (build.rs:13-124)
   - **Recommendation:** Yes, for both Windows and Linux

4. **Compute Capability Minimum:** Should we drop support for sm_35-sm_70?
   - **Current:** Supports sm_35+ (nvvm/src/lib.rs)
   - **CUDA 13:** Only officially supports sm_75+
   - **Recommendation:** Keep for CUDA 11.x/12.x, deprecate warning for CUDA 13

5. **Incremental Rollout:** Should CUDA 13 support be behind a feature flag initially?
   - **Recommendation:** Yes, use `cuda-13` Cargo feature for opt-in during development

---

## Conclusion

Adding CUDA 13 support with dynamic LLVM version selection is **feasible but complex**. The recommended approach is:

1. ✅ **Dual LLVM Support** (LLVM 7.0.1 + LLVM 20.1.0)
2. ✅ **Runtime Backend Selection** based on target architecture
3. ✅ **Backward Compatibility** with CUDA 11.2+
4. ✅ **Incremental Rollout** via feature flag

**Timeline:** 6-8 weeks for experienced developer
**Complexity:** High (similar to PR #227, but with more scope)
**Recommended:** Yes, aligns with NVIDIA's direction and enables Blackwell support

### Next Steps
1. Review this analysis with project maintainers
2. Create detailed GitHub issue with this plan
3. Set up development environment with CUDA 13 + dual LLVM
4. Begin Phase 1 (Preparation) with prototype

---

## References

- **CUDA 13.0 Release Notes:** https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/
- **NVVM IR Spec 13.0:** https://docs.nvidia.com/cuda/nvvm-ir-spec/
- **libnvvm API 4.0:** https://docs.nvidia.com/cuda/libnvvm-api/
- **PR #227 (LLVM 19):** https://github.com/Rust-GPU/rust-cuda/pull/227
- **Issue #299:** https://github.com/Rust-GPU/rust-cuda/issues/299
- **CUDA Compatibility Guide:** https://docs.nvidia.com/cuda/ada-compatibility-guide/

---

**Document Version:** 1.0
**Last Updated:** 2025-11-11
**Author:** Claude Code Analysis
