# CMakeLists.txt — the GPU server (FIDESlib + OpenFHE). Targets: gpu_real_model (the base harness) and, with
# -DFHE_SSM_EXPERIMENTAL=ON, gpu_real_model_x (the served binary of every recorded session: the same harness with the
# levers of the recorded configuration behind flags). boot_delta_probe (one GPU bootstrap, built on demand) and
# peer_pool_probe (two-card peer-access check used by the bring-up) are instruments.
#
cmake_minimum_required(VERSION 3.25.2)   # FIDESlib requires >= 3.25.2
project(fhe_ssm_server LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)

# Fallback ONLY for manual builds that skip run_hpc_benchmark.sh: 80 = A100
# (the hardware this kit was first exercised on, 2026-07-11). Override for
# other targets, e.g. -DCMAKE_CUDA_ARCHITECTURES=90 for H100.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80)
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ---- dependencies -----------------------------------------------------------
# fideslib::fideslib carries its OpenFHE + CUDA usage requirements
# transitively; OpenFHE is found explicitly too because fideslibConfig may
# find_dependency(OpenFHE).
find_package(OpenFHE CONFIG REQUIRED)
find_package(fideslib REQUIRED CONFIG)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

add_executable(gpu_real_model gpu_real_model.cu)
add_executable(boot_delta_probe boot_delta_probe.cu)
add_executable(peer_pool_probe peer_pool_probe.cu)

# -Xcompiler=-fopenmp: the diagonal-encode loop in matvecBatch is `#pragma omp
# parallel for`. That encode is ~99.7% of the forward pass (gpuBootstrapMs 890
# of 354,398), and it was running single-threaded on a 32-core host. OpenFHE
# itself is already built WITH_OPENMP=ON and libgomp is linked below, but nvcc
# does not forward -fopenmp to the host compiler on its own, so without this
# the pragma is silently IGNORED (no error, no speedup) -- check
# "ompMaxThreads" in the harness header line to confirm it took effect.
target_compile_options(gpu_real_model PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda -Xcompiler=-fopenmp>)

target_link_libraries(gpu_real_model PRIVATE
  fideslib::fideslib
  CUDA::cudart
  gomp                       # same libgomp rationale as above
  Threads::Threads)

set_target_properties(gpu_real_model PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(peer_pool_probe PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>)
target_link_libraries(peer_pool_probe PRIVATE fideslib::fideslib CUDA::cudart gomp Threads::Threads)

target_compile_options(boot_delta_probe PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>)
target_link_libraries(boot_delta_probe PRIVATE fideslib::fideslib CUDA::cudart gomp Threads::Threads)
# 2026-09-03 pod build_fideslib.log:917: the probe includes "openfhe.h" directly,
# and the plain harness configure (rider_build_fideslib.sh step 6, no
# -DCMAKE_CUDA_FLAGS include list) has no OpenFHE include path -> the probe
# failed and took RIDER:harness:ok with it (10_bringup.sh FAIL build). Give the
# probe the include dirs OpenFHE's own config exports (OpenFHE_INCLUDE, the
# layout FIDESlib's DEMO rider passes by hand) and keep it OUT of `all`: it is
# built on demand with `cmake --build <dir> --target boot_delta_probe`, never
# as a gate of the harness build.
if(NOT DEFINED OpenFHE_INCLUDE OR NOT EXISTS "${OpenFHE_INCLUDE}/pke/openfhe.h")
  find_path(OpenFHE_INCLUDE NAMES pke/openfhe.h PATH_SUFFIXES openfhe HINTS ${CMAKE_PREFIX_PATH})
endif()
if(OpenFHE_INCLUDE AND EXISTS "${OpenFHE_INCLUDE}/pke/openfhe.h")
  target_include_directories(boot_delta_probe PRIVATE
    ${OpenFHE_INCLUDE} ${OpenFHE_INCLUDE}/third-party/include ${OpenFHE_INCLUDE}/core
    ${OpenFHE_INCLUDE}/pke ${OpenFHE_INCLUDE}/binfhe)
else()
  message(WARNING "boot_delta_probe: openfhe.h not found (OpenFHE_INCLUDE='${OpenFHE_INCLUDE}'); the probe target will not compile, the harness is unaffected")
endif()
set_target_properties(boot_delta_probe PROPERTIES EXCLUDE_FROM_ALL TRUE)

# ---- run-stamping (provenance) ----------------------------------------------
# Stamps harnessFile / harnessSha256 / harnessCommit / harnessDirty / buildUtc
# / cmdline into every summary record. See run_stamp.h.
include(${CMAKE_CURRENT_LIST_DIR}/RunStamp.cmake)
fhe_ssm_run_stamp(gpu_real_model gpu_real_model.cu)

# ---- the served copy (2026-09-10). -DFHE_SSM_EXPERIMENTAL=ON builds gpu_real_model_x from
# hpc_gpu_port/experimental/gpu_real_model_x.cu with exactly gpu_real_model's options and links, so the pod can
# A/B one unverified lever at a time (each behind its own flag, default OFF; hpc_gpu_port/experimental/README.md).
option(FHE_SSM_EXPERIMENTAL "build gpu_real_model_x from hpc_gpu_port/experimental/ (the recorded configuration's levers)" OFF)
if(FHE_SSM_EXPERIMENTAL)
  add_executable(gpu_real_model_x experimental/gpu_real_model_x.cu)
  # 2026-09-17 pod-fix: the copy includes periodic_encode.hpp / seeded_keys.hpp, which live in hpc_gpu_port/ (one
  # directory up from experimental/): "fatal error: periodic_encode.hpp: No such file" (build_demo.log:403).
  target_include_directories(gpu_real_model_x PRIVATE ${CMAKE_CURRENT_LIST_DIR})
  # 2026-09-17 pod-fix 2: FIDESlib main declares PlaintextImpl(const PlaintextImpl&) in api/Plaintext.hpp but defines
  # only the context constructor (api/Plaintext.cpp) -> the X2 wrapper copy-construct does not link ("undefined
  # reference to fideslib::PlaintextImpl::PlaintextImpl(fideslib::PlaintextImpl const&)", rebuild.log:219). The X2
  # code's documented fallback: one zero encode per MISS as the wrapper (~12 ms, once per key per session; hits
  # from tick 2 on are still free) -- X2 REPORT / gpu_real_model_x.cu "FHE_SSM_X2_ZERO_WRAP".
  target_compile_definitions(gpu_real_model_x PRIVATE FHE_SSM_X2_ZERO_WRAP)
target_compile_options(gpu_real_model_x PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda -Xcompiler=-fopenmp>)

target_link_libraries(gpu_real_model_x PRIVATE
  fideslib::fideslib
  CUDA::cudart
  gomp                       # same libgomp rationale as above
  Threads::Threads)

set_target_properties(gpu_real_model_x PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON)

fhe_ssm_run_stamp(gpu_real_model_x experimental/gpu_real_model_x.cu)
endif()
