cmake_minimum_required(VERSION 3.16)
project(fhe_ssm_cpu CXX)

# The OpenFHE (CPU) twin of the served circuit and the probes whose outputs are cited in the reports:
#   cpu_real_model         the CPU twin of gpu_real_model (same circuit, small rings for tests)
#   verify_real_fhe        end-to-end check of the twin against the plaintext model
#   state_carry_probe      the canonical carry of the recurrent state across ticks
#   periodic_encode_probe  the periodic encoder for lane-replicated plaintexts vs the dense encode
#   boot_overflow_probe    planted bootstrap-input overflow sweeps (the extended bootstrap range decision)
#   overflow_dist_dump     the coefficient-overflow distribution of real bootstrap inputs (needs the secret key: test keys only)
#   overflow_tail_probe(2) the tail of that distribution
#   logqp_probe            the modulus chain (logQP) the server's context policy produces, against the library's ceiling
#   c11_sparse_repro       reproducer of the OpenFHE sparse-bootstrapping decoding-budget bug (openfhe-development#1264)
#   boot_envelope_probe, boot_floor_bench, chain_precision_probe, limb_probe   error and level-chain measurements
# The experimental copy of the twin (hpc_gpu_port/experimental/cpu_real_model_x.cpp) is built the same way from that directory.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenFHE CONFIG REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenFHE_CXX_FLAGS}")
include_directories(${OpenFHE_INCLUDE})
include_directories(${OpenFHE_INCLUDE}/third-party/include)
include_directories(${OpenFHE_INCLUDE}/core)
include_directories(${OpenFHE_INCLUDE}/pke)
include_directories(${OpenFHE_INCLUDE}/binfhe)
link_directories(${OpenFHE_LIBDIR})

# run-stamping (provenance): one shared module + header, kept next to the GPU harness so the CPU probes and the GPU
# harness stamp identically (fields and rationale: ../hpc_gpu_port/run_stamp.h).
include(${CMAKE_CURRENT_SOURCE_DIR}/../hpc_gpu_port/RunStamp.cmake)

foreach(tgt cpu_real_model verify_real_fhe state_carry_probe periodic_encode_probe boot_overflow_probe overflow_dist_dump
            overflow_tail_probe overflow_tail_probe2 logqp_probe c11_sparse_repro boot_envelope_probe boot_floor_bench
            chain_precision_probe limb_probe)
    add_executable(${tgt} ${tgt}.cpp)
    target_link_libraries(${tgt} ${OpenFHE_SHARED_LIBRARIES})
    target_include_directories(${tgt} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../hpc_gpu_port)
    fhe_ssm_run_stamp(${tgt} ${tgt}.cpp)
    set_target_properties(${tgt} PROPERTIES
        BUILD_RPATH "${OpenFHE_LIBDIR}"
        INSTALL_RPATH "${OpenFHE_LIBDIR}")
endforeach()
