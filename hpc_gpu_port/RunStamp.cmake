# RunStamp.cmake — inject the provenance stamp into a harness target.
#
# Usage (after add_executable):
#     include(${CMAKE_CURRENT_LIST_DIR}/RunStamp.cmake)
#     fhe_ssm_run_stamp(gpu_real_model gpu_real_model.cu)
#
# Rationale and field meanings: hpc_gpu_port/run_stamp.h.
#
# Everything is resolved at CONFIGURE time, which is the only time both the
# source tree and (maybe) a .git dir are guaranteed present. On a pod
# /root/src is an rsync'd copy with no .git, so the git fields degrade to
# "unknown"/dirty and the SHA-256 of the source file carries the provenance.
# That degradation is deliberate: an honest "unknown" beats a stale SHA.
#
# NOTE: a re-configure is required to refresh the stamp after editing a
# harness source. `fhe_ssm_run_stamp` adds a CONFIGURE_DEPENDS-style
# dependency on the source file so `cmake --build` re-runs cmake when the
# stamped source changes; without it a rebuilt binary would carry the
# previous file's hash, which is exactly the failure this is meant to prevent.

function(fhe_ssm_run_stamp TARGET SOURCE_FILE)
  set(_src "${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE}")

  # --- source-file identity (always available) --------------------------
  get_filename_component(_base "${_src}" NAME)
  if(EXISTS "${_src}")
    file(SHA256 "${_src}" _full_sha)
    string(SUBSTRING "${_full_sha}" 0 16 _sha)
  else()
    set(_sha "unknown")
    message(WARNING "fhe_ssm_run_stamp: ${_src} not found; stamping harnessSha256=unknown")
  endif()

  # Re-run cmake when the stamped source changes, so the hash never goes stale.
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_src}")

  # --- git identity (best effort) ---------------------------------------
  # Allow an explicit override for builds from an rsync'd tree:
  #     cmake ... -DFHE_SSM_GIT_SHA=b9cddd7 -DFHE_SSM_GIT_DIRTY=0
  if(DEFINED FHE_SSM_GIT_SHA)
    set(_gitsha "${FHE_SSM_GIT_SHA}")
    if(DEFINED FHE_SSM_GIT_DIRTY)
      set(_dirty "${FHE_SSM_GIT_DIRTY}")
    else()
      set(_dirty 0)
    endif()
  else()
    set(_gitsha "unknown")
    set(_dirty 1)
    find_package(Git QUIET)
    if(GIT_FOUND)
      execute_process(
        COMMAND "${GIT_EXECUTABLE}" rev-parse --short=12 HEAD
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        OUTPUT_VARIABLE _out RESULT_VARIABLE _rc
        OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
      if(_rc EQUAL 0 AND _out)
        set(_gitsha "${_out}")
        execute_process(
          COMMAND "${GIT_EXECUTABLE}" status --porcelain --untracked-files=no
          WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
          OUTPUT_VARIABLE _st RESULT_VARIABLE _rc2
          OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        if(_rc2 EQUAL 0 AND _st STREQUAL "")
          set(_dirty 0)
        endif()
      endif()
    endif()
  endif()

  string(TIMESTAMP _utc "%Y-%m-%dT%H:%M:%SZ" UTC)

  target_compile_definitions(${TARGET} PRIVATE
    FHE_SSM_HARNESS_FILE="${_base}"
    FHE_SSM_HARNESS_SHA256="${_sha}"
    FHE_SSM_GIT_SHA="${_gitsha}"
    FHE_SSM_GIT_DIRTY=${_dirty}
    FHE_SSM_BUILD_UTC="${_utc}")

  message(STATUS "run-stamp ${TARGET}: file=${_base} sha256=${_sha} commit=${_gitsha} dirty=${_dirty}")
endfunction()
