cmake_minimum_required(VERSION 3.20)
project(NITA C)  # NITA = NC Inference Tool for All

# ------------------------------------------------------------------
# Default to an optimised build. "-march=..." alone leaves GCC at -O0,
# which makes the scalar dequant/matvec kernels an order of magnitude
# slower than they should be.
# ------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
                 Debug Release RelWithDebInfo MinSizeRel)
endif()

# ------------------------------------------------------------------
# Target architecture.
#
# Historically this hardcoded `-march=znver3 -mavx2`, which produced a binary
# that faults with SIGILL on any CPU without AVX2. The code is written so all
# AVX2 intrinsics sit behind __AVX2__ guards with scalar fallbacks, so a plain
# x86-64 build is correct (and slower). Detection order on x86:
#   znver3+AVX2 (Zen 3+)  ->  any -mavx2-capable target  ->  generic baseline
# Override with -DNITA_ARCH=znver3|avx2|none (none = generic x86-64, portable).
# Note: this is build-time selection, not runtime dispatch; an AVX2 binary
# still faults on a pre-AVX2 CPU. Runtime dispatch remains a follow-up.
# ------------------------------------------------------------------
if(NOT NITA_ARCH)
    set(NITA_ARCH "auto" CACHE STRING "Architecture: auto|znver3|avx2|none" FORCE)
endif()
include(CheckCCompilerFlag)
set(_nita_arch_flags "")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86|AMD|i[3-6]86")
    if(NITA_ARCH STREQUAL "znver3")
        set(_nita_arch_flags "-march=znver3")
    elseif(NITA_ARCH STREQUAL "avx2")
        set(_nita_arch_flags "-mavx2")
    elseif(NITA_ARCH STREQUAL "none")
        set(_nita_arch_flags "")
    else() # auto
        check_c_compiler_flag("-march=znver3" _nita_znver3_ok)
        if(_nita_znver3_ok)
            set(_nita_arch_flags "-march=znver3")  # znver3 implies AVX2
        else()
            check_c_compiler_flag("-mavx2" _nita_avx2_ok)
            if(_nita_avx2_ok)
                set(_nita_arch_flags "-mavx2")
            endif()
        endif()
    endif()
else()
    message(STATUS "NITA: non-x86 processor (${CMAKE_SYSTEM_PROCESSOR}); building baseline, AVX2 paths disabled")
endif()
if(_nita_arch_flags)
    add_compile_options(${_nita_arch_flags})
    message(STATUS "NITA arch flags: ${_nita_arch_flags} (set -DNITA_ARCH to override)")
else()
    message(STATUS "NITA arch flags: baseline (no AVX2)")
endif()

# ------------------------------------------------------------------
# Quantisation feature toggles (CMake -D flags)
# ------------------------------------------------------------------
option(QUANTIZED_GEMV "Enable quantized GEMV base path" ON)
option(TERNARY "Enable ternary {-1,0,1} quantisation" OFF)
option(GPTQ "Enable 4‑bit grouped GPTQ with selective de‑quant" OFF)
option(AWQ "Enable activation‑aware weight quantisation (selective)" OFF)
option(BINARY_NET "Enable BitNet / binary quantisation" OFF)
option(BF16 "Enable brain‑float16 float matmul path" OFF)
option(FP16 "Enable IEEE float16 matmul path" OFF)
option(MXFP8 "Report the MXFP8 preset label only (no kernel exists; plain Q4_K_M dot path is used)" OFF)
option(DEBUG_BREADCRUMBS "Enable breadcrumb tracing to stderr" OFF)
option(SHERLOCK_LITE "Enable sherlock‑lite trace to stdout" OFF)

# ------------------------------------------------------------------
# Derive a single QUANT_TYPE macro from the enabled flag.
# Exactly one should be ON; if none is ON, default to Q4_K_M.
# The logic: check each flag in priority order; the first TRUE wins.
# ------------------------------------------------------------------
if(TERNARY)
    set(QUANT_TYPE -DQUANT_TYPE=TERNARY)
elseif(GPTQ)
    set(QUANT_TYPE -DQUANT_TYPE=GPTQ)
elseif(AWQ)
    set(QUANT_TYPE -DQUANT_TYPE=AWQ)
elseif(BINARY_NET)
    set(QUANT_TYPE -DQUANT_TYPE=BINARY_NET)
elseif(BF16)
    set(QUANT_TYPE -DQUANT_TYPE=BF16)
elseif(FP16)
    set(QUANT_TYPE -DQUANT_TYPE=FP16)
elseif(MXFP8)
    set(QUANT_TYPE -DQUANT_TYPE=MXFP8)
else()
    # Default: Q4_K_M when no other flag is given
    set(QUANT_TYPE -DQUANT_TYPE=Q4_K_M)
endif()

# ------------------------------------------------------------------
# Force the preprocessor macros from CMake options
# ------------------------------------------------------------------
if(QUANTIZED_GEMV)
    add_compile_options(-DQUANTIZED_GEMV)
endif()
if(TERNARY)
    add_compile_options(-DTERNARY)
endif()
if(GPTQ)
    add_compile_options(-DGPTQ)
endif()
if(AWQ)
    add_compile_options(-DAWQ)
endif()
if(BINARY_NET)
    add_compile_options(-DBINARY_NET)
endif()
if(BF16)
    add_compile_options(-DBF16)
endif()
if(FP16)
    add_compile_options(-DFP16)
endif()
if(MXFP8)
    add_compile_options(-DMXFP8)
endif()
if(DEBUG_BREADCRUMBS)
    add_compile_options(-DDEBUG_BREADCRUMBS)
endif()
if(SHERLOCK_LITE)
    add_compile_options(-DSHERLOCK_LITE)
endif()

# Add the derived QUANT_TYPE macro last (overrides any previous)
add_compile_options(${QUANT_TYPE})

# ------------------------------------------------------------------
# Source files – everything but main(), so the tests can link the same
# objects instead of repeating this list (and its flags) per test.
# ------------------------------------------------------------------
add_library(nita_core STATIC
    src/util.c
    src/gguf.c
    src/json.c
    src/tokenizer.c
    src/chat.c
    src/http.c
    src/dequant.c
    src/quant.c
    src/transformer.c
    src/qwen35.c
    src/hadamard.c
)

target_include_directories(nita_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

# libm for expf/cosf/powf/sqrtf.
if(NOT MSVC)
    target_link_libraries(nita_core PUBLIC m)
endif()

# Winsock is required by the HTTP server on Windows.
if(WIN32)
    target_link_libraries(nita_core PUBLIC ws2_32)
endif()

add_executable(nita src/main.c)
target_link_libraries(nita PRIVATE nita_core)

# Keep the historical binary name working for existing scripts.
if(WIN32)
    add_custom_command(TARGET nita POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:nita> $<TARGET_FILE_DIR:nita>/minimal_engine.exe)
else()
    add_custom_command(TARGET nita POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:nita> $<TARGET_FILE_DIR:nita>/minimal_engine)
endif()

# ------------------------------------------------------------------
# Optional: the reference logits driver
#
# tools/llama_logits.c prints llama.cpp's own logits, which is what the logits
# in the README were checked against. It links the PrismML fork, which is not
# part of this repository, so it is built only when a checkout is present.
# "Configured" means build-mingw exists: that step is the fork's own advice
# (see README) and involves a local patch, so it stays a manual step.
# ------------------------------------------------------------------
set(NITA_FORK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/build/prism-llama" CACHE PATH
    "PrismML llama.cpp checkout with a configured build-mingw directory")

set(NITA_FORK_DRIVER "")
set(_fork_libs
    "${NITA_FORK_DIR}/build-mingw/src/libllama.a"
    "${NITA_FORK_DIR}/build-mingw/ggml/src/ggml.a"
    "${NITA_FORK_DIR}/build-mingw/ggml/src/ggml-cpu.a"
    "${NITA_FORK_DIR}/build-mingw/ggml/src/ggml-base.a"
    "${NITA_FORK_DIR}/build-mingw/vendor/hash/libvendor-hash.a")

if(NITA_FORK_DIR AND EXISTS "${NITA_FORK_DIR}/include/llama.h")
    set(_fork_missing "")
    foreach(_lib ${_fork_libs})
        if(NOT EXISTS "${_lib}")
            list(APPEND _fork_missing ${_lib})
        endif()
    endforeach()

    if(_fork_missing)
        message(STATUS "fork logits driver: skipped, missing:")
        foreach(_lib ${_fork_missing})
            message(STATUS "  ${_lib}")
        endforeach()
    else()
        add_executable(fork_logits tools/llama_logits.c)
        target_include_directories(fork_logits PRIVATE
            "${NITA_FORK_DIR}/include"
            "${NITA_FORK_DIR}/ggml/include")
        if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
            # Opt this one target out of the project's -march=znver3/-mavx2.
            # With AVX enabled GCC kept the driver's struct arguments and
            # buffers as 32-byte stack slots addressed with vmovdqa, which
            # demands 32-byte alignment; Windows only guarantees 16 at main's
            # entry, so the first such store segfaulted whenever ASLR placed
            # the stack at 16 mod 32 - measured 7/20 runs with AVX against
            # 0/24 without, and the faulting instruction was
            # vmovdqa %ymm3,0x150(%rsp) in main. The driver only parses argv
            # and prints numbers, so baseline x86-64 costs it nothing, and the
            # fork's own AVX code lives in libraries built separately.
            target_compile_options(fork_logits PRIVATE -mno-avx -mno-avx2)
        endif()
        target_link_libraries(fork_logits PRIVATE ${_fork_libs})
        if(WIN32)
            target_link_libraries(fork_logits PRIVATE
                ws2_32 ole32 uuid advapi32 shlwapi)
        endif()
        if(MINGW)
            # llama.cpp is C++, this driver is C: gcc would not pull the C++
            # runtime in on its own.
            target_link_libraries(fork_logits PRIVATE stdc++)
            # The toolchain's bin is not necessarily on PATH where CTest starts
            # the process - a missing runtime DLL kills it with no output at
            # all - so put the DLLs next to the executable.
            # The DLLs sit in the toolchain's bin, which -print-file-name does
            # not search (it only reports its library directories).
            get_filename_component(_mingw_bin "${CMAKE_C_COMPILER}" DIRECTORY)
            foreach(_dll libstdc++-6.dll libgcc_s_seh-1.dll libwinpthread-1.dll)
                if(EXISTS "${_mingw_bin}/${_dll}")
                    add_custom_command(TARGET fork_logits POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                                "${_mingw_bin}/${_dll}"
                                "$<TARGET_FILE_DIR:fork_logits>")
                endif()
            endforeach()
        endif()
        set(NITA_FORK_DRIVER $<TARGET_FILE:fork_logits>)
        message(STATUS "fork logits driver: fork_logits, linking ${NITA_FORK_DIR}")
    endif()
endif()

# ------------------------------------------------------------------
# Tests
#
#   cmake --build build && ctest --test-dir build
#
# hadamard and q35_heads need no checkpoint, so they always register. The ones
# that read a model file are built unconditionally and registered once per
# checkpoint named in -DNITA_TEST_MODELS=<path>[;<path>...]. A list rather than
# one path because the two Bonsai checkpoints differ in quantisation type
# (PTQ1_0 against Q1_0), block count and head geometry, and a run that loads
# both covers both without a re-configure in between. Each checkpoint's tests
# carry its file stem, so the names stay unique and `ctest -R fork_match` still
# selects every checkpoint's comparison. tests/model_test.c and
# tests/kernel_test.c (RED-phase scaffolds for an unbuilt placeholder
# kernel) were removed in 2026-11; coverage of the quantised path is
# quant_dot@<tag> and quant_tile@<tag>.
# ------------------------------------------------------------------
include(CTest)
set(NITA_TEST_MODELS "" CACHE STRING
    "Bonsai checkpoints for the checkpoint tests; semicolon-separated list")
set(NITA_TEST_MODEL "" CACHE FILEPATH
    "Single-checkpoint alias for NITA_TEST_MODELS")
if(NITA_TEST_MODEL)
    if(NITA_TEST_MODELS)
        message(WARNING
            "checkpoint tests: NITA_TEST_MODEL is ignored because "
            "NITA_TEST_MODELS is set")
    else()
        set(NITA_TEST_MODELS "${NITA_TEST_MODEL}")
    endif()
endif()

add_executable(hadamard_test tests/hadamard_test.c)
target_link_libraries(hadamard_test PRIVATE nita_core)
add_test(NAME hadamard COMMAND hadamard_test)

add_executable(q35_heads_test tests/q35_heads_test.c)
target_link_libraries(q35_heads_test PRIVATE nita_core)
add_test(NAME q35_heads COMMAND q35_heads_test)

add_executable(q35_probe tests/q35_probe.c)
target_link_libraries(q35_probe PRIVATE nita_core)

add_executable(quant_dot_test tests/quant_dot_test.c)
target_link_libraries(quant_dot_test PRIVATE nita_core)

# Comparing against the reference runtime needs two things this repository does
# not build: the PrismML llama.cpp and the driver linked against it, which is
# tools/llama_logits.c (see README, "The reference harness"). Name the driver
# with -DNITA_FORK_LOGITS and the check joins the suite; the script exits 77
# when a piece is missing, which CTest reports as a skip rather than a pass.
set(NITA_FORK_LOGITS "" CACHE FILEPATH
    "logits driver linked against the PrismML fork; empty to use fork_logits")

# A full fork comparison costs roughly the whole forward pass on both sides:
# four seconds on the 2B but 82 on the 27B, which is long enough to put a plain
# `ctest` off a quick loop. So on a checkpoint over a gigabyte it is registered
# only with -DNITA_TEST_SLOW=ON; every other entry stays, since q35_probe and
# quant_dot on the 27B take about a second each and are the only cover the
# Q1_0 path gets. The configure step names anything it deferred, so the fast
# run is never quietly the whole story.
set(NITA_TEST_SLOW OFF CACHE BOOL
    "Register fork_match on checkpoints over 1 GiB too (adds about 80 s)")
set(_nita_slow_bytes 1073741824)

find_package(Python3 COMPONENTS Interpreter QUIET)

# A driver named by hand wins over the one built here, so a checkout that lives
# somewhere else can still be used.
set(_fork_test_driver "${NITA_FORK_LOGITS}")
if(NOT _fork_test_driver)
    set(_fork_test_driver "${NITA_FORK_DRIVER}")
endif()

set(_registered_tags "")
foreach(_model ${NITA_TEST_MODELS})
    if(NOT EXISTS "${_model}")
        message(STATUS "checkpoint tests: no file at ${_model}, skipped")
        continue()
    endif()

    get_filename_component(_stem "${_model}" NAME_WE)
    set(_tag "${_stem}")
    set(_n 1)
    while(_tag IN_LIST _registered_tags)   # same filename in two directories
        math(EXPR _n "${_n} + 1")
        set(_tag "${_stem}#${_n}")
    endwhile()
    list(APPEND _registered_tags "${_tag}")

    file(SIZE "${_model}" _model_bytes)
    if(_model_bytes GREATER _nita_slow_bytes)
        set(_model_slow ON)
    else()
        set(_model_slow OFF)
    endif()

    add_test(NAME q35_probe@${_tag} COMMAND q35_probe "${_model}")
    # --all walks the tensor table and tests every quantised type the file
    # stores, so this entry does not have to name a type id that could drift.
    add_test(NAME quant_dot@${_tag} COMMAND quant_dot_test --all "${_model}")
    # The sequential-vs-fused arbiter: catches a tile-kernel layout mistake
    # (the Q4_K AVX2 bug) that the float reference cannot see.
    add_test(NAME quant_tile@${_tag} COMMAND quant_dot_test --tile "${_model}")
    set_tests_properties(q35_probe@${_tag} quant_dot@${_tag} quant_tile@${_tag}
                         PROPERTIES LABELS "checkpoint")

    # The engine's own reader cannot vouch for an offset it read wrongly: a
    # shifted tensor moves the expected and the actual together. This walks the
    # header with struct alone and requires every tensor's declared geometry to
    # account for the data section, so it is the entry that catches a bad read.
    if(Python3_Interpreter_FOUND)
        add_test(NAME gguf_probe@${_tag}
                 COMMAND ${Python3_EXECUTABLE}
                         ${CMAKE_CURRENT_SOURCE_DIR}/tools/gguf_probe.py
                         --engine $<TARGET_FILE:nita> "${_model}")
        set_tests_properties(gguf_probe@${_tag} PROPERTIES LABELS "checkpoint")

        # Every weight the loaded model holds, against the file: the reader
        # check says the table is right, this says the bindings built from it
        # point at the tensors they claim, layer by layer. It loads the model at
        # context 1, so the KV cache of a 64-layer file never comes into play.
        add_test(NAME model_bind@${_tag}
                 COMMAND ${Python3_EXECUTABLE}
                         ${CMAKE_CURRENT_SOURCE_DIR}/tools/gguf_probe.py
                         --engine $<TARGET_FILE:nita> --model-bind "${_model}")
        set_tests_properties(model_bind@${_tag} PROPERTIES LABELS "checkpoint")
    endif()

    if(_fork_test_driver AND Python3_Interpreter_FOUND)
        if(_model_slow AND NOT NITA_TEST_SLOW)
            message(STATUS
                "checkpoint tests: fork_match@${_tag} deferred (large file, "
                "add -DNITA_TEST_SLOW=ON to register it)")
        else()
            add_test(NAME fork_match@${_tag}
                     COMMAND ${Python3_EXECUTABLE}
                             ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_fork_match.py
                             --engine $<TARGET_FILE:nita>
                             --fork ${_fork_test_driver}
                             --model "${_model}")
            set_tests_properties(fork_match@${_tag}
                                 PROPERTIES TIMEOUT 900 SKIP_RETURN_CODE 77
                                            LABELS "checkpoint")
            if(_model_slow)
                set_tests_properties(fork_match@${_tag}
                                     PROPERTIES LABELS "checkpoint;slow")
            endif()
        endif()
    endif()
endforeach()
if(NOT _registered_tags)
    message(STATUS "checkpoint tests: none, set -DNITA_TEST_MODELS=<a.gguf>[;...]")
endif()

# The tokenizer has no C program behind it, so its cover is the independent
# encoder pair in tools/reference_tokenizer.py over a generated vocabulary. The
# packs are a setup fixture, so `ctest -R tokenizer` brings them along even when
# that is the only entry selected; they are written into the build directory,
# which is where the manual commands in the README look for them.
if(Python3_Interpreter_FOUND)
    add_test(NAME tokenizer_vocab
             COMMAND ${Python3_EXECUTABLE}
                     ${CMAKE_CURRENT_SOURCE_DIR}/tools/make_tokenizer_gguf.py
                     ${CMAKE_CURRENT_BINARY_DIR})
    set_tests_properties(tokenizer_vocab PROPERTIES
                         FIXTURES_SETUP tokpacks LABELS "tokenizer")
    foreach(_scheme spm bpe)
        add_test(NAME tokenizer@${_scheme}
                 COMMAND ${Python3_EXECUTABLE}
                         ${CMAKE_CURRENT_SOURCE_DIR}/tools/reference_tokenizer.py
                         --gguf ${CMAKE_CURRENT_BINARY_DIR}/tok_${_scheme}.gguf
                         --json ${CMAKE_CURRENT_BINARY_DIR}/tok_${_scheme}.gguf.json
                         --exe $<TARGET_FILE:nita>)
        set_tests_properties(tokenizer@${_scheme} PROPERTIES
                             FIXTURES_REQUIRED tokpacks
                             LABELS "tokenizer")
    endforeach()
else()
    message(STATUS
        "tool checks: no Python 3 interpreter, gguf_probe and tokenizer skipped")
endif()

message(STATUS "Project: NITA (NC Inference Tool for All)")
message(STATUS "Engine flags:")
message(STATUS "  QUANTIZED_GEMV=${QUANTIZED_GEMV}")
message(STATUS "  TERNARY=${TERNARY}")
message(STATUS "  GPTQ=${GPTQ}")
message(STATUS "  AWQ=${AWQ}")
message(STATUS "  BINARY_NET=${BINARY_NET}")
message(STATUS "  BF16=${BF16}")
message(STATUS "  FP16=${FP16}")
message(STATUS "  MXFP8=${MXFP8}")
message(STATUS "  QUANT_TYPE=${QUANT_TYPE}")
