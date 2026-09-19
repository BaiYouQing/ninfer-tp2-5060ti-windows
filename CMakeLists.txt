cmake_minimum_required(VERSION 3.28)

# The current implementation is compiled only for sm_120a. Set and validate the
# architecture before CUDA compiler detection so CMake never injects a generic
# host-toolchain default or accepts an unsupported target.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120a CACHE STRING "CUDA architectures to build")
endif()
# Windows port (ported from ninfer-xx90-win): build.bat passes the SASS-only
# spelling "120a-real"; accept it as an alias of "120a".
string(REPLACE "120a-real" "120a" NINFER_ARCH_CHECKED "${CMAKE_CUDA_ARCHITECTURES}")
if(NOT NINFER_ARCH_CHECKED STREQUAL "120a")
  message(FATAL_ERROR
    "NInfer supports only CMAKE_CUDA_ARCHITECTURES=120a (or 120a-real); got "
    "'${CMAKE_CUDA_ARCHITECTURES}'")
endif()

project(ninfer LANGUAGES C CXX CUDA)

# --- Global settings ---------------------------------------------------------
option(NINFER_BUILD_APPS "Build the NInfer CLI and server" ON)
option(BUILD_TESTING "Build NInfer tests" OFF)
option(NINFER_BUILD_BENCHMARKS "Build NInfer benchmarks" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)        # clangd / IDE tooling
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_GENERATOR MATCHES "Ninja")
  set_property(GLOBAL PROPERTY JOB_POOLS cuda_link=1)
  set(CMAKE_JOB_POOL_LINK cuda_link)
endif()

if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.0)
  message(FATAL_ERROR
    "NInfer requires CUDA 13.0 or newer (fork downshifted from 13.1); found CUDA compiler "
    "${CMAKE_CUDA_COMPILER_VERSION}")
endif()

# --- Windows host toolchain (ported from ninfer-xx90-win) --------------------
# The Windows host toolchain needs explicit, uniform settings for every language:
# UTF-8 source and execution charset, object files large enough for the
# template-heavy kernel translations, and the current conformance switches. CUDA
# device compilation forwards the host-only flags to the MSVC host compiler.
if(MSVC)
  add_compile_options(
    $<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>
    $<$<COMPILE_LANGUAGE:C,CXX>:/bigobj>
    $<$<COMPILE_LANGUAGE:C,CXX>:/Zc:__cplusplus>
    $<$<COMPILE_LANGUAGE:C,CXX>:/Zc:preprocessor>
    $<$<COMPILE_LANGUAGE:C,CXX>:/permissive->
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8,/bigobj,/Zc:__cplusplus,/Zc:preprocessor>)
  add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
  # The static CUDA runtime and the CUDA host pass must not each pull in their own
  # CRT import library; dropping the default libcmt keeps one consistent CRT.
  add_link_options(/NODEFAULTLIB:LIBCMT)
endif()

set(NINFER_BUILD_MEDIA_ACQUIRE OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_MEDIA_ACQUIRE ON)
endif()

set(NINFER_BUILD_PROMPT_INPUT OFF)
if(NINFER_BUILD_APPS)
  set(NINFER_BUILD_PROMPT_INPUT ON)
endif()

set(NINFER_BUILD_SERVE OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_SERVE ON)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# Windows ships a static CUDA runtime; link it so the product is self-sufficient
# and needs neither cudart64_*.dll nor an installed CUDA Toolkit on PATH. Other
# platforms use the dynamic runtime.
if(WIN32)
  set(NINFER_CUDART_TARGET CUDA::cudart_static)
else()
  set(NINFER_CUDART_TARGET CUDA::cudart)
endif()

# Locates the shared library that backs a located dependency and appends it to
# the list named by ${out_dlls}. A located shared library is used directly; an
# import library is paired with the payload in the adjacent bin directory. A
# dependency without a payload is a static link and adds nothing to the list.
function(ninfer_resolve_runtime_dll dependency import_library out_dlls)
  set(resolved "")
  if(import_library MATCHES "\\.dll$")
    set(resolved "${import_library}")
  else()
    get_filename_component(library_directory "${import_library}" DIRECTORY)
    get_filename_component(dependency_prefix "${library_directory}" DIRECTORY)
    file(GLOB candidates LIST_DIRECTORIES FALSE "${dependency_prefix}/bin/${dependency}*.dll")
    list(LENGTH candidates candidate_count)
    if(candidate_count GREATER 0)
      list(GET candidates 0 resolved)
    endif()
  endif()
  if(resolved STREQUAL "")
    message(STATUS "NInfer: no shared runtime payload for ${dependency}; assuming a static link")
    return()
  endif()
  set(${out_dlls} "${${out_dlls}};${resolved}" PARENT_SCOPE)
endfunction()

# Copies the collected dependency payloads next to ${target} so that it resolves
# them from its own directory. A POSIX build resolves the same libraries through
# the system loader and needs nothing.
function(ninfer_copy_runtime_dlls target)
  if(WIN32 AND NINFER_RUNTIME_DLLS)
    add_custom_command(TARGET ${target} POST_BUILD
      COMMAND "${CMAKE_COMMAND}" -E copy_if_different ${NINFER_RUNTIME_DLLS}
              "$<TARGET_FILE_DIR:${target}>")
  endif()
endfunction()

# FFmpeg and libcurl are located once here and re-exported as ninfer_ffmpeg and
# ninfer_curl. A POSIX toolchain resolves them through the pkg-config database; a
# Windows toolchain has no pkg-config database, so the same development packages
# are resolved from CMAKE_PREFIX_PATH (see .local/deps) or the cache variables.
if(WIN32)
  find_path(NINFER_FFMPEG_INCLUDE_DIR NAMES libavformat/avformat.h)
  find_library(NINFER_AVFORMAT_LIBRARY NAMES avformat)
  find_library(NINFER_AVCODEC_LIBRARY NAMES avcodec)
  find_library(NINFER_AVUTIL_LIBRARY NAMES avutil)
  find_library(NINFER_SWSCALE_LIBRARY NAMES swscale)
  foreach(dependency NINFER_FFMPEG_INCLUDE_DIR NINFER_AVFORMAT_LIBRARY NINFER_AVCODEC_LIBRARY
                     NINFER_AVUTIL_LIBRARY NINFER_SWSCALE_LIBRARY)
    if(NOT ${dependency})
      message(FATAL_ERROR
        "NInfer requires the FFmpeg development packages libavformat, libavcodec, libavutil, and "
        "swscale; ${dependency} was not found. Point CMAKE_PREFIX_PATH at the installed FFmpeg "
        "tree.")
    endif()
  endforeach()
  add_library(ninfer_ffmpeg INTERFACE)
  target_include_directories(ninfer_ffmpeg INTERFACE ${NINFER_FFMPEG_INCLUDE_DIR})
  target_link_libraries(ninfer_ffmpeg
    INTERFACE
      ${NINFER_AVFORMAT_LIBRARY}
      ${NINFER_AVCODEC_LIBRARY}
      ${NINFER_AVUTIL_LIBRARY}
      ${NINFER_SWSCALE_LIBRARY})

  if(NINFER_BUILD_MEDIA_ACQUIRE)
    find_path(NINFER_CURL_INCLUDE_DIR NAMES curl/curl.h)
    find_library(NINFER_CURL_LIBRARY NAMES libcurl libcurl_imp libcurl_static)
    if(NOT NINFER_CURL_INCLUDE_DIR OR NOT NINFER_CURL_LIBRARY)
      message(FATAL_ERROR
        "NInfer requires the libcurl development package (headers and an import library). Point "
        "CMAKE_PREFIX_PATH at the installed libcurl tree.")
    endif()
    add_library(ninfer_curl INTERFACE)
    target_include_directories(ninfer_curl INTERFACE ${NINFER_CURL_INCLUDE_DIR})
    target_link_libraries(ninfer_curl INTERFACE ${NINFER_CURL_LIBRARY} ws2_32)
  endif()

  # A Windows process loads a shared library from its own directory before it
  # searches PATH, so the payload of every dependency resolved above is collected
  # and copied next to each executable at build time.
  set(NINFER_RUNTIME_DLLS "")
  ninfer_resolve_runtime_dll(avformat "${NINFER_AVFORMAT_LIBRARY}" NINFER_RUNTIME_DLLS)
  ninfer_resolve_runtime_dll(avcodec "${NINFER_AVCODEC_LIBRARY}" NINFER_RUNTIME_DLLS)
  ninfer_resolve_runtime_dll(avutil "${NINFER_AVUTIL_LIBRARY}" NINFER_RUNTIME_DLLS)
  ninfer_resolve_runtime_dll(swscale "${NINFER_SWSCALE_LIBRARY}" NINFER_RUNTIME_DLLS)
  if(NINFER_BUILD_MEDIA_ACQUIRE)
    ninfer_resolve_runtime_dll(libcurl "${NINFER_CURL_LIBRARY}" NINFER_RUNTIME_DLLS)
  endif()
else()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
    libavformat>=60 libavcodec>=60 libavutil>=58 libswscale>=7)
  add_library(ninfer_ffmpeg INTERFACE)
  target_link_libraries(ninfer_ffmpeg INTERFACE PkgConfig::FFMPEG)
  if(NINFER_BUILD_MEDIA_ACQUIRE)
    pkg_check_modules(LIBCURL REQUIRED IMPORTED_TARGET libcurl>=7.85)
    add_library(ninfer_curl INTERFACE)
    target_link_libraries(ninfer_curl INTERFACE PkgConfig::LIBCURL)
  endif()
endif()

# --- Subdirectories ----------------------------------------------------------
add_subdirectory(src)
if(NINFER_BUILD_APPS)
  add_subdirectory(apps)
endif()
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
if(NINFER_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()
