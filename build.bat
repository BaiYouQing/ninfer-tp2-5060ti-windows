@echo off
rem Build the Windows dev tree: MSVC x64, Ninja, Release.
rem
rem Usage: build.bat [--arch 86|89|120a|all|native] [--force|-f]
rem   --arch all     (default) universal binary with native SASS for sm_86, sm_89 and sm_120a.
rem   --arch 86|89|120a       single-architecture development build (faster rebuilds).
rem   --arch native           build for the architectures of the GPUs in this machine.
rem   --force/-f                clean rebuild.
rem Visual Studio is located through vswhere; no install path is hardcoded.
rem The generated ninja files do not pin the LIB environment, so link.exe relies on
rem the MSVC environment for Windows SDK import libraries; vcvars64 is mandatory.
setlocal EnableExtensions

rem Options are matched in the first few positions (positional checks are used
rem instead of `shift` because a shift makes cmd stop resolving %~dp0 to this
rem script's folder).
set "NINFER_FORCE="
if "%~1"=="/force" set "NINFER_FORCE=1"
if "%~1"=="--force" set "NINFER_FORCE=1"
if "%~1"=="-f" set "NINFER_FORCE=1"
if "%~2"=="/force" set "NINFER_FORCE=1"
if "%~2"=="--force" set "NINFER_FORCE=1"
if "%~2"=="-f" set "NINFER_FORCE=1"
if "%~3"=="/force" set "NINFER_FORCE=1"
if "%~3"=="--force" set "NINFER_FORCE=1"
if "%~3"=="-f" set "NINFER_FORCE=1"

set "ARCHREQ="
if "%~1"=="--arch" set "ARCHREQ=%~2"
if "%~2"=="--arch" set "ARCHREQ=%~3"
if "%~3"=="--arch" set "ARCHREQ=%~4"
if not defined ARCHREQ set "ARCHREQ=all"

set "ROOT=%~dp0"
set "ROOTN=%ROOT:~0,-1%"
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VSROOT_FILE=%TEMP%\ninfer_vsroot.txt"

rem Resolve the requested architecture to the CMake list and a dedicated build
rem directory, so switching architectures never leaves a stale cache behind.
if /i "%ARCHREQ%"=="all" goto arch_all
if /i "%ARCHREQ%"=="86" goto arch_86
if /i "%ARCHREQ%"=="89" goto arch_89
if /i "%ARCHREQ%"=="120a" goto arch_120a
if /i "%ARCHREQ%"=="native" goto arch_native
echo Unknown --arch value "%ARCHREQ%". Use 86, 89, 120a, all, or native.
exit /b 1
:arch_all
set "ARCH_LIST=86-real;89-real;120a-real"
set "BUILD_DIR=%ROOT%build-win"
goto arch_ready
:arch_86
set "ARCH_LIST=86-real"
set "BUILD_DIR=%ROOT%build-win-86"
goto arch_ready
:arch_89
set "ARCH_LIST=89-real"
set "BUILD_DIR=%ROOT%build-win-89"
goto arch_ready
:arch_120a
set "ARCH_LIST=120a-real"
set "BUILD_DIR=%ROOT%build-win-120a"
goto arch_ready
:arch_native
rem Build the union of architectures of the GPUs present in this machine.
set "ARCH_LIST="
for /f "usebackq tokens=*" %%a in (`powershell -NoProfile -Command "$d=nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>$null; $l=@(); foreach($c in $d){ switch($c.Trim()){'8.6'{$l+='86-real'} '8.9'{$l+='89-real'} '12.0'{$l+='120a-real'} default{ Write-Host ('Unsupported compute capability '+$c) } }; ($l | Sort-Object -Unique) -join ';'"`) do set "ARCH_LIST=%%a"
if not defined ARCH_LIST goto arch_native_failed
echo Detected native architectures: %ARCH_LIST%
set "BUILD_DIR=%ROOT%build-win-native"
goto arch_ready
:arch_native_failed
echo --arch native found no supported GPU (8.6, 8.9 or 12.0).
exit /b 1
:arch_ready

if exist "%VSWHERE%" goto vswhere_found
echo Missing "%VSWHERE%"
echo Install the Visual Studio 2022 Build Tools with the C++ CMake toolchain workload.
exit /b 1
:vswhere_found

del "%VSROOT_FILE%" 2>nul
"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath >"%VSROOT_FILE%" 2>nul
set "VSROOT="
if exist "%VSROOT_FILE%" set /p VSROOT=<"%VSROOT_FILE%"
del "%VSROOT_FILE%" 2>nul

if defined VSROOT goto vs_found
echo No Visual Studio with the C++ x64 toolset (Microsoft.VisualStudio.Component.VC.Tools.x86.x64) was found by vswhere.
exit /b 1
:vs_found


set "VCVARS=%VSROOT%\VC\Auxiliary\Build\vcvars64.bat"
if exist "%VCVARS%" goto vcvars_found
echo Missing "%VCVARS%"
exit /b 1
:vcvars_found

rem Prefer the CMake shipped with Visual Studio (the build cache was configured
rem with it); fall back to cmake on PATH.
set "CMAKE=%VSROOT%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
if exist "%CMAKE%" goto cmake_found
set "CMAKE=cmake"
:cmake_found

call "%VCVARS%" >nul
if errorlevel 1 goto vcvars_failed
goto env_ready
:vcvars_failed
echo Failed to initialize the MSVC x64 environment from "%VCVARS%".
exit /b 1
:env_ready

echo Visual Studio: "%VSROOT%"
echo Architectures: %ARCH_LIST%
if not exist "%BUILD_DIR%\CMakeCache.txt" goto configure
rem An existing cache must have been configured for the same architecture list.
set "CACHE_ARCH="
for /f "usebackq tokens=2 delims==" %%a in (`findstr /R /C:"^CMAKE_CUDA_ARCHITECTURES:STRING=" "%BUILD_DIR%\CMakeCache.txt" 2^>nul`) do set "CACHE_ARCH=%%a"
if "%CACHE_ARCH%"=="%ARCH_LIST%" goto build
echo Existing %BUILD_DIR% was configured for '%CACHE_ARCH%', not '%ARCH_LIST%'. Reconfiguring...
powershell -NoProfile -Command "Remove-Item -Recurse -Force '%BUILD_DIR%' -ErrorAction SilentlyContinue"
:configure
echo Configuring %BUILD_DIR%: Ninja, Release, %ARCH_LIST%...
"%CMAKE%" -S "%ROOTN%" -B "%BUILD_DIR%" -G Ninja -DCMAKE_BUILD_TYPE=Release "-DCMAKE_CUDA_ARCHITECTURES=%ARCH_LIST%" "-DCMAKE_PREFIX_PATH=%ROOT%.local\deps\ffmpeg_x\ffmpeg-n8.1-latest-win64-gpl-shared-8.1;%ROOT%.local\deps\curl\curl-8.22.0_1-win64-mingw"
if errorlevel 1 goto configure_failed
goto build
:configure_failed
echo Configure failed.
exit /b 1
:build

if not defined NINFER_FORCE goto run_build
rem A clean rebuild deletes %BUILD_DIR%\apps\*.exe. If one of THIS tree's exes is
rem still running it would fail with LNK1104, so check the actual output dir.
rem A ninfer running from a different tree does not hold these files and must
rem not block the build. Defaults to proceeding if powershell is unavailable.
set "LOCKFILE=%TEMP%\ninfer_lock.txt"
del "%LOCKFILE%" 2>nul
powershell -NoProfile -Command "$pat='%BUILD_DIR%\apps\*.exe'; if (Get-CimInstance Win32_Process | Where-Object { $_.ExecutablePath -like $pat }) { 'L' } else { 'O' }" >"%LOCKFILE%" 2>nul
set "LOCK=O"
if exist "%LOCKFILE%" set /p LOCK=<"%LOCKFILE%"
del "%LOCKFILE%" 2>nul
if "%LOCK%"=="L" goto build_locked
goto run_build
:build_locked
echo Cannot force a clean rebuild: a ninfer process from this build tree
echo (%BUILD_DIR%\apps\*.exe) is still running. Close it, then retry.
exit /b 1
:run_build
set "BUILD_FLAGS="
if defined NINFER_FORCE set "BUILD_FLAGS=--clean-first"
"%CMAKE%" --build "%BUILD_DIR%" %BUILD_FLAGS%
if errorlevel 1 goto build_failed
goto done
:build_failed
echo Build failed. See the errors above.
exit /b 1
:done

echo Build complete: %BUILD_DIR%\apps
endlocal
