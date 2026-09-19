@echo off
rem Build the release package for ninfer-xx90-win.
rem Run from the repository root after build.bat has produced build-win\apps.
rem Produces dist\ninfer-xx90-win-<VER>-win-x64.7z: a single LZMA archive with all
rem executables, runtime DLLs, helper .bat scripts and docs. This script itself is
rem NOT included in the release.
setlocal
set "ROOT=%~dp0"
set "VER=0.1.0"
set "NAME=ninfer-xx90-win-%VER%-win-x64"
set "APPS=%ROOT%build-win\apps"
set "DIST=%ROOT%dist"
set "STAGE=%DIST%\%NAME%"

if not exist "%APPS%\ninfer.exe" (
  echo Missing "%APPS%\ninfer.exe". Build the binaries first.
  exit /b 1
)
if not exist "%APPS%\ninfer-serve.exe" (
  echo Missing "%APPS%\ninfer-serve.exe". Build the binaries first.
  exit /b 1
)
if not exist "%APPS%\ninfer-perplexity.exe" (
  echo Missing "%APPS%\ninfer-perplexity.exe". Build the binaries first.
  exit /b 1
)

rmdir /s /q "%STAGE%" 2>nul
mkdir "%STAGE%"

copy /y "%APPS%\ninfer.exe"            "%STAGE%\" >nul
copy /y "%APPS%\ninfer-serve.exe"      "%STAGE%\" >nul
copy /y "%APPS%\ninfer-perplexity.exe" "%STAGE%\" >nul
copy /y "%APPS%\avcodec-62.dll"        "%STAGE%\" >nul
copy /y "%APPS%\avformat-62.dll"       "%STAGE%\" >nul
copy /y "%APPS%\avutil-60.dll"         "%STAGE%\" >nul
copy /y "%APPS%\swscale-9.dll"         "%STAGE%\" >nul
copy /y "%APPS%\swresample-6.dll"      "%STAGE%\" >nul
copy /y "%APPS%\libcurl-x64.dll"       "%STAGE%\" >nul

set "SWRES=%APPS%\swresample-6.dll"
if not exist "%SWRES%" for /f "delims=" %%F in ('where swresample-6.dll 2^>nul') do if not defined SWRESFOUND (
  set "SWRES=%%F"
  set "SWRESFOUND=1"
)
if not exist "%SWRES%" (
  echo Missing swresample-6.dll - copy it into "%APPS%" once, then re-run.
  exit /b 1
)
copy /y "%SWRES%" "%STAGE%\" >nul

copy /y "%ROOT%README.md"        "%STAGE%\" >nul
copy /y "%ROOT%LICENSE"          "%STAGE%\" >nul
copy /y "%ROOT%USAGE-win64.md"   "%STAGE%\" >nul
copy /y "%ROOT%scripts\download-model.bat"      "%STAGE%\" >nul
copy /y "%ROOT%scripts\run-model-3090-4090.bat" "%STAGE%\" >nul
copy /y "%ROOT%scripts\run-model-5090.bat"      "%STAGE%\" >nul

set "SEVENZIP=C:\Program Files\7-Zip\7z.exe"
if not exist "%SEVENZIP%" for /f "delims=" %%F in ('where 7z 2^>nul') do if not defined SEVENFOUND (
  set "SEVENZIP=%%F"
  set "SEVENFOUND=1"
)

del /q "%DIST%\%NAME%.7z" 2>nul
pushd "%DIST%"
"%SEVENZIP%" a -t7z -mx=9 -mtc=off -mta=off -mtm=off "%NAME%.7z" "%NAME%"
popd
if errorlevel 1 (
  echo 7-Zip compression failed.
  exit /b 1
)

for %%F in ("%DIST%\%NAME%.7z") do set "ARCHIVE_SIZE=%%~zF"
set "PWS=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
"%PWS%" -NoProfile -Command "(Get-FileHash '%DIST%\%NAME%.7z' -Algorithm SHA256).Hash.ToLower() + ' *%NAME%.7z'" > "%DIST%\%NAME%.7z.sha256"
for %%F in ("%DIST%\%NAME%.7z.sha256") do if %%~zF equ 0 (
  echo Failed to write SHA256.
  exit /b 1
)

echo.
echo Release archive: %DIST%\%NAME%.7z  (%ARCHIVE_SIZE% bytes)
echo SHA256 written to %DIST%\%NAME%.7z.sha256
endlocal
