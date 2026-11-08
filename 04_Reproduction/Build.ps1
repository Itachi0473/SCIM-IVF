$ErrorActionPreference='Stop'
$artifactRoot=Split-Path $PSScriptRoot -Parent
cmake -S "$artifactRoot/01_Source" -B "$artifactRoot/build" -G Ninja -DCMAKE_BUILD_TYPE=Release
if ($LASTEXITCODE -ne 0) {throw 'CMake configure failed'}
cmake --build "$artifactRoot/build" --parallel 2
if ($LASTEXITCODE -ne 0) {throw 'Build failed'}
