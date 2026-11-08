param([Parameter(Mandatory=$true)][string]$Base,[Parameter(Mandatory=$true)][string]$Queries,[string]$Output='size_control_results')
$artifactRoot=Split-Path $PSScriptRoot -Parent
& "$artifactRoot/build/deep1m_size_preserving_prd_control.exe" $Base $Queries $Output '' '32,128,256,512' '100'
if ($LASTEXITCODE -ne 0) {throw 'Experiment failed'}
