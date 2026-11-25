$ErrorActionPreference = 'Stop'
$logDirectory = Join-Path $PSScriptRoot 'logs'
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
New-Item -ItemType File -Path (Join-Path $logDirectory 'stop-training') -Force | Out-Null
Write-Output 'Stop requested. The driver will save a checkpoint and shut down its own simulator.'
