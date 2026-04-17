$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
& "$ScriptDir\setup_uv.ps1" @args
