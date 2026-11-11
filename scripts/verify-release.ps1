param([string]$Configuration = 'Release')
$ErrorActionPreference = 'Stop'
function Assert-Succeeded([string]$Step) {
    if ($LASTEXITCODE -ne 0) { throw "$Step failed with exit code $LASTEXITCODE." }
}
Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    dotnet build Ai.Tlbx.Inference.slnx -c $Configuration --nologo -m:1
    Assert-Succeeded 'Solution build'
    dotnet test tests/Ai.Tlbx.Inference.Tests/Ai.Tlbx.Inference.Tests.csproj -c $Configuration --no-build --nologo
    Assert-Succeeded 'Unit tests on .NET 10 and 11'
    node playground/tests/contracts.mjs
    Assert-Succeeded 'Playground contracts'
    if ($IsWindows) {
        $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
        $installationPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ([string]::IsNullOrWhiteSpace($installationPath)) { throw 'Visual Studio C++ x64 tools are required.' }
        & (Join-Path $installationPath 'Common7\Tools\Launch-VsDevShell.ps1') -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
    }
    $rid = if ($IsWindows) { 'win-x64' } else { 'linux-x64' }
    foreach ($framework in @('net10.0', 'net11.0')) {
        dotnet publish tests/Ai.Tlbx.Inference.AotAudit/Ai.Tlbx.Inference.AotAudit.csproj -c $Configuration -f $framework -r $rid --nologo -m:1
        Assert-Succeeded "Native AOT ($framework)"
        $name = if ($IsWindows) { 'Ai.Tlbx.Inference.AotAudit.exe' } else { 'Ai.Tlbx.Inference.AotAudit' }
        & (Join-Path "tests/Ai.Tlbx.Inference.AotAudit/bin/$Configuration/$framework/$rid/publish" $name)
        Assert-Succeeded "Native AOT smoke ($framework)"
    }
    Write-Host 'Release verification passed.'
} finally { Pop-Location }
