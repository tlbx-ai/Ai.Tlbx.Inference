[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [Parameter(Mandatory = $true)]
    [string]$Version
)

$ErrorActionPreference = 'Stop'

if (-not $env:NUGET_API_KEY)
{
    throw 'NUGET_API_KEY is not set.'
}

$packagePath = Join-Path $PSScriptRoot "..\src\Ai.Tlbx.Inference\bin\Release\Ai.Tlbx.Inference.$Version.nupkg"
$packagePath = [System.IO.Path]::GetFullPath($packagePath)

if (-not (Test-Path $packagePath))
{
    throw "Package not found: $packagePath"
}

$arguments = @(
    'nuget',
    'push',
    $packagePath,
    '--api-key', $env:NUGET_API_KEY,
    '--source', 'https://api.nuget.org/v3/index.json',
    '--skip-duplicate'
)

if ($PSCmdlet.ShouldProcess($packagePath, 'Publish NuGet package and associated symbols'))
{
    & dotnet @arguments
    if ($LASTEXITCODE -ne 0)
    {
        throw "dotnet $($arguments -join ' ') failed with exit code $LASTEXITCODE."
    }
}
