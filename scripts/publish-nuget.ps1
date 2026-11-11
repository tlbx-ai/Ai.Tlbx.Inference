[CmdletBinding(SupportsShouldProcess = $true)]
param([Parameter(Mandatory = $true)][ValidatePattern('^\d+\.\d+\.\d+$')][string]$Version)
$ErrorActionPreference = 'Stop'
[xml]$props = Get-Content (Join-Path $PSScriptRoot '../Directory.Build.props')
if ([string]$props.Project.PropertyGroup.Version -ne $Version) { throw 'Version must match Directory.Build.props.' }
if ($PSCmdlet.ShouldProcess("Ai.Tlbx.Inference $Version", 'Start GitHub OIDC Trusted Publishing from master')) {
    gh workflow run publish-nuget.yml --repo tlbx-ai/Ai.Tlbx.Inference --ref master -f "version=$Version"
    if ($LASTEXITCODE -ne 0) { throw 'Could not start the publishing workflow.' }
}
