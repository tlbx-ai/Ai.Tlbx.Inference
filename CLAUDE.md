# Ai.Tlbx.Inference — Project Rules

## Versioning
- Version is managed exclusively in `Directory.Build.props`, never in individual `.csproj` files
- This is a hard rule — csproj files should only contain technical/packaging metadata, not version numbers

## Architecture
- Provider implementations are internal — only `IAiInferenceClient` is the public API
- Tool-calling loop lives only in `AiInferenceClient` facade, not in providers
- All DTOs are sealed records
- Single shared `HttpClient`, providers use absolute URIs per request

## Build
- Target: net9.0
- Must build with 0 errors, 0 warnings (`TreatWarningsAsErrors` is enabled)
- Run tests: `dotnet test tests\Ai.Tlbx.Inference.Tests\Ai.Tlbx.Inference.Tests.csproj`
