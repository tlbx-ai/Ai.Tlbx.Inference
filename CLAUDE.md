# Ai.Tlbx.Inference — Project Rules

## Versioning
- Version is managed exclusively in `Directory.Build.props`, never in individual `.csproj` files
- This is a hard rule — csproj files should only contain technical/packaging metadata, not version numbers

## Architecture
- Provider implementations are internal — only `IAiInferenceClient` is the public API
- Tool-calling loop lives only in `AiInferenceClient` facade, not in providers
- All DTOs are sealed records
- Single shared `HttpClient`, providers use absolute URIs per request

## AOT / Trimming
- Library is marked `IsAotCompatible` and `IsTrimmable` — all internal code must be AOT-safe
- **Provider request bodies**: use `System.Text.Json.Nodes.JsonObject`/`JsonArray` — never anonymous types or `Dictionary<string, object?>`
- **Response parsing**: use `JsonDocument.Parse` + manual property access (already AOT-safe)
- **Never** use `JsonSerializer.Serialize/Deserialize` with reflection in internal code
- Any method that requires reflection must be annotated with `[RequiresDynamicCode]` / `[RequiresUnreferencedCode]`
- Generic methods (`CompleteAsync<T>`, `CompleteWithToolsAsync<T>`) have AOT-safe overloads accepting `JsonTypeInfo<T>`
- `JsonSchemaGenerator` uses reflection — AOT consumers provide `CompletionRequest.JsonSchema` directly
- AOT testbed: `dotnet publish tests\Ai.Tlbx.Inference.AotTestbed\Ai.Tlbx.Inference.AotTestbed.csproj -c Release -r win-x64`

## Build
- Target: net9.0
- Must build with 0 errors, 0 warnings (`TreatWarningsAsErrors` is enabled)
- Run tests: `dotnet test tests\Ai.Tlbx.Inference.Tests\Ai.Tlbx.Inference.Tests.csproj`

## Release Process

When user says **"publish a patch/minor/major release"**:

1. Bump the correct version part in `Directory.Build.props` (semver: patch = Z, minor = Y, major = X in X.Y.Z)
2. `dotnet build` + `dotnet test` — must pass
3. Commit: `"Bump version to X.Y.Z"`
4. `dotnet pack src\Ai.Tlbx.Inference\Ai.Tlbx.Inference.csproj -c Release -p:IncludeSymbols=true -p:SymbolPackageFormat=snupkg`
5. `dotnet nuget push` both `.nupkg` and `.snupkg` using `$NUGET_API_KEY` env var to `https://api.nuget.org/v3/index.json`
6. `git push` to main
7. `git tag vX.Y.Z && git push origin vX.Y.Z`
8. `gh release create vX.Y.Z --generate-notes`

The NuGet badge in README auto-updates (shields.io queries nuget.org dynamically).
