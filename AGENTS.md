# AGENTS.md

## Purpose

This repository is `Ai.Tlbx.Inference`, a .NET 9 library centered on a single public inference facade. Agents working here should preserve the existing API shape, keep the code AOT- and trimming-safe, and treat build cleanliness as a release requirement rather than a nice-to-have.

## Core Rules

### Versioning

- Keep the package version in `Directory.Build.props` only.
- Do not add or change version numbers inside individual `.csproj` files.
- Treat version placement as a repository rule, not a stylistic preference.

### Public API and Architecture

- Provider implementations must stay internal.
- `IAiInferenceClient` is the public API surface agents should protect.
- The tool-calling loop belongs in the `AiInferenceClient` facade only.
- Do not move tool orchestration into provider-specific implementations.
- DTOs should be modeled as sealed records.
- Reuse a single shared `HttpClient`; providers should send requests with absolute URIs.

## AOT and Trimming Requirements

This library is explicitly built for AOT/trimming compatibility.

- The project is marked `IsAotCompatible` and `IsTrimmable`; internal code must respect that.
- Build provider request payloads with `System.Text.Json.Nodes.JsonObject` and `JsonArray`.
- Do not use anonymous objects or `Dictionary<string, object?>` for provider request bodies.
- Parse provider responses with `JsonDocument.Parse` and explicit property access.
- Do not introduce internal reflection-based `JsonSerializer.Serialize` or `JsonSerializer.Deserialize` usage.
- If a method truly requires reflection-sensitive behavior, annotate it with `[RequiresDynamicCode]` and/or `[RequiresUnreferencedCode]` as appropriate.
- Generic APIs such as `CompleteAsync<T>` and `CompleteWithToolsAsync<T>` must continue to support AOT-safe overloads via `JsonTypeInfo<T>`.
- `JsonSchemaGenerator` relies on reflection; for AOT consumers, prefer `CompletionRequest.JsonSchema`.

### AOT Validation Command

Use this when validating the AOT path:

```powershell
dotnet publish tests\Ai.Tlbx.Inference.AotAudit\Ai.Tlbx.Inference.AotAudit.csproj -c Release -r win-x64
```

## Build and Test Expectations

- Target framework is `net9.0`.
- The repository is expected to build with zero errors and zero warnings.
- `TreatWarningsAsErrors` is enabled, so warnings are release-blocking by default.
- Live provider integration tests cost money and must never be run automatically.
- Run live provider integration tests only when the user explicitly asks for them.
- After large or high-impact changes, pause and ask the user to greenlight the paid integration test run before executing it.

Primary test command:

```powershell
dotnet test tests\Ai.Tlbx.Inference.Tests\Ai.Tlbx.Inference.Tests.csproj
```

Live provider integration test command:

```powershell
dotnet test tests\Ai.Tlbx.Inference.IntegrationTests\Ai.Tlbx.Inference.IntegrationTests.csproj
```

## Working Agreement for Agents

When making changes in this repo:

- Preserve the single-facade design around `IAiInferenceClient` and `AiInferenceClient`.
- Favor small, explicit JSON construction/parsing over convenience patterns that can break under AOT or trimming.
- Assume API surface changes are high-impact unless clearly requested.
- Verify builds/tests before considering work complete when the change could affect behavior or packaging.
- Treat provider/model integration coverage as important because model-specific API nuances can change independently.

## Release Workflow

If the user asks to publish a patch, minor, or major release, follow this exact sequence:

1. Update the semantic version in `Directory.Build.props`.
2. Run `dotnet build` and `dotnet test`; both must pass cleanly.
3. Commit with message `Bump version to X.Y.Z`.
4. Pack with:

```powershell
dotnet pack src\Ai.Tlbx.Inference\Ai.Tlbx.Inference.csproj -c Release -p:IncludeSymbols=true -p:SymbolPackageFormat=snupkg
```

5. Publish to NuGet with:

```powershell
.\scripts\publish-nuget.ps1 -Version X.Y.Z
```

6. Push `main`.
7. Create and push tag `vX.Y.Z`.
8. Create the GitHub release with generated notes:

```powershell
gh release create vX.Y.Z --generate-notes
```

## Notes

- The README NuGet badge updates automatically because `shields.io` reads from NuGet dynamically.
