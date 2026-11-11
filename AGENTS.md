# AGENTS.md

## Purpose

This repository is `Ai.Tlbx.Inference`, a .NET 10/11 library centered on a single public inference facade. Agents working here should preserve the existing API shape, keep the code AOT- and trimming-safe, and treat build cleanliness as a release requirement rather than a nice-to-have.

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
dotnet publish tests\Ai.Tlbx.Inference.AotAudit\Ai.Tlbx.Inference.AotAudit.csproj -c Release -f net11.0 -r win-x64
```

## Build and Test Expectations

- Library targets `net10.0` and `net11.0`; unit tests and AOT audit target both frameworks.
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

1. Keep the exact intended semantic version in `Directory.Build.props`.
2. Run `pwsh -File scripts/verify-release.ps1`: clean build, unit tests on both frameworks, playground contracts, Native AOT publish/smoke on both frameworks. No paid provider tests run in this gate.
3. Commit and push the existing default branch `master`; never create a release branch.
4. Publish using `./scripts/publish-nuget.ps1 -Version X.Y.Z`. This dispatches `.github/workflows/publish-nuget.yml` from `master`. The workflow verifies, packs only the library, and uses `NuGet/login` OIDC through the master-only `nuget` environment. Do not use a stored API key.
5. Wait for the exact run to finish. Retry the same version after a failed upload. NuGet policy must bind `tlbx-ai/Ai.Tlbx.Inference`, `publish-nuget.yml`, environment `nuget`, and package `Ai.Tlbx.Inference`.
For repeated startup_failure with zero jobs, inspect the run annotations, compare the working release configuration, then isolate runner and environment startup with a minimal no-upload workflow before escalating to external support. A generic internal-error annotation alone does not establish the cause.

6. Create/push `vX.Y.Z` and the GitHub release with package and symbols. Verify exact public NuGet downloads and fresh-cache package-only consumers for net10/net11. The web playground is a separate `IsPackable=false` application and must never enter the package.

## Notes

- The README NuGet badge updates automatically because `shields.io` reads from NuGet dynamically.

## Recurring model and API maintenance

Treat **"update ai.tlbx.inference"** as the complete maintenance workflow:

1. Preserve unrelated work; inspect current source and published version.
2. Compare official provider model catalogs, API guides, changelogs, retirements and prices. Cover OpenAI, Anthropic, Google and xAI; inspect relevant additional labs and record material coverage gaps in TODO.md.
3. Add important current text/reasoning, image, audio and embedding models supported by this facade. Append enum values so persisted selections remain stable. Update capabilities, endpoint routing, prices, dated promotions, aliases and retirement notes together.
4. Adapt existing operations to provider API changes, including tool/assistant history, signed or encrypted reasoning continuation, schemas, streaming failures and usage accounting. Preserve the single facade and AOT rules.
5. Run the unit suite, clean Release build and Native AOT audit. Run targeted paid acceptance for changed provider operations only when covered by explicit user approval or a standing paid-test budget; honor that budget and retain usage/cost logs. Do not run the whole paid suite by default.
6. Update README, CHANGELOG and the live-test manifest. Publish a semantically appropriate NuGet/GitHub release when there are verified changes; do not package unrelated work. Verify published artifacts and a package-only consumer. If current, report that without an empty release.
7. Report shipped models/API fixes, version/commit, validation and actual limitations. Update the existing TODO for material follow-ups; do not create a separate maintenance report. Do not update consuming products automatically.

Paid acceptance command: `dotnet test tests\Ai.Tlbx.Inference.IntegrationTests\Ai.Tlbx.Inference.IntegrationTests.csproj --filter FullyQualifiedName~CatalogMaintenanceTests`. Configure OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY and XAI_API_KEY only in the process environment; GEMINI_API_KEY can supply the Google key. Validate catalog access before spending on tests.
