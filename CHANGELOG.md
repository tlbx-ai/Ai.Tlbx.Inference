# Changelog

## 2.2.2 - 2026-08-15

### Fixed

- GPT-5.6 Luna now prefers the Responses API and sends its reasoning effort there, enabling high-reasoning function-tool loops.
- Responses API tool loops now preserve the distinct function-item (`fc_...`) and tool-call (`call_...`) identifiers across turns.

## 2.2.0 - 2026-08-15

### Added

- GPT-5.6 Luna (`gpt-5.6-luna`) with reasoning, endpoint capability, and standard-price metadata.

### Fixed

- Token totals and token costs no longer add reasoning tokens twice when a provider reports reasoning as a subset of billed output tokens.
- Google usage normalizes separately reported thought tokens into the billed output total while retaining the reasoning subset for diagnostics.

### Added

- Typed image-model routing through `ImageGenerationModel`, including OpenAI GPT Image 2, GPT Image 1.5, GPT Image 1, and GPT Image 1 Mini.
- OpenAI Image API generation with explicit PNG output plus size and quality mapping.

### Fixed

- Google image generation now targets `gemini-2.5-flash-image`, explicitly requests the `IMAGE` modality, and parses inline image data from current Gemini responses.

### Added

- First-class `AiModelCatalog` registry with explicit endpoint and capability metadata.
- `AiModelValidator` for preflight validation and agent-friendly model checks.
- `CompletionRequestProfiles.CreateSmoke` and `CreateSmokeRetry` for low-cost live verification flows.
- `CompletionDiagnostics` on completion responses.
- Dedicated live-test manifest for paid provider integration coverage.

### Changed

- OpenAI routing now uses the Responses API for models that are not chat-completions compatible in this library.
- Paid smoke tests now use model-aware token budgets and a single retry on empty success responses.
- Structured output and typed tool results now require `JsonTypeInfo<T>` overloads for a strictly AOT-/trim-safe public API.
- Google Vertex service-account authentication no longer depends on `Google.Apis.Auth`; the library now acquires tokens through a native JWT bearer flow.

### Model Catalog

- Removed GPT-5.3 Codex.
- Replaced GPT-5.3 Instant with GPT-5.3 Chat (`gpt-5.3-chat-latest`).
- Replaced Gemini 3.1 stable aliases with preview model ids where required by current provider docs.
- Removed obsolete models: `Gpt5`, `O4Mini`, `Gemini25Pro`, `O3`, `O3Pro`, `Gemini3ProPreview`, `Gemini25Flash`, and `Grok3`.
- Replaced `ClaudeSonnet45` / `ClaudeSonnet4` with `ClaudeSonnet46` (`claude-sonnet-4-6`).
