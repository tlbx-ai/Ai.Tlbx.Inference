# Changelog

## Unreleased

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
