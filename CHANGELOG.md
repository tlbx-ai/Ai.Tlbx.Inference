# Changelog

## 2.7.0 - 2026-10-27

### Added

- GPT-6 Astra, GPT-6.1 Terra and Chat Latest; Claude Fable 5.1 and Opus/Sonnet/Haiku 5.5; Gemini 4 Argon, 3.8 Flash and stable 3.5/3.1 Flash-Lite; Grok 4.7, 4.6 and 4.3.
- GPT Image 2.5 Sunburst/Flare, Nano Banana 2.1, Gemini 3.1 Flash Image and 3 Pro Image, plus Gemini Embedding 2 text embeddings.
- Google image editing with multiple references and explicit 1K/2K/4K resolutions.
- Voxtral 3 transcription/speech through OpenRouter or Mistral, and the global Vertex endpoint correction from the intervening source commits.
- Context-sensitive prices, expiring Google launch discounts and OpenAI Ultrafast tier support.
- Bounded manual maintenance acceptance and a recurring update workflow in the agent instructions.

### Fixed

- Current Claude adaptive thinking, effort and native structured output without replacing caller tools.
- Complete signed Claude tool history and encrypted OpenAI/xAI reasoning continuation, including streaming.
- Ordinary assistant history and JSON schema output in Responses requests; failed/incomplete stream handling and reported failure usage.
- Gemini thinking levels, JSON schemas, native tool IDs/names/signatures, thought filtering, and embedding token accounting.
- Incorrect non-reasoning Grok 4.1 alias; retirement warnings explain redirects and billing changes.
- Google debug URLs no longer include the API key.

### Limits

- Argon long-decode continuation is reported but not followed automatically. Gemini Embedding 2 vectors require re-embedding existing 001 data.
- New text provider integrations, including Mistral, remain explicit follow-up work; Mistral audio is retained.

## 2.6.0 - 2026-09-30

- GPT-6 Luna/Sol and GPT-6.1 Sol model metadata, Responses routing and reasoning/service-tier settings.

## 2.5.0 - 2026-09-11

- Per-provider-call tool-iteration accounting via `IToolIterationObserver` for normal, typed and streaming tool loops.
- NuGet publication failures no longer disclose credentials.

## 2.4.0 - 2026-08-20

### Added

- First-class `gpt-image-2` editing through `EditImageAsync`, including multiple reference images and explicit multipart Image API support.

## 2.3.0 - 2026-08-20

### Added

- Provider-neutral grounded web, image, and X-search options with domain filters, search limits, and approximate user location.
- Normalized cited sources, image results, search queries, and billable hosted-tool usage for OpenAI, Anthropic, Google, and xAI.
- Grounding metadata propagation through typed completions, client tool loops, and tool streaming.
- Hosted-search list prices and combined token-plus-tool cost calculation in `AiModelCostCatalog`.

### Changed

- Web search now uses each provider's native hosted-search contract instead of requiring every model to expose an OpenAI Responses capability.
- Existing `EnableWebSearch` and `EnableXSearch` flags remain compatible shorthands for the expanded grounding contract.

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
