# Inference Playground

Small ASP.NET Core app that uses the project reference to `Ai.Tlbx.Inference`. It is not packed into the NuGet library.

Set `OPENAI_API_KEY` in the server process and run:

```powershell
dotnet run --project playground/Ai.Tlbx.Inference.Playground
```

The browser never receives the API key. The app has no user authentication, so bind it only to an interface whose users you trust. It is an experiment, not a public service.

Normal chat uses `CompleteAsync` and keeps its history in the current browser tab. Hosted agent mode creates a persistent OpenAI Agents API session. The session ID is stored in browser local storage so a reload can retrieve saved items. New sessions use an OpenAI-hosted Linux workspace with outbound sandbox network access disabled by default. You can enable network access, web search, or configure a public HTTPS MCP server. HTTP MCP uses `connection_origin: service`, so OpenAI makes the connection and executes the MCP calls. Bash and file operations run in the hosted workspace. Application-side function callbacks are deliberately outside this mode.

Agent events are streamed as newline-delimited JSON. The library subscribes before submitting each task, and ends only when the root turn completes. On a broken stream, retrieve session items and status; do not send the task again automatically. The playground shows event types in the activity strip and offers saved-item recovery on mode switch or page reload. Use **Turn abbrechen** to request cancellation of remote work; closing the browser or the event stream does not cancel it. Session deletion is explicit.

The HTTP adapter is covered with mock tests. No paid live Agents API test runs in the normal test suite.

The header shows accumulated model-token cost and token counts for this browser. Normal chat uses `AiModelCostCatalog` and each completion's reported usage; agent sessions use all reported root and subagent turn usage. Agent accounting can arrive after a turn completes, so the app makes up to ten refresh attempts; switching back to Hosted Agent refreshes it again. The dollar amount is an estimate based on published short-context rates and excludes tool, sandbox compute, cache-write, and third-party charges. The app does not impose a spending cap. Agent token prices come from the library catalog; unknown model IDs show no invented estimate.
