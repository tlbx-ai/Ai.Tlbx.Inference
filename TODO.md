# TODO

- [ ] Evaluate adding text inference for Mistral Large 4/Medium 3.5/Small 4; AddMistral currently configures Voxtral audio only. Review other major labs against concrete consumer demand.
- [ ] Add opt-in bounded Argon long-decode continuation; current REST/streaming calls report the provider continuation boundary without starting more paid calls.
- [ ] Extend embedding inputs beyond text for Gemini Embedding 2 when a consumer needs multimodal vectors.

- [x] Add a first-class AI model registry with explicit capability metadata.
- [x] Route providers by model capability instead of ad hoc model-name checks where practical.
- [x] Add built-in smoke-test request profiles for low-cost live validation.
- [x] Improve completion diagnostics so callers can see endpoint family, stop reasons, and likely truncation.
- [x] Add a model validation helper for preflight checks and agent workflows.
- [x] Add a dedicated live-test manifest describing paid integration coverage.
- [x] Document paid integration test workflow and approval expectations.
- [x] Add changelog notes for model catalog changes.
