# Project Journal

## 2026-05-13

Context:

- The repository contained generated Python artifacts but no tracked source files.
- The first durable work establishes the planning foundation for a professional AlphaZero and MuZero platform.

Decisions:

- Use Zero Lab as the project-facing name.
- Keep public documentation in English.
- Treat chess as the default showcase domain.
- Keep toy games as correctness and CI domains.
- Make evaluation, replay, and hardware profiling first-class systems.
- Delay native search and distributed execution until profiling proves the need.

Artifacts added:

- README.
- Implementation plan.
- Evaluation plan.
- Merge request roadmap.
- Engineering standards.
- Reference map.
- Git ignore rules for generated Python artifacts.

Open questions:

- Which exact hardware target should define the first performance profile.
- Whether the first native backend should be Rust or C++.
- Whether the initial chess adapter should depend on an existing rules library or a small internal representation.
- Which experiment tracker, if any, should be adopted after the first local run format is stable.

Next entry template:

```text
Date:

Context:

Decision:

Implementation:

Verification:

Follow-up:
```
