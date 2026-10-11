Read README.md first. It holds the whole plan, the reasoning behind it, the decisions taken so
far and the current status.

Working rules for this project:

- Plain Python spike first. Do not reach for Rust until the model semantics have settled.
- Deterministic seeding and common random numbers from the first commit, not retrofitted.
- Every analytical validation in the README (M/M/1, M/M/c Erlang C, Little's Law) gets a test, and
  those tests run before any work against real data.
- A simulation run is an experiment: config hash, seed, version and KPIs recorded. Never report a
  number from a single unreplicated run.
- Run `uv run pytest` before and after any change to the engine. The validation suite is the
  definition of correct.
- Prose style for anything written here: no em dashes, no en dashes, no semicolons.
