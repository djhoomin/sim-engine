Read README.md first. It holds the whole plan and the reasoning behind it, and nothing has been
built yet.

Working rules for this project:

- Plain Python spike first. Do not reach for Rust until the model semantics have settled.
- Deterministic seeding and common random numbers from the first commit, not retrofitted.
- Every analytical validation in the README (M/M/1, M/M/c Erlang C, Little's Law) gets a test, and
  those tests run before any work against real data.
- A simulation run is an experiment: config hash, seed, version and KPIs recorded. Never report a
  number from a single unreplicated run.
- Prose style for anything written here: no em dashes, no en dashes, no semicolons.
