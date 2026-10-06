# Toy example tests

Smoke tests for `toy_tps_aimmd.py` and `toy_tis_aimmd.py`. They check that:

- `aimmdTIS` imports without ops-setup;
- both scripts start (`--help`);
- a short TPS run produces a model and paths that a short TIS run can use.

Run from `examples/toy_examples` in an environment with aimmd, OPS and ops-setup:

```bash
pytest tests
```
