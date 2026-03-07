# Testing Notes

All files collected by pytest are now assertion-based tests.

## Pytest test suite

- tests/test_parser_basic.py
- tests/test_packaging_smoke.py
- tests/test_nested_parsing.py
- tests/test_required_fields.py
- tests/test_triple_nested.py
- tests/test_toon_generalization.py
- tests/test_lcel_integration.py

## Diagnostics (manual, not part of pytest collection)

- tests/diagnostics/verify_lcel.py

## Local release gates

```bash
python -m pytest -q
python -m ruff check .
```
