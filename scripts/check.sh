#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
ruff check src tests scripts/live_smoke.py
ruff format --check src tests scripts/live_smoke.py
for test_file in tests/test_auth.py tests/test_contract.py tests/test_transport.py tests/test_operations.py; do
    pytest -q "$test_file"
done
python -m subscription_gateway.cli --help
