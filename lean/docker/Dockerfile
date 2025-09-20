# Base QuantConnect Lean CLI image
FROM quantconnect/lean:latest

# Install additional tooling for provenance/logging if needed
RUN pip install --no-cache-dir \
    boto3 \
    pandas \
    pyyaml \
    requests \
    lean

RUN python - <<'PY'
from pathlib import Path
import requests

url = "https://cdn.quantconnect.com/cli/modules-1.14.json"
path = Path("/opt/miniconda3/lib/python3.11/site-packages/lean/modules-1.14.json")
path.parent.mkdir(parents=True, exist_ok=True)
res = requests.get(url, timeout=30)
res.raise_for_status()
path.write_bytes(res.content)
PY

RUN chmod -R a+rX /opt/miniconda3/lib/python3.11/site-packages/lean

# Non-root execution for security
RUN useradd -m leanuser
USER leanuser

WORKDIR /Lean

# Expect algorithms/config mounted at runtime; no secrets baked into image.
CMD ["lean", "--help"]
