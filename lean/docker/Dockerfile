# Base QuantConnect Lean CLI image
FROM quantconnect/lean:latest

# Install additional tooling for provenance/logging if needed
RUN pip install --no-cache-dir \
    boto3 \
    pandas \
    pyyaml \
    requests

# Non-root execution for security
RUN useradd -m leanuser
USER leanuser

WORKDIR /Lean

# Expect algorithms/config mounted at runtime; no secrets baked into image.
CMD ["lean", "--help"]
