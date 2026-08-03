
FROM python:3.11-slim-bookworm AS compile-image

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/monarchs

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY . /opt/monarchs/MONARCHS
RUN pip install -e "/opt/monarchs/MONARCHS"

FROM python:3.11-slim-bookworm AS build-image

RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    nano \
    libjemalloc2 \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/venv/bin:$PATH"

# faster malloc implementation for multithreaded workloads - ensure this is
# on PATH
ENV LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libjemalloc.so.2"
COPY --from=compile-image /opt/venv /opt/venv
# the editable install points at this path, so carry the source over from the
# compile stage rather than re-copying and re-installing
COPY --from=compile-image /opt/monarchs/MONARCHS /opt/monarchs/MONARCHS

WORKDIR /opt/monarchs/MONARCHS

# keep the container alive when run detached so users can exec/shell in
CMD ["tail", "-f", "/dev/null"]
