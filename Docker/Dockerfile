
FROM python:3.11-slim-bookworm AS compile-image

RUN apt-get update && apt-get install -y --no-install-recommends \
    gfortran \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/monarchs

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ensure we install the Numba dependencies with [numba]
COPY . /opt/monarchs/MONARCHS
RUN pip install -e "/opt/monarchs/MONARCHS[numba]"

FROM python:3.11-slim-bookworm AS build-image

RUN apt-get update && apt-get install -y --no-install-recommends \
    gfortran \
    vim \
    nano \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/venv/bin:$PATH"
COPY --from=compile-image /opt/venv /opt/venv
# the editable install points at this path, so carry the source over from the
# compile stage rather than re-copying and re-installing
COPY --from=compile-image /opt/monarchs/MONARCHS /opt/monarchs/MONARCHS

WORKDIR /opt/monarchs/MONARCHS

# keep the container alive when run detached so users can exec/shell in
CMD ["tail", "-f", "/dev/null"]
