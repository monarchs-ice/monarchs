
FROM python:3.11-slim-bookworm AS compile-image

RUN apt-get update && apt-get install -y --no-install-recommends \
    gfortran \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/monarchs

COPY requirements.txt .

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install -r requirements.txt

FROM python:3.11-slim-bookworm AS build-image

RUN apt-get update && apt-get install -y --no-install-recommends \
    gfortran \
    vim \
    nano \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/venv/bin:$PATH"
COPY --from=compile-image /opt/venv /opt/venv

# .git must be included in the build context: the package version is
# derived from the git tag at install time (hatch-vcs)
COPY . /opt/monarchs/MONARCHS
RUN pip install -e /opt/monarchs/MONARCHS

WORKDIR /opt/monarchs/MONARCHS

# keep the container alive when run detached so users can exec/shell in
CMD ["tail", "-f", "/dev/null"]
