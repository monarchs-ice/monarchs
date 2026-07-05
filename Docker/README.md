# MONARCHS Docker image

A prebuilt image with MONARCHS and all of its dependencies (including a Fortran
compiler and `NumbaMinpack`) installed. The MONARCHS source lives at
`/opt/monarchs/MONARCHS` inside the container, and the `monarchs` CLI is on
`$PATH`.

## Getting the image

### Docker

```bash
docker pull jelsey92/monarchs:latest
```

### Apptainer / Singularity (HPC)

```bash
apptainer pull docker://jelsey92/monarchs      # or: singularity pull ...
```

This will create a `.sif` file that you can run. 

If access is restricted, contact the model maintainers. 

## Running MONARCHS

MONARCHS is driven by a `model_setup.py` file. The CLI is:

```bash
monarchs -i /path/to/model_setup.py            # -i defaults to ./model_setup.py
```

Your setup file, input data, and outputs live on the host, so mount the working
directory into the container.

### Docker

You need to bind your working directory into the container.
From a directory containing `model_setup.py` and its input data, run:

```bash
docker run --rm -v "$PWD:/work" -w /work jelsey92/monarchs monarchs -i model_setup.py
```
Or to do so interactively do

```bash
# interactive shell instead
docker run -it --rm -v "$PWD:/work" -w /work jelsey92/monarchs bash
#   then, inside the container:
monarchs -i model_setup.py
```

`-v "$PWD:/work"` bind-mounts the current directory to `/work`; `-w /work` makes
it the working directory so relative paths in `model_setup.py` resolve and
outputs are written back to the host.

### Apptainer

Apptainer binds your current directory and `$HOME` by default, so you can do:

```bash
# from a directory containing model_setup.py and its input data
apptainer exec monarchs_latest.sif monarchs -i model_setup.py

# or an interactive shell
apptainer shell monarchs_latest.sif
```

For HPC batch jobs, call the `apptainer exec ... monarchs -i model_setup.py`
line from your scheduler script (e.g. a SLURM `sbatch` submission).

## Building the image

```bash
./Docker/docker_image.sh
```

