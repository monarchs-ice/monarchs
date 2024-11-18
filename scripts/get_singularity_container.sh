#!/bin/bash

# Simple script for obtaining a container that can run MONARCHS onto a HPC system using Singularity.
module load singularity
# remove container if it already exists so we can overwrite
rm monarchs-latest.sif

singularity pull docker://jelsey92/monarchs