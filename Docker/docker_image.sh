#!/bin/bash
# call from MONARCHS root directory
# replace "monarchs" with "<username>/monarchs" if intending to upload to Dockerhub.
docker build  -t jelsey92/monarchs -f Docker/Dockerfile .
docker push jelsey92/monarchs