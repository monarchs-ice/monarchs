#Deriving the latest base image
FROM python:3.11-slim-bullseye AS compile-image

# set working directory
WORKDIR .

# copy requirements.txt file from local (source) to file structure of container (destination) 
COPY ../requirements.txt requirements.txt

# Run update on base image so we can locate gfortran later
RUN apt -y update && apt -y upgrade && apt-get install -y \
gfortran \
build-essential 
RUN apt-get install mpich -y

# Install the requirements specified in file using RUN
# switch to virtual environment first
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install -r requirements.txt

# copy all items in current local directory (source) to current container directory (destination)
COPY .. ./MONARCHS

FROM python:3.11-slim-bullseye AS build-image
RUN apt -y update && apt -y upgrade && apt-get install -y \
gfortran \
build-essential \
vim
RUN apt-get install nano -y 
RUN apt-get install mpich -y

ENV PATH="/opt/venv/bin:$PATH"
COPY --from=compile-image /opt/venv /opt/venv
COPY .. ./MONARCHS
RUN echo $PWD | ls -l
RUN pip install -e ./MONARCHS/.

# command to run when image is executed inside a container
# CMD [ "bash" ]
# don't perform any commands when the container is run
CMD tail -f /dev/null
