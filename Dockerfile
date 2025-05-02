FROM ghcr.io/osgeo/gdal:ubuntu-small-3.10.2

# Create and set working directory for deployable code
WORKDIR /deploy

# Copy requirements file
COPY requirements.txt ./

# Install Python pip and clean up
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3-pip wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    # Allow pip install in newer Ubuntu/Debian
    rm -f /usr/lib/python3.*/EXTERNALLY-MANAGED

# Install Python dependencies from requirements.txt
# Using --no-cache-dir potentially saves space
RUN pip3 install --no-cache-dir -r requirements.txt && \
    rm requirements.txt

# copy jobs into deploy directory so nomad clients can run them
COPY fim_mosaicker /deploy/fim_mosaicker
COPY hand_inundator /deploy/hand_inundator
