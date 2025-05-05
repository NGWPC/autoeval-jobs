# Base image 
FROM ghcr.io/osgeo/gdal:ubuntu-small-3.10.2

# Create and set working directory for deployable code
WORKDIR /deploy

# Copy requirements file
COPY requirements.txt ./

# Install prerequisites:
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      python3-dev \
      libgeos-dev \
      python3-pip \
      wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    # Allow system-wide pip install in newer Ubuntu/Debian
    # Add * after python3. to catch versioned directories like python3.10
    rm -f /usr/lib/python3.*/EXTERNALLY-MANAGED

RUN python3 --version
RUN pip3 --version

RUN pip3 install --no-cache-dir -r requirements.txt && \
    rm requirements.txt

# Copy jobs into deploy directory so nomad clients can run them
COPY fim_mosaicker /deploy/fim_mosaicker
COPY hand_inundator /deploy/hand_inundator
COPY metrics_calculator /deploy/calculate_metrics  
COPY agreement_maker /deploy/make_agreement


