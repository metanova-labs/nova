# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Accept build argument to override the DEVICE setting in PSICHIC's runtime config.
ARG DEVICE_OVERRIDE="none"

ENV PYTHONUNBUFFERED=1

# Install OS dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirement files
COPY requirements1.txt requirements1.txt
COPY requirements2.txt requirements2.txt

# Upgrade pip and install dependencies
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install -r requirements1.txt
RUN pip install -r requirements2.txt

# Copy the entire project
COPY . /app

# Create /scripts folder and copy the entrypoint script
RUN mkdir -p /scripts
COPY scripts/entrypoint.sh /scripts/entrypoint.sh
RUN chmod +x /scripts/entrypoint.sh

# Optionally override DEVICE in PSICHIC/runtime_config.py.
# For example, to run on CPU, build with:
#   docker build --build-arg DEVICE_OVERRIDE=cpu -t nova-image .
RUN if [ "$DEVICE_OVERRIDE" != "none" ]; then \
    sed -i "s/DEVICE = 'cuda:0'/DEVICE = '$DEVICE_OVERRIDE'/g" PSICHIC/runtime_config.py; \
    fi

# Use ENTRYPOINT for flexibility. The default command runs the miner.
ENTRYPOINT ["/scripts/entrypoint.sh"]
CMD ["neurons/miner.py", "--logging.debug"] 