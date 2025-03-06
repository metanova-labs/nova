# Use an official Python runtime as a parent image
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

# Install OS dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /root

# Copy requirement files
COPY requirements1.txt requirements1.txt
COPY requirements2.txt requirements2.txt

# Upgrade pip and install dependencies
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install -r requirements1.txt
RUN pip install -r requirements2.txt

# Copy the entire project
COPY . /root

# Create /scripts folder and copy the entrypoint script
RUN mkdir -p /scripts
COPY scripts/entrypoint.sh /scripts/entrypoint.sh
RUN chmod +x /scripts/entrypoint.sh

# Use ENTRYPOINT for future flexibility. Run validator
ENTRYPOINT ["/scripts/entrypoint.sh"]
CMD ["neurons/validator.py", "--logging.debug"] 
