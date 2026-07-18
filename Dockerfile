# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Some packages in requirements.txt might be heavy (like jupyter), but we'll install as-is
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Run server.py when the container launches
CMD ["python", "server.py"]
