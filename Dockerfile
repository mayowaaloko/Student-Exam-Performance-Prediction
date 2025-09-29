# Use the official Python 3.13 image as the base
FROM python:3.13-slim

# Set the working directory inside the container
WORKDIR /app

# Copy your application code into the container
COPY . /app

# Install any Python dependencies (if you have a requirements.txt)
RUN pip install -r requirements.txt

# Define the command to run your application
CMD ["python", "app.py"]