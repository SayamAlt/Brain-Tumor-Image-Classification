# Use Python 3.10 base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy requirements first to cache dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port and run the app
EXPOSE 8000
CMD ["gunicorn", "--workers=4", "--bind", "0.0.0.0:8000", "app:app"]
