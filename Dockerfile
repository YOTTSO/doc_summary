# Use a slim Debian-based Python image
# This allows us to easily install system packages like Tesseract
FROM python:3.10-slim-bullseye

# Set the working directory in the container
WORKDIR /app

# Install Tesseract OCR and other necessary system packages
# tesseract-ocr is the main engine
# tesseract-ocr-eng installs the English language data (usually included by default, but explicit is clearer)
# libtesseract-dev and libleptonica-dev might be needed for building pytesseract, though often not strictly required with wheels
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    libleptonica-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install Python dependencies
# Using --no-cache-dir reduces image size
COPY src/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
# This includes app.py, engine.py, model.py, and any templates/static folders
COPY src .

# Expose the port the Flask app runs on (default 5000)
EXPOSE 5000

# Command to run the Flask application
# Set Flask environment variables
CMD ["python", "app.py"]