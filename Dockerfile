FROM python:3.12-slim

# Set up a working directory
WORKDIR /app

# Copy requirement list and install it first to cache the layer
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the application files
COPY . /app/

# Hugging Face Spaces requires running the app as a non-root user
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Change working directory to the user's home
WORKDIR $HOME/app
COPY --chown=user . $HOME/app

# Expose port 7860 which is the exact port Hugging Face Spaces listens to
EXPOSE 7860

# Start the Flask app using Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:7860", "--workers", "2", "--timeout", "120", "app:app"]
