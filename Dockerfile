FROM ghcr.io/astral-sh/uv:debian

# Add the contents of the current directory to /app in the container
WORKDIR /app
COPY . /app
# Install dependencies using uv
RUN uv sync --locked

# Expose the port that the app runs on
EXPOSE 8000

# Start the app
CMD ["uv", "run", "streamlit", "run", "app.py", "--server.headless", "true", "--server.port=8000", "--server.address=0.0.0.0"]

# Trigger the build for 2025-08-13