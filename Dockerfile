# =============================================================================
# 1. BASE IMAGE
# =============================================================================
# Instead of building an OS from scratch, we start with a lightweight official
# Python image. 'slim' means it strips out unnecessary GUI tools to keep our 
# container footprint as small as possible.
FROM python:3.13-slim

# =============================================================================
# 2. WORKSPACE SETTING
# =============================================================================
# This creates a folder named '/app' inside the container filesystem and sets
# it as the active directory for any subsequent commands.
WORKDIR /app

# =============================================================================
# 3. DEPENDENCY INSTALLATION (Optimized for caching)
# =============================================================================
# We copy ONLY the requirements file first. Docker builds in layers. If you 
# don't change your requirements.txt, Docker will skip installing these 
# libraries on subsequent builds, making your deployment lightning fast.
COPY requirements.txt .

# We run pip inside the container to install the specific packages.
# --no-cache-dir keeps the final Docker image slim by deleting temporary files.
RUN pip install --no-cache-dir -r requirements.txt

# =============================================================================
# 4. COPY CODE & MODEL ARTIFACTS
# =============================================================================
# Now we copy our API logic and the pre-trained mathematical weights into the 
# container's active working directory.
COPY app.py .
COPY model.joblib .

# =============================================================================
# 5. NETWORKING & EXECUTION
# =============================================================================
# Inform Docker that the container will listen for network traffic on port 8000
EXPOSE 8000

# This is the command that fires up the FastAPI server when the container boots.
# We set host to '0.0.0.0' so the API can accept incoming calls from outside the container.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]