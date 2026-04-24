FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /code

# Force CPU-only PyTorch to save gigabytes of space
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Run the FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]