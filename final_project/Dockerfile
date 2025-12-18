FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for pandas, numpy, scikit-learn
RUN apt-get update && apt-get install -y gcc g++ build-essential curl && rm -rf /var/lib/apt/lists/*

# Copy dependency files and directories
COPY src ./src
COPY artifacts ./artifacts
COPY pyproject.toml config.toml ./
RUN mkdir models
RUN pip install .

# Check it against the config.toml file content!
EXPOSE 8000

CMD ["python", "-m", "telco_predict"]