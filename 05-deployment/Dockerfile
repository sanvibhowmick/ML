FROM agrigorev/zoomcamp-model:2025

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock .python-version ./

# Install dependencies using uv
RUN pip install uv && uv sync --locked && uv pip install uvicorn fastapi

# Copy your app files
COPY predict.py test.py pipeline_v1.bin ./

EXPOSE 9696

# Run the FastAPI app
ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]
