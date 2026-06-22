FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn httpx

# Install Kronos via git (for the model source files)
RUN git clone https://github.com/shiyu-coder/Kronos.git /app/kronos_repo \
    && cp -r /app/kronos_repo/model /app/ \
    && cp -r /app/kronos_repo/examples /app/ \
    && rm -rf /app/kronos_repo

# Copy application files
COPY app.py .
COPY kronos_server.py .

# Pre-download model on build (cached in image)
RUN python3 -c "
import sys
sys.path.insert(0, '.')
from model import Kronos, KronosTokenizer
print('Downloading Kronos-small...')
tokenizer = KronosTokenizer.from_pretrained('NeoQuasar/Kronos-Tokenizer-base')
model = Kronos.from_pretrained('NeoQuasar/Kronos-small')
print('Model loaded successfully')
" 2>&1 || echo "Model download deferred to runtime (no GPU during build)"

# Expose port
ENV PORT=7860
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
