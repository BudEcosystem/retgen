# RETGEN REST API

Production-ready REST API for RETGEN (Retrieval-Enhanced Text Generation).

## Features

- 🚀 FastAPI-based REST API with async support
- 🐳 Docker and Docker Compose deployment
- 🔄 Model training, saving, and loading
- 📝 Single and batch text generation
- 🔒 Nginx reverse proxy with rate limiting
- 📊 Health checks and monitoring
- 🐍 Python client library

## Quick Start

### Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the API server:
```bash
python -m uvicorn src.api.server:app --reload
```

3. Access the API:
- API endpoint: http://localhost:8000
- Interactive docs: http://localhost:8000/docs
- Health check: http://localhost:8000/

### Docker Deployment

1. Build and run with Docker Compose:
```bash
docker-compose up -d
```

2. Check logs:
```bash
docker-compose logs -f
```

3. Stop services:
```bash
docker-compose down
```

### Production Deployment

Use the deployment script:
```bash
# Local deployment
python scripts/deploy_api.py --mode local

# Production deployment
python scripts/deploy_api.py --mode production --host your-server.com
```

## API Endpoints

### Health & Info

- `GET /` - Health check
- `GET /model/info` - Get model information

### Model Management

- `POST /model/train` - Train a new model
- `POST /model/load` - Load a pre-trained model
- `POST /model/save` - Save the current model

### Text Generation

- `POST /generate` - Generate text from a single prompt
- `POST /generate/batch` - Generate text for multiple prompts

## Python Client

```python
from src.api.client import RETGENClient

# Create client
client = RETGENClient("http://localhost:8000")

# Train model
documents = ["Your training documents...", "More documents..."]
client.train_model(documents)

# Generate text
response = client.generate(
    prompt="Natural language processing",
    max_length=50,
    temperature=0.8
)
print(response['generations'][0])
```

## Testing

Run the API tests:
```bash
python scripts/test_api.py
```

## Configuration

### Environment Variables

- `RETGEN_MODEL_PATH` - Path to pre-trained model to load on startup
- `LOG_LEVEL` - Logging level (DEBUG, INFO, WARNING, ERROR)
- `WORKERS` - Number of worker processes

### Docker Configuration

Edit `docker-compose.yml` to:
- Adjust memory limits
- Mount model directories
- Configure environment variables

### Nginx Configuration

Edit `nginx.conf` to:
- Adjust rate limiting
- Configure SSL/TLS
- Add authentication

## Performance Tuning

### API Server

1. Increase workers for CPU-bound operations:
```bash
uvicorn src.api.server:app --workers 4
```

2. Use Gunicorn for production:
```bash
gunicorn src.api.server:app -w 4 -k uvicorn.workers.UvicornWorker
```

### Docker Resources

Adjust in `docker-compose.yml`:
```yaml
deploy:
  resources:
    limits:
      memory: 8G
      cpus: '4'
```

### Model Optimization

1. Use smaller embedding models for faster inference
2. Reduce `retrieval_k` for faster generation
3. Enable caching for repeated queries

## Monitoring

### Logs

- Application logs: `docker-compose logs retgen-api`
- Nginx logs: `docker-compose logs nginx`

### Metrics

Access model metrics via `/model/info`:
- Pattern count
- Model size
- Training time
- Cache hit rate

### Health Checks

- Docker health check: Every 30s
- Endpoint: `GET /`
- Nginx health endpoint: `/health`

## Security

### Rate Limiting

Default limits in nginx.conf:
- General API: 10 requests/second
- Generation endpoints: 2 requests/second

### Authentication

Add basic auth to nginx.conf:
```nginx
auth_basic "Admin Area";
auth_basic_user_file /etc/nginx/.htpasswd;
```

### CORS

Configure allowed origins in server.py:
```python
allow_origins=["https://your-domain.com"]
```

## Troubleshooting

### Import Errors

Ensure PYTHONPATH is set:
```bash
export PYTHONPATH=/app:/app/src
```

### Memory Issues

Increase Docker memory limit or reduce:
- Batch size
- Pattern count
- Embedding cache size

### Slow Generation

- Use CPU-optimized index (Flat instead of IVF)
- Reduce retrieval_k
- Enable embedding cache

## API Examples

### Train Model

```bash
curl -X POST http://localhost:8000/model/train \
  -H "Content-Type: application/json" \
  -d '{
    "documents": ["Document 1", "Document 2"],
    "validation_split": 0.1
  }'
```

### Generate Text

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Natural language",
    "max_length": 50,
    "temperature": 0.8
  }'
```

### Batch Generation

```bash
curl -X POST http://localhost:8000/generate/batch \
  -H "Content-Type: application/json" \
  -d '["Prompt 1", "Prompt 2", "Prompt 3"]' \
  -G --data-urlencode "max_length=30"
```

## License

This project is part of the RETGEN implementation.