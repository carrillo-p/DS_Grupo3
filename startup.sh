gunicorn \
    --timeout 600 \  # Ajuste de tiempo de espera a 10 minutos
    --workers 1 \  # Un solo worker por los recursos de B3
    --threads 4 \  # Mayor concurrencia sin aumentar el número de workers
    --access-logfile - \  # Logs de acceso, pueden ser opcionales o JSON
    --error-logfile - \  # Logs de error
    --log-level info \  # Ajustar el nivel de logs según necesidades
    --bind 0.0.0.0:8000 \  # Escucha en todas las interfaces
    --worker-class uvicorn.workers.UvicornWorker \  # Usar Uvicorn para ASGI
    --worker-tmp-dir /dev/shm \  # Directorio temporal en memoria compartida
    app:app