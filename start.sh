#!/bin/bash
set -e

echo "=== APITO GPT ==="

# Constrói o índice ANTES de subir os serviços.
#
# A API e a interface carregam o mesmo motor. Se ambas subissem com o índice
# ausente, cada uma dispararia criar_base() ao mesmo tempo, gravando nos mesmos
# arquivos .faiss/.pkl — a corrida deixa o índice corrompido. Fazer aqui, num
# único processo, também evita manter duas cópias dos modelos em memória
# durante a indexação.
echo "Preparando índice das Regras (pode levar alguns minutos na 1ª vez)..."
python -c "import app; app._carregar_indices(); print('Índice pronto.')"

# FastAPI em background
uvicorn api:app --host 0.0.0.0 --port 8000 &
API_PID=$!
echo "API REST iniciada (PID $API_PID) -> porta 8000"

# Encerra a API junto com o container, em vez de deixá-la órfã
trap 'kill "$API_PID" 2>/dev/null || true' TERM INT

# Streamlit em foreground (processo principal do container)
echo "Iniciando Streamlit -> porta 8501"
exec streamlit run streamlit_app.py \
    --server.address=0.0.0.0 \
    --server.port=8501 \
    --server.headless=true
