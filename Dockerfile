# ── Base ──────────────────────────────────────────────────────────────
FROM python:3.11-slim

# System deps necessários para faiss, torch e sentence-transformers
# curl é usado pelo HEALTHCHECK do compose e não vem na imagem slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Dependências Python ───────────────────────────────────────────────
# Torch vem antes, pelo índice CPU-only: instalado como dependência de
# sentence-transformers ele traria as bibliotecas CUDA, que somam alguns GB
# inúteis num container sem GPU.
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Pré-download dos modelos (torna o container auto-suficiente) ──────
# Precisa espelhar EMBED_MODEL_NAME / CROSS_ENCODER_NAME de app.py; caso
# contrário o container baixa os pesos no primeiro acesso do usuário.
RUN python - <<'EOF'
from sentence_transformers import SentenceTransformer, CrossEncoder
print("Baixando embedding model...")
SentenceTransformer("intfloat/multilingual-e5-base")
print("Baixando cross-encoder...")
CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", max_length=512)
print("Modelos baixados.")
EOF

# ── Código da aplicação ───────────────────────────────────────────────
# .dockerignore impede que índices gerados por outro modelo entrem na imagem;
# são reconstruídos no primeiro start conforme o modelo configurado.
COPY . .

# ── Streamlit config (headless, sem CORS) ────────────────────────────
RUN mkdir -p /root/.streamlit && cat > /root/.streamlit/config.toml <<'EOF'
[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = false

[theme]
base = "dark"
primaryColor = "#7c3aed"
backgroundColor = "#0f172a"
secondaryBackgroundColor = "#1e1e2e"
textColor = "#f1f5f9"
EOF

# ── Portas ────────────────────────────────────────────────────────────
EXPOSE 8501 8000

# ── Startup ───────────────────────────────────────────────────────────
COPY start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]
