# -*- coding: utf-8 -*-
"""
build.py — Gera o índice FAISS e os metadados a partir dos documentos em ./livros

Uso:
    python build.py

Reaproveita a lógica de parsing/indexação do app.py.
"""

from app import criar_base, INDEX_PATH, META_PATH
import os

if __name__ == "__main__":
    # Remove índice antigo para reconstruir do zero
    for p in (INDEX_PATH, META_PATH):
        if os.path.exists(p):
            os.remove(p)
            print(f"🗑️  Removido: {p}")

    print("🔨 Reconstruindo base de conhecimento...")
    criar_base()
    print("✅ Build concluído.")
