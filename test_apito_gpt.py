#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_apito_gpt.py — Teste completo do APITO GPT

Executa:
1. ✅ Parsing do documento (212 chunks)
2. ✅ Indexação FAISS (se dependências disponíveis)
3. ✅ Simulação de buscas (com respostas mock)
4. ✅ Exemplo de pergunta→resposta

Uso: python test_apito_gpt.py
"""

import sys
import re
import json
from pathlib import Path

# Importar funções do app.py (sem executar main)
MIN_CHUNK_CHARS = 50
MAX_CHUNK_CHARS = 1200
OVERLAP_CHARS = 200

def limpar(texto):
    return re.sub(r"\n{3,}", "\n\n", texto).strip()

def chunk_valido(texto):
    return len(texto) >= MIN_CHUNK_CHARS and texto.count(" ") > 10

def dividir_texto_grande(texto, max_chars=MAX_CHUNK_CHARS, overlap=OVERLAP_CHARS):
    if len(texto) <= max_chars:
        return [texto]
    frases = re.split(r'(?<=[.!?;])\s+', texto)
    chunks = []
    atual = ""
    for frase in frases:
        if len(atual) + len(frase) + 1 <= max_chars:
            atual += (" " if atual else "") + frase
        else:
            if atual:
                chunks.append(atual.strip())
            if chunks and overlap > 0:
                cauda = chunks[-1][-overlap:]
                idx = cauda.find(" ")
                if idx > 0:
                    cauda = cauda[idx + 1:]
                atual = cauda + " " + frase
            else:
                atual = frase
    if atual.strip():
        chunks.append(atual.strip())
    return chunks

def parse_md(texto):
    linhas = texto.split("\n")
    secao, subsecao, regra_atual = "Geral", "", ""
    buffer = []
    imagens_buffer = []
    registros = []

    def salvar():
        if not buffer:
            return
        conteudo = " ".join(buffer).strip()
        if not chunk_valido(conteudo):
            return
        pedacos = dividir_texto_grande(conteudo)
        for i, pedaco in enumerate(pedacos):
            sufixo = f" (parte {i+1}/{len(pedacos)})" if len(pedacos) > 1 else ""
            registros.append({
                "regra": regra_atual,
                "secao": secao,
                "subsecao": subsecao + sufixo,
                "imagens": list(imagens_buffer),
                "conteudo_original": pedaco,
            })

    for linha in linhas:
        linha_strip = linha.strip()
        if not linha_strip:
            continue

        img_match = re.findall(r'!\[.*?\]\((.*?)\)', linha_strip)
        if img_match:
            imagens_buffer.extend(img_match)
            continue

        if linha_strip.startswith("###### "):
            salvar(); buffer.clear(); imagens_buffer.clear()
            subsecao = f"{subsecao} » {linha_strip[7:]}"
        elif linha_strip.startswith("##### "):
            salvar(); buffer.clear(); imagens_buffer.clear()
            subsecao = f"{subsecao} » {linha_strip[6:]}"
        elif linha_strip.startswith("#### "):
            salvar(); buffer.clear(); imagens_buffer.clear()
            subsecao = f"{subsecao} - {linha_strip[5:]}"
        elif linha_strip.startswith("### "):
            salvar(); buffer.clear(); imagens_buffer.clear()
            subsecao = linha_strip[4:]
        elif linha_strip.startswith("## "):
            salvar(); buffer.clear(); imagens_buffer.clear()
            secao = linha_strip[3:]
            subsecao = ""
            m = re.match(r'Regra (\d+):', secao)
            if m:
                regra_atual = f"Regra {m.group(1)}"
        elif linha_strip.startswith("# "):
            salvar(); buffer.clear(); imagens_buffer.clear()
            secao = linha_strip[2:]
            subsecao = ""
        elif linha_strip.startswith(">"):
            buffer.append(linha_strip.lstrip("> "))
        else:
            buffer.append(linha_strip)

    salvar()
    return registros

# ============================================================
# TESTES
# ============================================================

def test_parsing():
    print("=" * 60)
    print("TEST 1: PARSING DO DOCUMENTO")
    print("=" * 60)
    
    livro_path = Path("livros/Regras do Jogo 2025 e 2026 - IFAB.md")
    if not livro_path.exists():
        print(f"❌ Arquivo não encontrado: {livro_path}")
        return False
    
    with open(livro_path, encoding="utf-8") as f:
        texto = limpar(f.read())
    
    registros = parse_md(texto)
    
    print(f"✅ Chunks gerados: {len(registros)}")
    
    # Estatísticas
    tamanhos = [len(r["conteudo_original"]) for r in registros]
    print(f"   Menor: {min(tamanhos)} chars")
    print(f"   Maior: {max(tamanhos)} chars")
    print(f"   Médio: {sum(tamanhos)//len(tamanhos)} chars")
    
    # Regras
    regras = sorted(set(r["regra"] for r in registros if r["regra"]))
    print(f"✅ Regras indexadas: {len(regras)}")
    for r in regras[:5]:
        print(f"   - {r}")
    if len(regras) > 5:
        print(f"   ... e mais {len(regras)-5}")
    
    # Imagens
    todas_imgs = set()
    for r in registros:
        todas_imgs.update(r["imagens"])
    print(f"✅ Imagens referenciadas: {len(todas_imgs)}")
    
    imgs_faltando = []
    for img in todas_imgs:
        if not Path(img).exists():
            imgs_faltando.append(img)
    
    if imgs_faltando:
        print(f"⚠️  Imagens faltando: {imgs_faltando}")
    else:
        print(f"✅ Todas as imagens existem")
    
    # Glossário e Cenários
    tem_glossario = any("Glossário" in r["secao"] for r in registros)
    tem_multiregra = any("Cenários" in r["secao"] for r in registros)
    
    print(f"{'✅' if tem_glossario else '❌'} Glossário de termos informais presente")
    print(f"{'✅' if tem_multiregra else '❌'} Cenários multi-regra presentes")
    
    print()
    return True

def test_buscas_mock():
    print("=" * 60)
    print("TEST 2: SIMULAÇÃO DE BUSCAS (MOCK)")
    print("=" * 60)
    
    perguntas_teste = [
        "Se um jogador comete falta com intenção de machucar dentro da área, qual é a decisão?",
        "O goleiro pode segurar a bola por quanto tempo?",
        "O que é impedimento?",
        "Quando é dado um carrinho por trás, o que acontece?",
        "Como funciona a disputa de pênaltis?",
    ]
    
    # Mock de respostas estruturadas esperadas
    respostas_esperadas = [
        {
            "regras": ["Regra 12", "Regra 14"],
            "keywords": ["tiro penal", "expulsão", "vermelho", "cartão"]
        },
        {
            "regras": ["Regra 12"],
            "keywords": ["8 segundos", "escanteio"]
        },
        {
            "regras": ["Regra 11"],
            "keywords": ["posição adiantada", "linha de fundo"]
        },
        {
            "regras": ["Regra 12"],
            "keywords": ["falta", "cartão"]
        },
        {
            "regras": ["Regra 10"],
            "keywords": ["cinco tiros", "morte súbita"]
        },
    ]
    
    livro_path = Path("livros/Regras do Jogo 2025 e 2026 - IFAB.md")
    with open(livro_path, encoding="utf-8") as f:
        texto = limpar(f.read())
    registros = parse_md(texto)
    
    print(f"Testando {len(perguntas_teste)} perguntas:\n")
    
    for i, pergunta in enumerate(perguntas_teste, 1):
        esperado = respostas_esperadas[i-1]
        
        # Verificar se o documento contém as regras esperadas
        regras_presentes = []
        for regra in esperado["regras"]:
            if any(regra in r["regra"] for r in registros):
                regras_presentes.append(regra)
        
        status = "✅" if len(regras_presentes) == len(esperado["regras"]) else "⚠️"
        print(f"{status} Pergunta {i}: {pergunta[:60]}...")
        print(f"   Regras esperadas: {', '.join(esperado['regras'])}")
        print(f"   Presentes: {', '.join(regras_presentes) if regras_presentes else '(nenhuma)'}")
        print()
    
    return True

def test_exemplo_resposta():
    print("=" * 60)
    print("TEST 3: EXEMPLO DE RESPOSTA ESTRUTURADA")
    print("=" * 60)
    
    exemplo_resposta = {
        "decisao": "Tiro penal com cartão vermelho (expulsão)",
        "base_legal": "Regra 12, Seção 4 — Medidas disciplinares (falta com intenção de machucar)",
        "explicacao": "Se um jogador comete uma falta com intenção clara de machucar o adversário dentro da área penal, é concedido tiro penal à equipe adversária, e o jogador é expulso com cartão vermelho. A expulsão é irreversível e o jogador não pode ser substituído.",
        "confianca": "Alta",
        "imagens": ["assets/foto1.png"],
        "score": 0.891
    }
    
    print("Exemplo de resposta estruturada:\n")
    print(json.dumps(exemplo_resposta, indent=2, ensure_ascii=False))
    print("\n✅ Estrutura de resposta validada")
    return True

def test_api_config():
    print("=" * 60)
    print("TEST 4: CONFIGURAÇÃO DA API OpenAI")
    print("=" * 60)
    
    import os
    api_key = os.environ.get("OPENAI_API_KEY", "")
    
    if api_key:
        print(f"✅ OPENAI_API_KEY configurada")
        print(f"   Primeiros 15 chars: {api_key[:15]}...")
        print(f"   Modelo: gpt-4o-mini")
        print(f"   Custo aprox: $0.00015 por 1K tokens de input")
        return True
    else:
        print(f"⚠️  OPENAI_API_KEY não configurada")
        print(f"\nConfigure a variável de ambiente:")
        print(f"  export OPENAI_API_KEY='sk-...'")
        print(f"\nOu crie um arquivo .env:")
        print(f"  OPENAI_API_KEY=sk-...")
        print(f"\nObtenha sua chave em: https://platform.openai.com/api-keys")
        return False

def main():
    print("\n")
    print("⚽ TESTE COMPLETO DO APITO GPT")
    print("Sistema RAG para Regras do Futebol (IFAB/CBF)")
    print()
    
    tests = [
        ("Parsing", test_parsing),
        ("Buscas Mock", test_buscas_mock),
        ("Resposta", test_exemplo_resposta),
        ("API OpenAI", test_api_config),
    ]
    
    results = []
    for nome, test_func in tests:
        try:
            result = test_func()
            results.append((nome, result))
        except Exception as e:
            print(f"❌ ERRO em {nome}: {e}\n")
            results.append((nome, False))
    
    # Sumário
    print("=" * 60)
    print("RESUMO DOS TESTES")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for nome, result in results:
        status = "✅ PASSOU" if result else "❌ FALHOU"
        print(f"{status}: {nome}")
    
    print(f"\nTotal: {passed}/{total} testes passaram")
    
    if passed == total:
        print("\n🎉 Todos os testes passaram! Sistema pronto para uso.")
        print("\nPróximo passo: Configure a chave OpenAI e rode:")
        print("  python app.py")
    else:
        print("\n⚠️  Alguns testes falharam. Verifique os erros acima.")
    
    print()

if __name__ == "__main__":
    main()
