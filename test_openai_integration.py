#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_openai_integration.py — Teste da integração com API OpenAI

Modos:
1. MOCK: simula resposta sem chamar API (não gasta tokens)
2. REAL: chamada real à OpenAI (requer OPENAI_API_KEY e saldo na conta)

Uso:
  python test_openai_integration.py mock    # teste mock (padrão)
  python test_openai_integration.py real    # teste real com sua chave
"""

import sys
import os
import json
import requests
from pathlib import Path

def test_mock():
    """Testa a estrutura da resposta com mock (sem chamar API)."""
    print("=" * 70)
    print("TESTE MOCK — Simulação da Resposta OpenAI (sem custo)")
    print("=" * 70)
    print()
    
    # Simula a resposta JSON que a API retornaria
    mock_response = {
        "id": "chatcmpl-mock-test",
        "object": "chat.completion",
        "created": 1719076800,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps({
                        "decisao": "Tiro penal com cartão vermelho (expulsão do jogador)",
                        "base_legal": "Regra 12, Seção 4 — Medidas disciplinares (falta com intenção de machucar dentro da área)",
                        "explicacao": "Se um jogador comete uma falta com intenção de machucar o adversário dentro da área penal, é concedido tiro penal à equipe adversária e o jogador recebe cartão vermelho (expulsão). A expulsão é irreversível e o jogador não pode ser substituído no jogo. Se a falta foi tentativa legítima de jogar a bola mas excessivamente brusca, seria apenas tiro penal com cartão amarelo."
                    })
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 487,
            "completion_tokens": 68,
            "total_tokens": 555
        }
    }
    
    # Processa a resposta como o app.py faria
    try:
        texto = mock_response["choices"][0]["message"]["content"].strip()
        parsed = json.loads(texto)
        
        print("✅ Resposta parseada com sucesso:\n")
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
        
        # Calcula custo estimado (gpt-4o-mini: $0.15 por 1M input tokens, $0.60 por 1M output)
        prompt_tokens = mock_response["usage"]["prompt_tokens"]
        completion_tokens = mock_response["usage"]["completion_tokens"]
        
        custo_input = (prompt_tokens / 1_000_000) * 0.15
        custo_output = (completion_tokens / 1_000_000) * 0.60
        custo_total = custo_input + custo_output
        
        print(f"\n💰 Estimativa de custo (gpt-4o-mini):")
        print(f"   Input:  {prompt_tokens:,} tokens × $0.15/1M = ${custo_input:.6f}")
        print(f"   Output: {completion_tokens:,} tokens × $0.60/1M = ${custo_output:.6f}")
        print(f"   Total:  ${custo_total:.6f}")
        print(f"   Para 1000 consultas: ${custo_total * 1000:.2f}")
        
        print("\n✅ Mock test PASSOU")
        return True
        
    except Exception as e:
        print(f"❌ Erro ao processar resposta: {e}")
        return False

def test_real():
    """Testa chamada real à API OpenAI."""
    print("=" * 70)
    print("TESTE REAL — Chamada à API OpenAI (gasta tokens)")
    print("=" * 70)
    print()
    
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    
    if not api_key:
        print("❌ Erro: variável OPENAI_API_KEY não configurada")
        print("\nConfigure com:")
        print("  export OPENAI_API_KEY='sk-...'")
        print("\nOu crie um arquivo .env com:")
        print("  OPENAI_API_KEY=sk-...")
        print("\nObtenha sua chave em: https://platform.openai.com/api-keys")
        return False
    
    if not api_key.startswith("sk-"):
        print(f"❌ Erro: chave OpenAI inválida (deve começar com 'sk-')")
        return False
    
    print(f"🔑 Chave OpenAI detectada: {api_key[:15]}...")
    print()
    
    # Contexto simulado (similar ao que o RAG retornaria)
    contexto_exemplo = """[Regra 12 | Faltas e Condutas Incorretas | 4. Medidas disciplinares]
Um jogador que comete uma falta passível de advertência com cartão amarelo pode ser 
advertido com cartão amarelo. Se a infração foi uma conduta violenta ou impedimento 
de oportunidade clara de golo sem disputar a bola, o jogador é expulso com cartão vermelho.
Se o jogador cometer uma falta com intenção de machucar o adversário dentro da área penal,
o árbitro concede tiro penal e expulsa o jogador com cartão vermelho."""
    
    pergunta = "Se um jogador comete falta com intenção de machucar dentro da área, qual é a decisão?"
    
    prompt = f"""Você é um especialista nas Regras do Jogo de futebol da IFAB/CBF.
Responda à pergunta usando SOMENTE o contexto fornecido.

REGRAS DE RESPOSTA:
- NÃO invente informações nem use conhecimento externo.
- Responda EXCLUSIVAMENTE em JSON válido com os campos:
  - "decisao": a decisão arbitral aplicável
  - "base_legal": a(s) Regra(s) que fundamentam
  - "explicacao": explicação acessível (2-4 frases)

CONTEXTO:
{contexto_exemplo}

PERGUNTA:
{pergunta}

JSON:"""
    
    print(f"📝 Pergunta: {pergunta}")
    print(f"\n🚀 Enviando para OpenAI (gpt-4o-mini)...")
    print()
    
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "Responda em JSON válido."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
                "max_tokens": 500,
            },
            timeout=30,
        )
        
        data = response.json()
        
        # Verificar erros
        if "error" in data:
            error_msg = data["error"].get("message", str(data["error"]))
            print(f"❌ Erro da API: {error_msg}")
            
            if "invalid_request_error" in str(data["error"].get("type", "")):
                print(f"\nPossível causa: Chave inválida ou expirada")
            elif "rate_limit_error" in str(data["error"].get("type", "")):
                print(f"\nPossível causa: Rate limit atingido. Aguarde um momento.")
            
            return False
        
        if "choices" not in data:
            print(f"❌ Resposta inesperada: {data}")
            return False
        
        # Extrair resposta
        texto = data["choices"][0]["message"]["content"].strip()
        
        # Remover cercas de código se houver
        import re
        texto = re.sub(r"^```(?:json)?|```$", "", texto, flags=re.MULTILINE).strip()
        parsed = json.loads(texto)
        
        print("✅ Resposta recebida:\n")
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
        
        # Mostrar uso de tokens
        usage = data.get("usage", {})
        print(f"\n📊 Tokens utilizados:")
        print(f"   Input:  {usage.get('prompt_tokens', '?')} tokens")
        print(f"   Output: {usage.get('completion_tokens', '?')} tokens")
        print(f"   Total:  {usage.get('total_tokens', '?')} tokens")
        
        # Calcular custo
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        custo_input = (prompt_tokens / 1_000_000) * 0.15
        custo_output = (completion_tokens / 1_000_000) * 0.60
        custo_total = custo_input + custo_output
        
        print(f"\n💰 Custo dessa consulta: ${custo_total:.6f}")
        
        print("\n✅ Real test PASSOU")
        return True
        
    except requests.exceptions.Timeout:
        print("❌ Erro: Timeout ao conectar com OpenAI (excesso de latência)")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Erro de conexão: {e}")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ Erro ao parsear JSON: {e}")
        return False
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
        return False

def main():
    if len(sys.argv) > 1 and sys.argv[1].lower() == "real":
        modo = "real"
    else:
        modo = "mock"
    
    print()
    result = test_real() if modo == "real" else test_mock()
    print()
    
    if result:
        if modo == "mock":
            print("Para testar com a API real, rode:")
            print("  export OPENAI_API_KEY='sk-...'")
            print("  python test_openai_integration.py real")
    
    sys.exit(0 if result else 1)

if __name__ == "__main__":
    main()
