# ⚽ APITO GPT

Sistema de perguntas e respostas sobre as Regras do Jogo de futebol (IFAB/CBF
2025-2026). Você pergunta em linguagem natural — inclusive com gírias de campo —
e recebe a decisão arbitral, a sanção disciplinar, a base legal na Regra
correspondente e as figuras oficiais do trecho citado.

![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)
![Docker](https://img.shields.io/badge/Docker-igorgarciaoliveira%2Fapito--gpt-2496ED)
![Python](https://img.shields.io/badge/Python-3.11-3776AB)

---

## Rodando

Você precisa de uma chave da OpenAI — a sua própria. Ela é digitada na barra
lateral da interface, vale só para aquela sessão e não é gravada em lugar nenhum.

### Docker (recomendado)

```bash
docker run -d -p 8501:8501 -p 8000:8000 igorgarciaoliveira/apito-gpt:latest
```

Abra `http://localhost:8501`, cole sua chave na barra lateral e pergunte.

Na primeira execução o container indexa o livro de regras, o que leva alguns
minutos. Para não repetir isso a cada recriação, monte um volume:

```bash
docker run -d -p 8501:8501 -p 8000:8000 \
  -e APITO_INDEX_DIR=/app/index_data \
  -v apito_indices:/app/index_data \
  igorgarciaoliveira/apito-gpt:latest
```

### Local

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## Como funciona

O sistema é um RAG: em vez de confiar na memória do modelo de linguagem, ele
recupera os trechos aplicáveis do livro oficial e responde apoiado neles.

```
pergunta
   │
   ├─ expansão por glossário      "carrinho por trás" → "jogo brusco grave"
   ├─ decomposição multi-regra    separa perguntas que envolvem 2+ Regras
   │
   ├─ busca híbrida (RRF)         vetorial (e5-base) + BM25
   ├─ reranking                   cross-encoder reordena por pertinência
   ├─ filtro de relevância        corte adaptativo, nunca devolve vazio
   │
   ├─ geração (GPT-4o)            base legal restrita aos trechos recuperados
   ├─ verificação                 confere se a resposta se apoia no contexto
   └─ vínculo de mídia            figuras só da seção efetivamente citada
```

**Por que busca híbrida.** Só vetorial erra em termos literais e gírias; só BM25
erra em paráfrase. A fusão usa Reciprocal Rank Fusion, que combina as *posições*
de cada ranking — somar os scores brutos não funcionaria, porque o cosseno vive
entre 0,7 e 0,9 enquanto o BM25 normalizado vai de 0 a 1.

**Por que o vínculo de mídia.** As figuras só aparecem quando a seção que as
contém é a mesma citada na base legal. Sem isso, uma resposta sobre faltas vinha
acompanhada de diagramas de outra Regra.

**O selo de fundamentação.** Depois de gerar, o sistema confere se a resposta se
sustenta nos trechos recuperados. Quando não se sustenta, a interface avisa em
vez de apresentar a resposta como certa.

---

## Interface

A resposta é estruturada em campos separados — decisão, sanção, base legal e
explicação — em vez de um texto corrido, porque cada perfil de usuário lê um
pedaço diferente: o árbitro quer a decisão e o cartão; quem escreve sobre o lance
quer a base legal citável; quem está aprendendo quer a explicação.

O painel "Detalhes técnicos" mostra quantos candidatos a busca levantou, quantos
sobreviveram ao reranking e a pontuação de cada trecho usado.

---

## API REST

```bash
curl -X POST http://localhost:8000/perguntar \
  -H "Content-Type: application/json" \
  -d '{"pergunta": "O goleiro pode segurar a bola por quanto tempo?",
       "api_key": "sk-..."}'
```

```json
{
  "decisao": "Escanteio para a equipe adversária",
  "cartao": "Nenhum",
  "base_legal": "Regra 12 — 3. Escanteio",
  "explicacao": "O goleiro pode segurar a bola por até oito segundos...",
  "confianca": "Média",
  "fundamentado": true,
  "imagens": []
}
```

`GET /health` responde `{"status": "ok"}`.

---

## Configuração

| Variável | Padrão | Para quê |
|---|---|---|
| `OPENAI_API_KEY` | vazio | Chave do servidor. Se definida, dispensa o campo da interface |
| `APITO_INDEX_DIR` | `.` | Onde gravar o índice. Aponte a um volume para persistir |

Modelos e parâmetros de recuperação ficam no topo de `app.py`. Trocar o modelo
de embedding ou editar os livros dispara reindexação automática — o índice
carrega uma assinatura da configuração que o gerou.

---

## Estrutura

```
app.py             motor RAG e API FastAPI
streamlit_app.py   interface web
api.py             ponto de entrada do FastAPI
build.py           indexação standalone
livros/            livro de regras em Markdown
assets/            figuras extraídas do livro
Dockerfile         imagem com os modelos pré-baixados
```

---

## Créditos

As Regras do Jogo são publicadas pela [IFAB](https://www.theifab.com/) e adotadas
pela CBF. Este projeto é uma ferramenta de consulta e estudo; em caso de
divergência, o texto oficial prevalece.
