# -*- coding: utf-8 -*-
"""
APITO GPT — Motor RAG para Regras do Futebol (IFAB/CBF) v2
============================================================
Melhorias v2:
  1. Busca híbrida (BM25 + vetorial) — robustez em termos exatos
  2. Reranking com cross-encoder multilíngue — melhor ordenação
  3. Relevance grading via cross-encoder — filtra chunks irrelevantes
  4. Decomposição de query multi-regra + verificação de alucinação
"""

import os
import re
import json
import time
import pickle
import hashlib
import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from fastapi import FastAPI
from pydantic import BaseModel

# ── CONFIG ────────────────────────────────────────────────────────────

PASTA_LIVROS = "livros"
PASTA_ASSETS = "assets"

# Diretório dos índices. Configurável para que o container possa apontá-lo a um
# volume e não reindexar a cada recriação; o padrão "." mantém o comportamento
# local de sempre.
INDEX_DIR    = os.environ.get("APITO_INDEX_DIR", ".")
os.makedirs(INDEX_DIR, exist_ok=True)

INDEX_PATH   = os.path.join(INDEX_DIR, "index.faiss")
META_PATH    = os.path.join(INDEX_DIR, "meta.pkl")
BM25_PATH    = os.path.join(INDEX_DIR, "bm25.pkl")

# GPT-4o: melhor raciocínio normativo que o mini em regras condicionais
MODEL_NAME = "gpt-4o"

# Decompor a pergunta é reescrita simples e roda no modelo menor.
# A verificação de fundamentação NÃO: testada no mini, ela reprovava respostas
# corretas e bem ancoradas. Um selo de alerta que dispara no caso certo é pior
# que nenhum selo, porque ensina o usuário a ignorá-lo — então ela fica no
# modelo principal.
MODEL_AUXILIAR = "gpt-4o-mini"

# Cole a chave aqui — ou passe via variável de ambiente OPENAI_API_KEY
OPENAI_API_KEY_EMBUTIDA = ""
OPENAI_API_KEY = OPENAI_API_KEY_EMBUTIDA or os.environ.get("OPENAI_API_KEY", "")

# e5-base, e não a variante large: com o cross-encoder reordenando depois, o
# que a busca vetorial precisa entregar é recall dentro dos 30 candidatos, não
# precisão fina — a precisão vem do reranking. Em troca, o modelo ocupa cerca
# de metade da memória, o que decide entre o container subir ou morrer durante
# a indexação em máquinas com ~4 GB.
EMBED_MODEL_NAME = "intfloat/multilingual-e5-base"
CROSS_ENCODER_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

print(f"Carregando embeddings ({EMBED_MODEL_NAME})...")
EMBED_MODEL = SentenceTransformer(EMBED_MODEL_NAME)
print("Modelo de embedding pronto.")

_CROSS_ENCODER = None

def cross_encoder():
    """
    Carrega o cross-encoder sob demanda.

    A indexação usa apenas o modelo de embedding, mas é a etapa de maior pico
    de memória. Manter o reranker fora dela reduz esse pico em algumas centenas
    de MB e evita que o container morra em máquinas com pouca RAM — o custo é
    um atraso único na primeira consulta.
    """
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        print(f"Carregando cross-encoder ({CROSS_ENCODER_NAME})...")
        _CROSS_ENCODER = CrossEncoder(CROSS_ENCODER_NAME, max_length=512)
    return _CROSS_ENCODER

# Parâmetros de chunking
MAX_CHUNK_CHARS = 1200
OVERLAP_CHARS   = 200
MIN_CHUNK_CHARS = 50

# Parâmetros de recuperação
K_RETRIEVAL   = 30    # candidatos iniciais (vetorial + BM25)
RERANK_TOP_N  = 6     # chunks após cross-encoder
ALPHA         = 0.6   # peso do score vetorial na fusão híbrida

# Filtro de relevância do cross-encoder.
# Calibrado empiricamente (_teste_ce.py): em pares (pergunta, trecho) do corpus
# IFAB, trechos RELEVANTES pontuam de -1,7 a +4 e IRRELEVANTES de -9 a -2.
# Um corte absoluto único é frágil nessa faixa, então usamos três critérios:
#   1. piso absoluto — descarta lixo evidente;
#   2. margem relativa ao melhor chunk — adapta-se à dificuldade da pergunta;
#   3. mínimo garantido — nunca devolve vazio se houve candidatos.
CE_FLOOR_ABS  = -8.0  # abaixo disso é ruído em qualquer cenário
CE_MARGIN     = 5.0   # mantém chunks até (melhor_score - CE_MARGIN)
CE_MIN_CHUNKS = 2     # sempre entrega ao menos N chunks se existirem candidatos

# Seções de apoio do documento-fonte: ajudam a recuperar e a interpretar, mas
# não são texto normativo. Entram no contexto e ficam fora da base legal, que
# deve sempre apontar a Regra correspondente.
SECOES_NAO_NORMATIVAS = ("glossário", "glossario", "cenários multi-regra",
                         "cenarios multi-regra")

MAX_CONTEXT_CHARS = 4000

# ── UTIL ──────────────────────────────────────────────────────────────

def limpar(texto):
    return re.sub(r"\n{3,}", "\n\n", texto).strip()

def chunk_valido(texto):
    return len(texto) >= MIN_CHUNK_CHARS and texto.count(" ") > 10

def tokenizar(texto):
    return re.findall(r"\w+", texto.lower())

def gerar_embeddings(textos, tipo="passage"):
    # Lote pequeno de propósito: o ganho de velocidade de lotes maiores não
    # compensa o pico de memória durante a indexação, que é onde o container
    # chega perto do limite em máquinas modestas.
    textos_fmt = [f"{tipo}: {t}" for t in textos]
    emb = EMBED_MODEL.encode(textos_fmt, show_progress_bar=False, batch_size=8)
    emb = np.array(emb).astype("float32")
    faiss.normalize_L2(emb)
    return emb

def _segmentar(texto):
    """
    Quebra o texto em unidades menores que um chunk.

    Quebrar só em fim de frase não basta: listas como o glossário são itens sem
    ponto final, e o bloco inteiro sairia como uma unidade só. Por isso divide
    também por linha e por item de lista, e, se ainda restar um trecho maior
    que o limite, corta por palavras — assim nenhuma unidade estoura o teto.
    """
    partes = []
    for linha in re.split(r"\n+", texto):
        for frase in re.split(r"(?<=[.!?;])\s+", linha):
            frase = frase.strip()
            if not frase:
                continue
            if len(frase) <= MAX_CHUNK_CHARS:
                partes.append(frase)
                continue
            palavras, atual = frase.split(), ""
            for palavra in palavras:
                if len(atual) + len(palavra) + 1 <= MAX_CHUNK_CHARS:
                    atual += (" " if atual else "") + palavra
                else:
                    partes.append(atual)
                    atual = palavra
            if atual:
                partes.append(atual)
    return partes


def dividir_texto_grande(texto, max_chars=MAX_CHUNK_CHARS, overlap=OVERLAP_CHARS):
    if len(texto) <= max_chars:
        return [texto]

    chunks, atual = [], ""
    for parte in _segmentar(texto):
        if len(atual) + len(parte) + 1 <= max_chars:
            atual += (" " if atual else "") + parte
            continue
        if atual:
            chunks.append(atual.strip())
        if chunks and overlap > 0:
            cauda = chunks[-1][-overlap:]
            idx = cauda.find(" ")
            atual = (cauda[idx + 1:] if idx > 0 else cauda) + " " + parte
        else:
            atual = parte
    if atual.strip():
        chunks.append(atual.strip())
    return chunks

# ── PARSER + INDEXAÇÃO ────────────────────────────────────────────────

_FRONTMATTER = "__frontmatter__"

def parse_md(texto):
    linhas = texto.split("\n")
    # Sentinela: tudo antes do primeiro cabeçalho (título, editora, data de
    # vigência) é metadado do documento, não conteúdo normativo. Sem essa
    # marca, esse texto virava um chunk indexável com regra/subseção vazios —
    # citável como base legal e sem rótulo nenhum.
    secao, subsecao, regra_atual = _FRONTMATTER, "", ""
    niveis = {}  # nível do cabeçalho markdown → título vigente
    buffer, imagens_buffer, registros = [], [], []

    def salvar():
        if not buffer or secao == _FRONTMATTER:
            return
        # Junta preservando a quebra de linha: em listas (glossário, tabelas de
        # infrações) a linha é a única fronteira natural, e colapsá-las em um
        # parágrafo único impediria a divisão em chunks do tamanho certo.
        conteudo = "\n".join(buffer).strip()
        if not chunk_valido(conteudo):
            return
        pedacos = dividir_texto_grande(conteudo)
        for i, pedaco in enumerate(pedacos):
            sufixo = f" (parte {i+1}/{len(pedacos)})" if len(pedacos) > 1 else ""
            contexto = (
                f"Regra: {regra_atual} | Seção: {secao} | "
                f"Subseção: {subsecao}{sufixo} | Texto: {pedaco}"
            )
            registros.append({
                "regra": regra_atual,
                "secao": secao,
                "subsecao": subsecao + sufixo,
                "imagens": list(imagens_buffer),
                "conteudo_original": pedaco,
                "conteudo_embed": contexto,
            })

    for linha in linhas:
        ls = linha.strip()
        if not ls:
            continue
        img = re.findall(r"!\[.*?\]\((.*?)\)", ls)
        if img:
            imagens_buffer.extend(img)
            continue
        cabecalho = re.match(r"^(#{1,6})\s+(.*)$", ls)
        if cabecalho:
            nivel, titulo = len(cabecalho.group(1)), cabecalho.group(2).strip()
            salvar(); buffer.clear(); imagens_buffer.clear()

            if nivel == 1:
                # O único H1 do documento é o título de capa (editora, data de
                # vigência) — mantém a sentinela até o primeiro "##" real,
                # senão o próprio título vira "secao" e libera o corpo do
                # preâmbulo para ser indexado como se fosse conteúdo normativo.
                secao = _FRONTMATTER
            elif nivel == 2:
                secao = titulo
                niveis.clear()
                m = re.match(r"Regra (\d+):", secao)
                if m:
                    regra_atual = f"Regra {m.group(1)}"
            else:
                # Um cabeçalho SUBSTITUI o irmão de mesmo nível e invalida os
                # mais profundos. Concatenar sem limpar faria subseções irmãs
                # se acumularem ("A - B - C" onde o correto é "A - C").
                niveis[nivel] = titulo
                for mais_fundo in [n for n in niveis if n > nivel]:
                    del niveis[mais_fundo]

            subsecao = ""
            for n in sorted(niveis):
                if not subsecao:
                    subsecao = niveis[n]
                else:
                    subsecao += (" - " if n == 4 else " » ") + niveis[n]
        elif ls.startswith(">"):
            buffer.append(ls.lstrip("> "))
        else:
            buffer.append(ls)
    salvar()
    return registros

def criar_base():
    textos, metadados = [], []
    for arquivo in sorted(os.listdir(PASTA_LIVROS)):
        if not arquivo.endswith(".md"):
            continue
        path = os.path.join(PASTA_LIVROS, arquivo)
        print(f"Indexando: {arquivo}")
        for r in parse_md(limpar(open(path, encoding="utf-8").read())):
            textos.append(r["conteudo_embed"])
            metadados.append(r)

    if not textos:
        print("Nenhum conteúdo encontrado.")
        return

    # Índice vetorial (FAISS)
    embeddings = gerar_embeddings(textos, "passage")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)

    # Índice BM25 (sobre conteúdo original, sem prefixo de metadados)
    textos_bm25 = [r["conteudo_original"] for r in metadados]
    corpus_tok  = [tokenizar(t) for t in textos_bm25]
    bm25 = BM25Okapi(corpus_tok)
    with open(BM25_PATH, "wb") as f:
        pickle.dump((bm25, textos_bm25), f)

    # A assinatura viaja junto do índice para que o próximo start detecte
    # sozinho se os livros ou o modelo mudaram desde esta indexação.
    with open(META_PATH, "wb") as f:
        pickle.dump({"assinatura": _assinatura_corpus(), "chunks": metadados}, f)

    print(f"Base criada: {len(metadados)} chunks")

# ── GLOSSÁRIO + QUERY EXPANSION ───────────────────────────────────────

_GLOSSARIO_CACHE = None

def carregar_glossario():
    global _GLOSSARIO_CACHE
    if _GLOSSARIO_CACHE is not None:
        return _GLOSSARIO_CACHE
    mapa = {}
    try:
        for arquivo in os.listdir(PASTA_LIVROS):
            if not arquivo.endswith(".md"):
                continue
            for linha in open(os.path.join(PASTA_LIVROS, arquivo), encoding="utf-8"):
                if "significa" not in linha or '"' not in linha:
                    continue
                termos = re.findall(r'"([^"]+)"', linha)
                desc = linha.split("significa", 1)[1].strip(" .-\n")
                for t in termos:
                    mapa[t.lower().strip()] = desc
    except Exception:
        pass
    _GLOSSARIO_CACHE = mapa
    return mapa

def expandir_query(pergunta):
    glossario = carregar_glossario()
    if not glossario:
        return pergunta
    lower = pergunta.lower()
    expansoes = []
    for termo in sorted(glossario, key=len, reverse=True):
        if termo in lower:
            desc = glossario[termo]
            if desc not in expansoes:
                expansoes.append(desc)
    if not expansoes:
        return pergunta
    return pergunta + " | Termos oficiais: " + "; ".join(expansoes)

# ── 4. DECOMPOSIÇÃO DE QUERY MULTI-REGRA ──────────────────────────────

def precisa_decomposicao(pergunta):
    """Heurística: indica se a query provavelmente envolve múltiplas regras."""
    p = pergunta.lower()
    indicadores = [
        "além disso", "também", "ao mesmo tempo", "simultaneamente",
        "e também", "e além", "e o que acontece", "e como fica",
        "e qual é", "e quais são", "além de",
    ]
    return len(pergunta) > 70 and any(ind in p for ind in indicadores)

def decompor_query(pergunta, api_key=None):
    """LLM decompõe query complexa em sub-perguntas independentes."""
    prompt = (
        "Você é um analisador de perguntas sobre regras de futebol.\n"
        "Decomponha a pergunta abaixo em 2 ou 3 sub-perguntas independentes "
        "que, respondidas juntas, respondam à pergunta original.\n"
        "Se a pergunta for simples, retorne apenas ela numa lista de 1 elemento.\n"
        'Responda SOMENTE em JSON: {"sub_perguntas": ["...", "..."]}\n\n'
        f"PERGUNTA: {pergunta}\nJSON:"
    )
    try:
        resp = _llm_call(prompt, max_tokens=200, modelo=MODEL_AUXILIAR, api_key=api_key)
        data = json.loads(re.sub(r"^```(?:json)?|```$", "", resp, flags=re.MULTILINE).strip())
        subs = data.get("sub_perguntas", [pergunta])
        return [s for s in subs if s.strip()] or [pergunta]
    except Exception:
        return [pergunta]

# ── 1. BUSCA HÍBRIDA (BM25 + VETORIAL) ───────────────────────────────

_index_cache    = None
_meta_cache     = None
_bm25_cache     = None
_bm25txt_cache  = None

# Incrementar sempre que a LÓGICA de parse_md/dividir_texto_grande mudar —
# os parâmetros numéricos e o conteúdo dos livros já entram na assinatura
# abaixo, mas uma mudança de comportamento do parser com os mesmos parâmetros
# e o mesmo texto-fonte não alteraria o hash, e um volume persistente
# continuaria servindo o índice antigo com o bug já corrigido no código.
PARSER_VERSION = 2

def _assinatura_corpus():
    """
    Identifica a configuração que gerou o índice: versão do parser, modelo de
    embedding, regras de chunking e conteúdo dos livros. Qualquer mudança
    nesses itens invalida os vetores existentes.
    """
    h = hashlib.sha256()
    h.update(f"parser={PARSER_VERSION}".encode())
    h.update(EMBED_MODEL_NAME.encode())
    h.update(f"{MAX_CHUNK_CHARS}/{OVERLAP_CHARS}/{MIN_CHUNK_CHARS}".encode())
    for arquivo in sorted(os.listdir(PASTA_LIVROS)):
        if arquivo.endswith(".md"):
            caminho = os.path.join(PASTA_LIVROS, arquivo)
            h.update(arquivo.encode())
            with open(caminho, "rb") as f:
                h.update(f.read())
    return h.hexdigest()


def _indice_compativel():
    """
    Confere se o índice em disco corresponde à configuração atual.

    Duas checagens: a dimensão dos vetores, porque trocar de modelo faria o
    FAISS comparar vetores de espaços diferentes e devolver vizinhos sem
    sentido; e a assinatura do corpus, porque editar um livro sem reindexar
    deixaria o sistema respondendo pelo texto antigo.
    """
    try:
        if faiss.read_index(INDEX_PATH).d != EMBED_MODEL.get_sentence_embedding_dimension():
            print("Índice gerado por outro modelo de embedding — reindexando.")
            return False
        with open(META_PATH, "rb") as f:
            if pickle.load(f).get("assinatura") != _assinatura_corpus():
                print("Livros ou parâmetros de chunking mudaram — reindexando.")
                return False
        return True
    except Exception as e:
        print(f"Índice ilegível ({e}) — reindexando.")
        return False


def _carregar_indices():
    global _index_cache, _meta_cache, _bm25_cache, _bm25txt_cache
    if _index_cache is not None:
        return _index_cache, _meta_cache, _bm25_cache, _bm25txt_cache

    faltando = not all(os.path.exists(p)
                       for p in (INDEX_PATH, META_PATH, BM25_PATH))
    if faltando or not _indice_compativel():
        criar_base()

    _index_cache = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        _meta_cache = pickle.load(f)["chunks"]
    with open(BM25_PATH, "rb") as f:
        _bm25_cache, _bm25txt_cache = pickle.load(f)
    return _index_cache, _meta_cache, _bm25_cache, _bm25txt_cache

RRF_K = 60  # constante de amortecimento da Reciprocal Rank Fusion

def buscar_hibrido(pergunta_expandida, k=K_RETRIEVAL):
    """
    Funde recuperação vetorial e BM25 por Reciprocal Rank Fusion ponderada.

    Fusão por soma de scores brutos não funciona aqui: o cosseno do FAISS vive
    em ~0,7–0,9 enquanto o BM25 normalizado vai de 0 a 1, e um candidato achado
    só por um dos métodos recebia 0 do outro. O resultado penalizava o BM25 —
    justamente o componente que recupera termos informais e citações literais.
    A RRF usa apenas a POSIÇÃO em cada ranking, o que é imune a escala.

    Retorna lista de (idx, score_fundido, score_vetorial, score_bm25).
    """
    index, meta, bm25, _ = _carregar_indices()
    n = len(meta)
    if n == 0:
        return []

    # Ranking vetorial
    emb  = gerar_embeddings([pergunta_expandida], "query")
    D, I = index.search(emb, min(k, n))
    faiss_scores, rank_vec = {}, {}
    for posicao, idx in enumerate(I[0]):
        if idx < 0:
            continue
        idx = int(idx)
        faiss_scores[idx] = float(D[0][posicao])
        rank_vec[idx]     = posicao

    # Ranking BM25
    bm25_raw = bm25.get_scores(tokenizar(pergunta_expandida))
    top_bm25 = np.argsort(bm25_raw)[::-1][:k]
    rank_bm  = {int(idx): posicao for posicao, idx in enumerate(top_bm25)
                if bm25_raw[idx] > 0}

    # Normaliza BM25 apenas para exibição/diagnóstico
    bm25_max = float(bm25_raw.max())
    def bm25_exibicao(idx):
        return float(bm25_raw[idx]) / bm25_max if bm25_max > 0 else 0.0

    resultados = []
    for idx in set(rank_vec) | set(rank_bm):
        fundido = 0.0
        if idx in rank_vec:
            fundido += ALPHA * (1.0 / (RRF_K + rank_vec[idx]))
        if idx in rank_bm:
            fundido += (1 - ALPHA) * (1.0 / (RRF_K + rank_bm[idx]))
        resultados.append((
            idx,
            fundido,
            faiss_scores.get(idx, 0.0),
            bm25_exibicao(idx),
        ))

    resultados.sort(key=lambda x: x[1], reverse=True)
    return resultados[:k]

# ── 2+3. RERANKING + RELEVANCE GRADING (cross-encoder) ───────────────

def rerankear_e_filtrar(pergunta_original, candidatos):
    """
    Cross-encoder pontua pares (pergunta, chunk) e reordena.

    O corte de relevância é adaptativo: piso absoluto + margem relativa ao
    melhor chunk, com um mínimo garantido para nunca devolver contexto vazio
    quando havia candidatos (ver comentário em CE_FLOOR_ABS).
    """
    _, meta, _, _ = _carregar_indices()
    candidatos = [c for c in candidatos if c[0] < len(meta)]
    if not candidatos:
        return []

    pares = [(pergunta_original, meta[idx]["conteudo_original"])
             for idx, *_ in candidatos]
    scores = cross_encoder().predict(pares, show_progress_bar=False)

    pontuados = []
    for (idx, combined, f_sc, b_sc), ce_score in zip(candidatos, scores):
        item = dict(meta[idx])
        item["score"]    = round(float(combined), 3)
        item["ce_score"] = round(float(ce_score), 3)
        pontuados.append((float(ce_score), item))

    pontuados.sort(key=lambda x: x[0], reverse=True)
    melhor = pontuados[0][0]
    corte  = max(CE_FLOOR_ABS, melhor - CE_MARGIN)

    aprovados = [item for sc, item in pontuados if sc >= corte]
    if len(aprovados) < CE_MIN_CHUNKS:
        aprovados = [item for _, item in pontuados[:CE_MIN_CHUNKS]]

    return aprovados[:RERANK_TOP_N]

# ── ORQUESTRADOR DE BUSCA ─────────────────────────────────────────────

def limitar_contexto(contextos):
    """
    Preenche o orçamento de contexto respeitando a ordem de relevância.

    Um chunk que não cabe é pulado, não encerra o laço: interromper na primeira
    peça grande descartaria chunks menores e mais relevantes logo atrás dela.
    O chunk mais bem ranqueado entra sempre, mesmo que sozinho estoure o limite,
    para que a resposta nunca perca sua principal fundamentação.
    """
    total, final = 0, []
    for c in contextos:
        t = len(c["conteudo_original"])
        if not final:
            final.append(c)
            total += t
            continue
        if total + t <= MAX_CONTEXT_CHARS:
            final.append(c)
            total += t
    return final

def buscar(pergunta, api_key=None):
    """Pipeline completo: expand → decompose → hybrid → rerank+grade → limit."""
    # Expansão via glossário
    pergunta_exp = expandir_query(pergunta)

    # Decomposição multi-regra (item 4)
    if precisa_decomposicao(pergunta):
        sub_perguntas = decompor_query(pergunta, api_key=api_key)
    else:
        sub_perguntas = [pergunta]

    # Busca híbrida para cada sub-pergunta, merge por max score
    candidatos_map = {}  # idx → (idx, combined, f_sc, b_sc)
    for sp in sub_perguntas:
        sp_exp = expandir_query(sp) if sp != pergunta else pergunta_exp
        for idx, combined, f_sc, b_sc in buscar_hibrido(sp_exp, K_RETRIEVAL):
            if idx not in candidatos_map or combined > candidatos_map[idx][1]:
                candidatos_map[idx] = (idx, combined, f_sc, b_sc)

    candidatos = sorted(candidatos_map.values(), key=lambda x: x[1], reverse=True)[:K_RETRIEVAL]

    # Reranking + relevance grading (itens 2+3)
    chunks = rerankear_e_filtrar(pergunta, candidatos)

    # Limita contexto
    resultado = limitar_contexto(chunks)

    # Metadados de diagnóstico
    return {
        "chunks": resultado,
        "sub_perguntas": sub_perguntas if len(sub_perguntas) > 1 else [],
        "n_candidatos": len(candidatos),
        "n_apos_rerank": len(chunks),
    }

# ── LLM HELPER ────────────────────────────────────────────────────────

def _llm_call(prompt, max_tokens=600,
              system="Responda apenas com base no contexto, em JSON válido.",
              tentativas=3, modelo=None, api_key=None):
    """
    Chama a API de chat com backoff. Falhas transitórias (429, 5xx, timeout)
    são reprocessadas; erros de requisição — chave inválida, modelo inexistente
    — sobem de imediato, pois repetir não muda o resultado.

    `api_key` é recebida por parâmetro, e não lida de um estado global, porque
    cada usuário usa a própria chave: num servidor compartilhado, guardar a
    chave em variável de módulo faria a de um vazar para a consulta do outro.
    """
    chave = api_key or OPENAI_API_KEY
    if not chave:
        raise ValueError(
            "Nenhuma chave da OpenAI informada. Preencha o campo na barra "
            "lateral ou defina a variável de ambiente OPENAI_API_KEY."
        )

    ultimo_erro = None
    for tentativa in range(tentativas):
        try:
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {chave}",
                         "Content-Type": "application/json"},
                json={
                    "model": modelo or MODEL_NAME,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user",   "content": prompt},
                    ],
                    "temperature": 0.0,
                    "max_tokens": max_tokens,
                },
                timeout=60,
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                ultimo_erro = ValueError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                time.sleep(2 ** tentativa)
                continue

            data = resp.json()
            if "error" in data:
                raise ValueError(data["error"].get("message", str(data["error"])))
            if "choices" not in data:
                raise ValueError(f"Resposta inesperada: {data}")
            return data["choices"][0]["message"]["content"].strip()

        except requests.exceptions.RequestException as e:
            ultimo_erro = e
            time.sleep(2 ** tentativa)

    raise ValueError(f"Falha ao chamar a API após {tentativas} tentativas: {ultimo_erro}")

# ── VERIFICAÇÃO DE ALUCINAÇÃO ─────────────────────────────────────────

def verificar_alucinacao(contexto_txt, resposta, api_key=None):
    """
    Retorna True se a resposta está fundamentada no contexto.

    Recebe o contexto inteiro que foi entregue ao gerador. Truncar aqui produz
    falsos alertas: uma resposta apoiada num trecho mais ao final seria acusada
    de infundada apenas por a evidência ter ficado fora do recorte.
    """
    prompt = (
        "Verifique se a RESPOSTA contradiz o CONTEXTO ou afirma algo que não "
        "se sustenta nele.\n\n"
        "Responda 'sim' (está fundamentada) nestes casos:\n"
        "- a resposta COMBINA informações de trechos diferentes do contexto;\n"
        "- a resposta conclui POR EXCLUSÃO a partir de uma lista fechada;\n"
        "- a resposta reformula o texto normativo em linguagem acessível.\n\n"
        "Responda 'não' apenas se a resposta contradisser o contexto ou "
        "introduzir regra, número ou sanção que não aparece nele.\n\n"
        f"CONTEXTO:\n{contexto_txt}\n\n"
        f"RESPOSTA: {resposta}\n\nResponda apenas 'sim' ou 'não':"
    )
    try:
        r = _llm_call(prompt, max_tokens=5, api_key=api_key,
                      system="Você verifica fundamentação factual. Responda só 'sim' ou 'não'.")
        return "sim" in r.lower()
    except Exception:
        return True  # benefício da dúvida em caso de erro

# ── NÍVEL DE CONFIANÇA ────────────────────────────────────────────────

def cita_regra(regra, texto):
    """
    Verifica se `texto` cita `regra` de fato.

    Comparação por substring simples não serve: "Regra 1" está contido em
    "Regra 12", "Regra 14" e "Regra 17", o que validaria uma base legal errada
    e faria imagens vazarem entre regras diferentes. Exige limite de palavra
    após o número.
    """
    if not regra or not texto:
        return False
    return re.search(rf"{re.escape(regra)}(?!\d)", texto) is not None


def nivel_confianca(ce_score):
    """
    Faixas derivadas da calibração do cross-encoder no corpus IFAB:
    trechos claramente relevantes ficam acima de +2, plausíveis entre -2 e +2,
    e abaixo de -2 a correspondência costuma ser apenas temática.
    """
    if ce_score >= 2:
        return "Alta"
    if ce_score >= -2:
        return "Média"
    return "Baixa"

# ── GERAÇÃO DE RESPOSTA ───────────────────────────────────────────────

def gerar_resposta(pergunta, resultado_busca, api_key=None):
    contextos    = resultado_busca["chunks"]
    sub_perguntas = resultado_busca.get("sub_perguntas", [])

    if not contextos:
        return {
            "decisao":      "Não encontrado no material",
            "cartao":       "-",
            "base_legal":   "-",
            "explicacao":   "Nenhum trecho relevante encontrado. Tente reformular com termos das Regras do Jogo.",
            "confianca":    "Baixa",
            "fundamentado": True,
            "imagens":      [],
            "debug":        {"sub_perguntas": sub_perguntas, "chunks_usados": 0},
        }

    contexto_txt    = ""
    refs_disponiveis = []
    chunks_info     = []

    for c in contextos:
        contexto_txt += f"[{c['regra']} | {c['secao']} | {c['subsecao']}]\n{c['conteudo_original']}\n\n"
        secao_lower = (c.get("secao") or "").lower()
        if any(s in secao_lower for s in SECOES_NAO_NORMATIVAS):
            continue
        ref = c.get("regra") or c.get("secao") or ""
        sub = (c.get("subsecao") or "").split(" (parte ")[0].strip()
        if sub:
            ref = f"{ref} — {sub}" if ref else sub
        if ref and ref not in refs_disponiveis:
            refs_disponiveis.append(ref)
        chunks_info.append((ref, c.get("ce_score", 0), c.get("imagens", []), sub))

    melhor_ce = max((c.get("ce_score", 0) for c in contextos), default=0)
    refs_txt  = "\n".join(f"- {r}" for r in refs_disponiveis) or "- (nenhuma)"

    prompt = f"""Você é um especialista nas Regras do Jogo de futebol da IFAB/CBF.
Responda à pergunta usando SOMENTE o contexto fornecido.

REGRAS DE RESPOSTA:
- NÃO invente informações nem use conhecimento externo.
- RACIOCÍNIO POR EXCLUSÃO: se o contexto trouxer uma LISTA FECHADA/EXAUSTIVA (ex.:
  "o VAR só pode intervir nas seguintes situações: ...") e a pergunta perguntar por
  um item que NÃO está nessa lista, a resposta é "Não se aplica" — explique citando
  a lista. NÃO diga "Não encontrado no material" nesse caso.
- Se a resposta realmente não estiver no contexto, escreva "Não encontrado no material".
- SANÇÃO DISCIPLINAR (campo cartao) — avalie SEMPRE para perguntas sobre faltas:
    • Falta imprudente            → "Nenhum"
    • Falta temerária             → "Cartão amarelo"
    • Força excessiva / jogo brusco grave / conduta violenta / intenção de machucar
                                  → "Cartão vermelho (expulsão)"
    • Sem falta envolvida         → "Nenhum"
- COERÊNCIA: cartao e explicacao NÃO podem se contradizer.
- CONDICIONALIDADE: se a gravidade da infração depende da intensidade, explique
  a condição em vez de afirmar uma única punição.
- SEÇÕES DE APOIO NÃO SÃO BASE LEGAL: o Glossário e os Cenários Multi-Regra
  ajudam a interpretar, mas a base legal é sempre a Regra correspondente
  (ex.: Regra 12). Quando um cenário combinado citar as Regras que ele une,
  use essas Regras na base legal.
- BASE LEGAL: use EXCLUSIVAMENTE as referências listadas em REFERÊNCIAS DISPONÍVEIS.
  Jamais invente números de seção ou artigo fora dessa lista.

REFERÊNCIAS DISPONÍVEIS (use apenas estas em base_legal):
{refs_txt}

Responda EXCLUSIVAMENTE em JSON válido, sem texto antes ou depois:
  - "decisao":    decisão arbitral / reinício de jogo (curta e objetiva)
  - "cartao":     "Nenhum" | "Cartão amarelo" | "Cartão vermelho (expulsão)"
  - "base_legal": referência(s) da lista acima
  - "explicacao": explicação acessível (2-4 frases)

EXEMPLO (formato esperado):
PERGUNTA: Um jogador segura o adversário com a mão para impedir um ataque fora da área.
JSON:
{{"decisao": "Tiro livre direto para a equipe adversária", "cartao": "Cartão amarelo", "base_legal": "Regra 12 — 1. Tiro livre direto", "explicacao": "Segurar um adversário é infração passível de tiro livre direto. Por interromper um ataque promissor de forma antidesportiva, o infrator é advertido com cartão amarelo. Como ocorreu fora da área, não há tiro penal."}}

CONTEXTO:
{contexto_txt}

PERGUNTA:
{pergunta}

JSON:"""

    try:
        texto  = _llm_call(prompt, max_tokens=600, api_key=api_key)
        texto  = re.sub(r"^```(?:json)?|```$", "", texto, flags=re.MULTILINE).strip()
        parsed = json.loads(texto)
    except Exception as e:
        parsed = {
            "decisao":    "Ver explicação",
            "cartao":     "Indeterminado",
            "base_legal": "; ".join(refs_disponiveis) or "-",
            "explicacao": f"(Erro ao processar com LLM) {str(e)[:120]}",
        }

    parsed.setdefault("cartao", "Nenhum")

    # Anti-alucinação: base_legal só pode citar regras realmente recuperadas
    regras_validas = {c["regra"] for c in contextos if c.get("regra")}
    base_legal = str(parsed.get("base_legal", "")).strip()
    if regras_validas and not any(cita_regra(rv, base_legal) for rv in regras_validas):
        parsed["base_legal"] = "; ".join(refs_disponiveis) or "-"
        base_legal = parsed["base_legal"]

    # Aterrar mídia na base legal: exibe apenas imagens dos trechos que
    # efetivamente fundamentam a resposta. Sem esse vínculo, figuras de seções
    # não relacionadas apareciam junto de respostas sobre outro assunto.
    #
    # O casamento é por prefixo em ambos os sentidos porque a citação costuma
    # ser mais curta que a referência do chunk: a base legal diz "Regra 10 —
    # 3. Disputa de pênaltis" enquanto o chunk é "...  - Procedimento » Antes".
    # Comparar só por igualdade descartaria a figura da própria seção citada.
    citadas = [c.strip() for c in re.split(r"[;,]", base_legal) if c.strip()]

    def prefixo_de(curto, longo):
        # "Regra 1" NÃO pode valer como prefixo de "Regra 12 — ...", senão o
        # casamento por número volta a confundir regras diferentes.
        return longo.startswith(curto) and not longo[len(curto):len(curto) + 1].isdigit()

    def fundamenta(ref):
        return any(prefixo_de(cit, ref) or prefixo_de(ref, cit) for cit in citadas)

    imagens = []
    for ref, ce_sc, imgs, sub in chunks_info:
        if imgs and ref and fundamenta(ref):
            imagens.extend(imgs)

    imagens = list(dict.fromkeys(imagens))

    # Verificação de alucinação (item 4)
    resumo_resp = f"{parsed.get('decisao','')} {parsed.get('base_legal','')} {parsed.get('explicacao','')}"
    fundamentado = verificar_alucinacao(contexto_txt, resumo_resp, api_key=api_key)

    parsed["confianca"]    = nivel_confianca(melhor_ce)
    parsed["fundamentado"] = fundamentado
    parsed["imagens"]      = imagens
    parsed["score"]        = round(melhor_ce, 3)
    parsed["debug"]        = {
        "sub_perguntas":  sub_perguntas,
        "chunks_usados":  len(contextos),
        "n_candidatos":   resultado_busca.get("n_candidatos", 0),
        "n_apos_rerank":  resultado_busca.get("n_apos_rerank", 0),
    }
    return parsed

# ── FASTAPI ───────────────────────────────────────────────────────────

app = FastAPI(title="APITO GPT API v2")

class Pergunta(BaseModel):
    pergunta: str
    # Cada chamador usa a própria chave. Só cai na variável de ambiente quando
    # o campo vem vazio, o que cobre quem hospeda a própria instância.
    api_key: str | None = None


@app.post("/perguntar")
def perguntar_api(data: Pergunta):
    resultado = buscar(data.pergunta, api_key=data.api_key)
    return gerar_resposta(data.pergunta, resultado, api_key=data.api_key)

@app.get("/health")
def health():
    return {"status": "ok"}
