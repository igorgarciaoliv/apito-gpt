# -*- coding: utf-8 -*-
"""
APITO GPT — Interface Streamlit
================================
Executa: streamlit run streamlit_app.py
"""

import os
import html
import streamlit as st
from PIL import Image

# ── Page config (deve ser o primeiro comando Streamlit) ───────────────
st.set_page_config(
    page_title="APITO GPT",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Carregamento do motor RAG (cacheado — carrega UMA vez) ────────────
@st.cache_resource(show_spinner="Carregando modelos de IA...")
def carregar_motor():
    import app as rag
    # Delegado ao motor: além de criar o índice quando ausente, ele detecta
    # índice gerado por outro modelo de embedding e reindexa sozinho.
    with st.spinner("Preparando índice das Regras..."):
        rag._carregar_indices()
    return rag

rag = carregar_motor()

# ── CSS customizado ───────────────────────────────────────────────────
st.markdown("""
<style>
.card {
    background: #1e1e2e;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 12px;
    border-left: 4px solid #7c3aed;
}
.card-red   { border-left-color: #ef4444; }
.card-green { border-left-color: #22c55e; }
.card-blue  { border-left-color: #3b82f6; }
.card-amber { border-left-color: #f59e0b; }
.card-title { font-size: 0.78rem; font-weight: 700; letter-spacing: .08em;
              text-transform: uppercase; color: #94a3b8; margin-bottom: 6px; }
.card-body  { font-size: 1.05rem; color: #f1f5f9; }
.badge      { display: inline-block; padding: 2px 10px; border-radius: 999px;
              font-size: 0.8rem; font-weight: 600; }
.badge-green { background:#16a34a; color:#fff; }
.badge-red   { background:#dc2626; color:#fff; }
.badge-amber { background:#d97706; color:#fff; }
.badge-gray  { background:#475569; color:#fff; }
.warn-box   { background:#7c2d12; border-radius:8px; padding:10px 14px;
              color:#fed7aa; font-size:0.9rem; margin-top:8px; }
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚽ APITO GPT")
    st.caption("Consulta às Regras do Futebol (IFAB/CBF 2025/26)")
    st.divider()

    st.subheader("🔑 Sua chave da OpenAI")
    # A chave é de quem usa. Fica só na sessão do navegador: não é gravada em
    # disco, não entra na imagem e não é compartilhada com outras sessões.
    chave_usuario = st.text_input(
        "Chave da API",
        type="password",
        placeholder="sk-...",
        help="Usada apenas nesta sessão para consultar a OpenAI em seu nome. "
             "Não é armazenada em nenhum lugar.",
        label_visibility="collapsed",
    )
    if chave_usuario:
        st.caption("✅ Chave informada — válida nesta sessão.")
    elif rag.OPENAI_API_KEY:
        st.caption("Usando a chave configurada no servidor.")
    else:
        st.caption("Cole sua chave para consultar. "
                   "[Obter uma](https://platform.openai.com/api-keys)")

    st.divider()

    st.subheader("Exemplos")
    exemplos = [
        "Se um jogador comete falta com intenção de machucar dentro da área, qual é a decisão?",
        "O VAR pode inverter uma falta comum para cartão amarelo?",
        "O goleiro pode segurar a bola por quanto tempo?",
        "O que é impedimento?",
        "Quando é dado um carrinho por trás, o que acontece?",
        "Quais são os sinais do árbitro assistente?",
        "Como funciona a disputa de pênaltis?",
        "Um jogador erra ao chutar um pênalti, pode repetir?",
    ]
    exemplo_sel = st.radio("", exemplos, label_visibility="collapsed")

    st.divider()
    st.subheader("Sobre o sistema")
    # Lidos do motor para não divergirem da configuração real em app.py
    st.markdown(f"""
**Motor RAG** com:
- 🔍 Busca híbrida BM25 + vetorial (RRF)
- 🔁 Reranking cross-encoder
- ✅ Filtro de relevância adaptativo
- 🧩 Decomposição multi-regra
- 🛡️ Verificação de alucinação

Geração: `{rag.MODEL_NAME}`
Embedding: `{rag.EMBED_MODEL_NAME.split('/')[-1]}`
Reranker: `{rag.CROSS_ENCODER_NAME.split('/')[-1]}`
""")

# ── MAIN AREA ─────────────────────────────────────────────────────────
st.markdown("## ⚽ APITO GPT — Consulta às Regras do Futebol")
st.caption("Sistema RAG que responde dúvidas arbitrais com base no livro oficial da IFAB/CBF.")

col_input, col_btn = st.columns([5, 1])
with col_input:
    pergunta = st.text_area(
        "Sua pergunta",
        value=exemplo_sel,
        height=80,
        placeholder="Ex.: O que acontece se a bola bater no árbitro?",
        label_visibility="collapsed",
    )
with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    consultar = st.button("⚽ Consultar", use_container_width=True, type="primary")

st.divider()

# ── RESPOSTA ──────────────────────────────────────────────────────────
if consultar and pergunta.strip():
    chave_ativa = chave_usuario or rag.OPENAI_API_KEY
    if not chave_ativa:
        st.error(
            "Informe sua chave da OpenAI na barra lateral para consultar. "
            "Você pode gerar uma em platform.openai.com/api-keys."
        )
        st.stop()

    with st.spinner("Buscando nas Regras..."):
        resultado_busca = rag.buscar(pergunta, api_key=chave_ativa)

    try:
        with st.spinner("Gerando resposta..."):
            r = rag.gerar_resposta(pergunta, resultado_busca, api_key=chave_ativa)
    except Exception as e:
        st.error(f"Não foi possível consultar a OpenAI: {e}")
        st.stop()

    # Os cards usam HTML bruto, então todo texto vindo do LLM precisa ser
    # escapado — um "<" solto na explicação engoliria o resto do card.
    def esc(valor, padrao="-"):
        return html.escape(str(valor if valor not in (None, "") else padrao))

    col_left, col_right = st.columns(2)

    # ── Decisão ──────────────────────────────────────────────────────
    with col_left:
        st.markdown(f"""
<div class="card card-blue">
  <div class="card-title">⚖️ Decisão</div>
  <div class="card-body">{esc(r.get('decisao'))}</div>
</div>""", unsafe_allow_html=True)

    # ── Sanção ───────────────────────────────────────────────────────
    cartao = r.get("cartao", "Nenhum")
    if "vermelho" in cartao.lower():
        cartao_class, badge_class = "card-red", "badge-red"
    elif "amarelo" in cartao.lower():
        cartao_class, badge_class = "card-amber", "badge-amber"
    else:
        cartao_class, badge_class = "card-green", "badge-green"

    with col_right:
        st.markdown(f"""
<div class="card {cartao_class}">
  <div class="card-title">🟥 Sanção Disciplinar</div>
  <div class="card-body">
    <span class="badge {badge_class}">{esc(cartao, "Nenhum")}</span>
  </div>
</div>""", unsafe_allow_html=True)

    # ── Base Legal ───────────────────────────────────────────────────
    st.markdown(f"""
<div class="card card-green">
  <div class="card-title">📖 Base Legal</div>
  <div class="card-body">{esc(r.get('base_legal'))}</div>
</div>""", unsafe_allow_html=True)

    # ── Explicação ───────────────────────────────────────────────────
    st.markdown(f"""
<div class="card">
  <div class="card-title">💬 Explicação</div>
  <div class="card-body">{esc(r.get('explicacao'))}</div>
</div>""", unsafe_allow_html=True)

    # ── Confiança + fundamentação ─────────────────────────────────────
    conf  = r.get("confianca", "-")
    score = r.get("score", 0)
    fund  = r.get("fundamentado", True)
    fund_badge = ('<span class="badge badge-green">✅ Fundamentado</span>'
                  if fund else
                  '<span class="badge badge-red">⚠️ Verificar</span>')
    conf_badge_cls = {"Alta": "badge-green", "Média": "badge-amber"}.get(conf, "badge-gray")
    st.markdown(f"""
<div class="card card-amber">
  <div class="card-title">🎯 Confiança</div>
  <div class="card-body">
    <span class="badge {conf_badge_cls}">{conf}</span>&nbsp;
    <span style="color:#94a3b8; font-size:.85rem;">CE score: {score}</span>&nbsp;&nbsp;
    {fund_badge}
  </div>
</div>""", unsafe_allow_html=True)

    if not fund:
        st.markdown("""
<div class="warn-box">
⚠️ O verificador de alucinação sinalizou que partes da resposta podem não estar
totalmente fundamentadas no material. Consulte a base legal para confirmar.
</div>""", unsafe_allow_html=True)

    # ── Imagens ───────────────────────────────────────────────────────
    imagens_validas = []
    for img_path in r.get("imagens", []):
        for candidato in [img_path, os.path.join(rag.PASTA_ASSETS, os.path.basename(img_path))]:
            if os.path.exists(candidato):
                imagens_validas.append(candidato)
                break

    if imagens_validas:
        st.divider()
        st.markdown("#### 🖼️ Mídia associada")
        cols_img = st.columns(min(len(imagens_validas), 4))
        for i, img_path in enumerate(imagens_validas):
            try:
                with cols_img[i % 4]:
                    st.image(Image.open(img_path), use_container_width=True)
            except Exception:
                pass

    # ── Detalhes técnicos ─────────────────────────────────────────────
    debug = r.get("debug", {})
    with st.expander("🔬 Detalhes técnicos"):
        col_d1, col_d2, col_d3 = st.columns(3)
        col_d1.metric("Candidatos híbridos", debug.get("n_candidatos", "-"))
        col_d2.metric("Após reranking/grading", debug.get("n_apos_rerank", "-"))
        col_d3.metric("Chunks no contexto", debug.get("chunks_usados", "-"))

        subs = debug.get("sub_perguntas", [])
        if subs:
            st.markdown("**Sub-perguntas geradas (decomposição multi-regra):**")
            for i, sp in enumerate(subs, 1):
                st.markdown(f"{i}. {sp}")

        st.markdown("**Chunks utilizados:**")
        for c in resultado_busca.get("chunks", []):
            # Seções de apoio (Cenários Multi-Regra) não têm "regra" própria —
            # cai para "secao" em vez de mostrar em branco, senão o rótulo
            # some justamente nos chunks mais úteis para casos combinados.
            rotulo = c.get("regra") or c.get("secao") or "?"
            st.markdown(
                f"- `{rotulo} | {c.get('subsecao','?')[:60]}` "
                f"— CE: `{c.get('ce_score','?')}` · Híbrido: `{c.get('score','?')}`"
            )

elif consultar and not pergunta.strip():
    st.warning("Digite uma pergunta antes de consultar.")
