# 🚀 SETUP do APITO GPT

Guia completo de instalação e configuração do sistema RAG para Regras do Futebol.

## ✅ Pré-requisitos

- Python 3.8+
- pip (gerenciador de pacotes Python)
- Conta OpenAI com saldo de créditos (https://platform.openai.com/)

## 📥 Instalação

### 1. Clonar ou extrair o projeto

```bash
cd apito_gpt_final/
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

Isso instala:
- `faiss-cpu` — índice vetorial ultra-rápido
- `sentence-transformers` — embeddings multilíngues
- `gradio` — interface web interativa
- `fastapi` + `uvicorn` — API REST
- `openai` — SDK da OpenAI (já incluído no requests)

### 3. Configurar a chave OpenAI

**Opção A: Variável de ambiente (Linux/Mac)**

```bash
export OPENAI_API_KEY='sk-...'
```

**Opção B: Arquivo .env (todos os SOs)**

```bash
cp .env.example .env
# edite .env e preencha sua chave
cat .env
```

**Opção C: Permanent (Linux/Mac)**

Adicione ao `~/.bashrc` ou `~/.zshrc`:

```bash
export OPENAI_API_KEY='sk-...'
```

## 🧪 Testar a instalação

### Test 1: Validação completa do sistema

```bash
python test_apito_gpt.py
```

Esperado: 3/4 testes passam (o 4º avisa sobre a chave OpenAI)

```
✅ PASSOU: Parsing
✅ PASSOU: Buscas Mock
✅ PASSOU: Resposta
⚠️  API OpenAI (aviso esperado se chave não configurada)
```

### Test 2: Teste mock da API OpenAI (sem custos)

```bash
python test_openai_integration.py mock
```

Esperado: mostra estrutura de resposta e custo estimado (~$0.0001 por consulta)

### Test 3: Teste real da API OpenAI (gasta tokens)

⚠️  **Cuidado: isso gasta saldo da sua conta OpenAI**

```bash
export OPENAI_API_KEY='sk-...'
python test_openai_integration.py real
```

Esperado: resposta estruturada com custo real

## 🎯 Rodar o APITO GPT

Após validar com os testes, inicie o sistema:

```bash
python app.py
```

A interface Gradio abre automaticamente em `http://localhost:7860`

A API REST fica disponível em `http://127.0.0.1:8000/perguntar`

### Exemplo de pergunta via API REST

```bash
curl -X POST http://127.0.0.1:8000/perguntar \
  -H "Content-Type: application/json" \
  -d '{"pergunta": "O goleiro pode segurar a bola por quanto tempo?"}'
```

Resposta esperada:

```json
{
  "decisao": "8 segundos no máximo",
  "base_legal": "Regra 12, Seção 1",
  "explicacao": "O goleiro pode controlar a bola com as mãos por um máximo de 8 segundos. Após esse tempo, se continuar controlando, o árbitro concede um escanteio à equipe adversária.",
  "confianca": "Alta",
  "imagens": ["assets/..."],
  "score": 0.89
}
```

## 💰 Custos esperados

Modelo padrão: **gpt-4o**, escolhido pela precisão em punições condicionadas
à intensidade da infração. Cada consulta faz até 3 chamadas à API — geração,
verificação de fundamentação e, em perguntas multi-regra, decomposição.

| Operação | Tokens | Custo aproximado |
|----------|--------|------------------|
| Pergunta média | 1,5K-2K | ~$0.008 |
| 100 perguntas | 150K-200K | ~$0.80 |
| 1000 perguntas | 1,5M-2M | ~$8.00 |

Para reduzir custo, troque `MODEL_NAME` para `gpt-4o-mini` em `app.py` —
a resposta fica menos precisa em regras que dependem de gradação de gravidade.

Embeddings e reranking rodam localmente (`multilingual-e5-large` e
`mmarco-mMiniLMv2`), sem custo por consulta.

## 🐛 Solução de problemas

### Erro: "Chave da API OpenAI não configurada"

**Causa**: Variável `OPENAI_API_KEY` não encontrada

**Solução**:
```bash
export OPENAI_API_KEY='sk-...'
python app.py
```

### Erro: "OpenAI API error: 401 Unauthorized"

**Causa**: Chave inválida ou expirada

**Solução**: Verifique sua chave em https://platform.openai.com/api-keys

### Erro: "429 — Rate limit exceeded"

**Causa**: Muitas requisições em pouco tempo

**Solução**: Aguarde alguns minutos antes de fazer novas requisições

### Erro: "Timeout ao conectar com OpenAI"

**Causa**: Latência de rede alta ou OpenAI com problemas

**Solução**: Verifique sua conexão e tente novamente em alguns minutos

### O Gradio não abre no navegador

**Causa**: Porta 7860 já em uso ou firewall bloqueando

**Solução**:
```bash
# Especificar uma porta diferente
python app.py --server_port 8080
```

## 📚 Arquivos principais

| Arquivo | Descrição |
|---------|-----------|
| `app.py` | Aplicação principal (RAG + API + UI) |
| `build.py` | Script para reconstruir o índice FAISS |
| `test_apito_gpt.py` | Suite de testes do sistema |
| `test_openai_integration.py` | Teste da API OpenAI |
| `livros/` | Documentos markdown (Regras do Futebol) |
| `assets/` | Imagens (sinais de arbitragem, diagramas) |
| `requirements.txt` | Dependências Python |
| `.env.example` | Template de variáveis de ambiente |

## 🔄 Reconstruir o índice FAISS

Se modificar o documento em `livros/`, reconstrua o índice:

```bash
python build.py
```

Isso regenera `index.faiss` e `meta.pkl` a partir dos documentos.

## 🎓 Recursos adicionais

- **Documentação OpenAI**: https://platform.openai.com/docs
- **Documentação Gradio**: https://www.gradio.app/docs
- **FastAPI**: https://fastapi.tiangolo.com/

## ❓ Dúvidas?

Consulte o `README.md` para descrição completa do projeto e melhorias implementadas.

---

**Happy questioning! ⚽🎯**
