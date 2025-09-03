import json
import os
from openai import OpenAI  # Usando o cliente OpenAI correto
import faiss
import numpy as np
from tqdm import tqdm
from typing import List
from flask import Flask, jsonify, request

# Configuração do Flask
app = Flask(__name__)

# Configurar chave de API da OpenAI
chave = "" #chave da api aqui
openai_api_key = chave
client = OpenAI(api_key=openai_api_key)  # Usando a nova forma de instanciar o cliente OpenAI

# Caminhos dos arquivos
OCR_PATH = "tormenta_ocr_result.json"
TABELA_PATH = "tabelas_tormenta_unificado.json"
INDEX_PATH = "tormenta_index.faiss"
METADATA_PATH = "tormenta_metadata.json"

# Modelo de embedding
EMBEDDING_MODEL = "text-embedding-ada-002"

# --- ETAPA 1: PRÉ-PROCESSAMENTO ---
def carregar_ocr_chunks(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        ocr = json.load(f)
    return [{"id": k, "text": v} for k, v in ocr.items() if v.strip()]

def carregar_tabelas_como_chunks(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        tabelas = json.load(f)
    chunks = []
    for tipo, entradas in tabelas.items():
        for entrada in entradas:
            texto = "\n".join(f"{k}: {v}" for k, v in entrada.items())
            nome = entrada.get("nome") or entrada.get("Nome") or "(sem nome)"
            chunks.append({"id": f"{tipo}:{nome}", "text": texto})
    return chunks

# --- ETAPA 2: EMBEDDINGS ---
def gerar_embedding(texto: str) -> List[float]:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,  # Modelo para embeddings
        input=texto,
        encoding_format="float"
    )
    return response.data[0].embedding  # Acessando diretamente com o atributo 'data' e 'embedding'

# --- ETAPA 3: CRIAR BASE FAISS ---
def criar_base_faiss(chunks: List[dict]):
    embeddings = []
    metadata = []
    for chunk in tqdm(chunks, desc="Gerando embeddings"):
        emb = gerar_embedding(chunk["text"])
        embeddings.append(emb)
        metadata.append({"id": chunk["id"], "text": chunk["text"]})

    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))

    # Salvar índice e metadados
    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\nIndex criado com {len(embeddings)} chunks.")

# --- ETAPA 4: CONSULTA ---
@app.route('/consultar', methods=['GET'])
def consultar_pergunta_api():
    pergunta = request.args.get('pergunta')  # Parâmetro da query string
    if not pergunta:
        return jsonify({"error": "Pergunta não fornecida"}), 400
    
    return jsonify({"resposta": consultar_pergunta(pergunta)})

def consultar_pergunta(pergunta: str, top_k: int = 5):
    pergunta_emb = gerar_embedding(pergunta)
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    D, I = index.search(np.array([pergunta_emb]).astype("float32"), top_k)
    contextos = [metadata[i]["text"] for i in I[0] if i < len(metadata)]

    # Limitação de tokens para baratear o custo
    contextos_joined = " ".join(contextos)
    contextos_limitados = contextos_joined[:3500]  # Limita a aproximadamente 3500 caracteres

    prompt = f"""
Você é um especialista no sistema de RPG Tormenta20. Use os textos abaixo retirados do livro e das regras oficiais para responder à pergunta do jogador.

--- Contexto ---
{contextos_limitados}

--- Pergunta ---
{pergunta}

--- Resposta ---
Responda apenas com base nas informações fornecidas, em português brasileiro.
"""

    resposta = client.chat.completions.create(
        model="gpt-5", # Ou 'gpt-3.5-turbo'
        messages=[
            {"role": "system", "content": "Você é um assistente que responde perguntas sobre o RPG Tormenta20."},
            {"role": "user", "content": prompt}
        ]
    )

    return resposta.choices[0].message.content


# --- CONSULTA POR TEMA ---
@app.route('/consultar_tema', methods=['GET'])
def consultar_tema_api():
    tema = request.args.get('tema')  # Parâmetro da query string
    if not tema:
        return jsonify({"error": "Tema não fornecido"}), 400
    
    return jsonify({"resposta": consultar_pergunta_por_tema(tema)})

def consultar_pergunta_por_tema(tema: str):
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    contextos = [item["text"] for item in metadata if tema.lower() in item["id"].lower()]

    if not contextos:
        return "Nenhum contexto encontrado sobre esse tema."

    # Limitação de tokens para baratear o custo
    contextos_joined = " ".join(contextos)
    contextos_limitados = contextos_joined[:2000]  # Limita a aproximadamente 3500 caracteres

    prompt = f"""
Você é um especialista no sistema de RPG Tormenta20.
O usuário quer saber sobre o seguinte tema: {tema}.
Responda usando as informações detalhadas abaixo.

--- Informações ---
{contextos_limitados}

--- Pergunta ---
Liste e explique brevemente os itens relacionados ao tema \"{tema}\" acima.

*RESPONDA SEMRE NO FORMATO JSON*

--- Resposta ---
"""

    resposta = client.chat.completions.create(
        model="gpt-5",  # Ou 'gpt-3.5-turbo'
        messages=[
            {"role": "system", "content": "Você é um assistente que responde perguntas sobre o RPG Tormenta20."},
            {"role": "user", "content": prompt}
        ]
    )

    return resposta.choices[0].message.content


# --- EXECUÇÃO INICIAL ---
if __name__ == "__main__":
    if not (os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH)):
        ocr_chunks = carregar_ocr_chunks(OCR_PATH)
        tabela_chunks = carregar_tabelas_como_chunks(TABELA_PATH)
        todos_chunks = ocr_chunks + tabela_chunks
        criar_base_faiss(todos_chunks)
    else:
        print("✅ Índice FAISS e metadados já existem. Pulando criação.")

    app.run(host='0.0.0.0', port=5001)  # Rodando na porta 5001
