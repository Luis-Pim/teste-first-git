# Tormenta20 RPG Assistant

Este projeto visa criar um assistente inteligente para responder perguntas sobre o sistema de RPG Tormenta20, utilizando técnicas de embeddings para melhorar a busca e consulta de informações extraídas de documentos. O sistema utiliza a API da OpenAI para processamento de linguagem natural e FAISS para armazenar e consultar os embeddings gerados.

## Estrutura do Projeto

O projeto é dividido em várias etapas para processar e consultar as informações sobre o sistema de RPG Tormenta20:

1. **Pré-processamento**: Carregamento dos dados extraídos de OCR e tabelas JSON para estruturas de dados.
2. **Geração de Embeddings**: Criação de embeddings para representar as informações de forma vetorial.
3. **Criação de Base FAISS**: Armazenamento dos embeddings em um índice FAISS para consultas eficientes.
4. **Consulta**: Consultas por pergunta ou tema utilizando o índice FAISS e a API da OpenAI para gerar respostas baseadas no contexto fornecido.

## Requisitos

### Dependências

Para executar este projeto, é necessário instalar as dependências abaixo:

* `openai`: Para interagir com a API da OpenAI.
* `faiss-cpu` ou `faiss-gpu`: Para operações de busca vetorial usando o FAISS.
* `numpy`: Para manipulação de arrays numéricos.
* `tqdm`: Para exibição de barras de progresso durante o processamento.

Você pode instalar essas dependências usando o seguinte comando:

```bash
pip install openai faiss-cpu numpy tqdm
```

### Arquivos de Entrada

Este projeto depende de quatro arquivos principais de entrada:

* `tormenta_ocr_result.json`: Resultado do OCR (Reconhecimento Óptico de Caracteres) do livro ou material de Tormenta20.
* `tabelas_tormenta_unificado.json`: Tabelas com informações estruturadas sobre o sistema Tormenta20.
* `tormenta_index.faiss`: Índice FAISS (se já existente) que armazena os embeddings gerados.
* `tormenta_metadata.json`: Metadados associados aos embeddings, contendo informações sobre o conteúdo de cada chunk de texto.

### Chave da API OpenAI

A chave de API da OpenAI deve ser configurada antes de rodar o projeto. A chave pode ser obtida ao se cadastrar na OpenAI e configurada como variável `openai_api_key` no código.

## Como Executar

1. **Configuração Inicial**:
   Antes de executar, insira a chave da API da OpenAI no código, substituindo `chave` por sua chave real:

   ```python
   openai_api_key = "sua_chave_aqui"
   ```

2. **Execução do Código**:
   O código pode ser executado diretamente. Na execução inicial, ele irá gerar o índice FAISS e os metadados a partir dos arquivos de entrada, caso esses índices ainda não existam. Se o índice e os metadados já estiverem presentes, o sistema pulará a criação do índice.

   ```bash
   python tormenta_assistant.py
   ```

## Funcionamento

O processo está dividido em quatro etapas principais:

### Etapa 1: Pré-processamento

O pré-processamento carrega os dados extraídos de OCR e de tabelas e os transforma em *chunks* de texto estruturados.

#### Funções

* **`carregar_ocr_chunks(path)`**: Carrega os dados de OCR de um arquivo JSON e estrutura os dados em uma lista de dicionários contendo um `id` e o `texto`.
* **`carregar_tabelas_como_chunks(path)`**: Carrega as tabelas unificadas de Tormenta20 de um arquivo JSON, transformando-as em uma lista de chunks de texto.

### Etapa 2: Geração de Embeddings

Utilizando o modelo `text-embedding-ada-002` da OpenAI, o sistema gera embeddings para cada *chunk* de texto carregado. Embeddings são representações vetoriais do texto que permitem a busca e a comparação eficiente.

#### Funções

* **`gerar_embedding(texto)`**: Gera um embedding para um texto dado, utilizando a API da OpenAI.

### Etapa 3: Criação da Base FAISS

Os embeddings gerados são armazenados em um índice FAISS para permitir consultas rápidas. O índice FAISS é um método eficiente de pesquisa em grandes volumes de dados vetoriais.

#### Funções

* **`criar_base_faiss(chunks)`**: Cria o índice FAISS a partir de uma lista de chunks. O índice é então salvo em disco como `tormenta_index.faiss`, e os metadados são salvos em `tormenta_metadata.json`.

### Etapa 4: Consultas

Com o índice e os metadados prontos, o sistema pode responder a perguntas dos usuários com base nas informações do sistema Tormenta20.

#### Funções

* **`consultar_pergunta(pergunta, top_k=5)`**: Recebe uma pergunta, gera o embedding da pergunta, realiza uma busca no índice FAISS e usa a API da OpenAI para gerar uma resposta com base nos contextos mais relevantes.

* **`consultar_pergunta_por_tema(tema)`**: Realiza uma busca por tema no índice FAISS e gera uma resposta baseada nos contextos relacionados ao tema solicitado.

## Exemplo de Uso

### Consultar por Pergunta

Para fazer uma consulta por pergunta, basta chamar a função `consultar_pergunta` passando a pergunta desejada:

```python
resposta = consultar_pergunta("Como funcionam os ataques em Tormenta20?")
print(resposta)
```

### Consultar por Tema

Para consultar um tema específico, utilize a função `consultar_pergunta_por_tema`:

```python
resposta = consultar_pergunta_por_tema("magias")
print(resposta)
```

## Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests no repositório. Certifique-se de seguir as melhores práticas de codificação e testar seu código antes de enviar.

## Licença

Este projeto está licenciado sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

Se você precisar de mais informações ou tiver dúvidas sobre a implementação, sinta-se à vontade para perguntar!
