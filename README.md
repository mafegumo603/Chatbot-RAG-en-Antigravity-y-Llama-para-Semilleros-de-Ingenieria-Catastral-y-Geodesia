# Chatbot RAG en Antigravity y Llama para Semilleros de Ingeniería Catastral y Geodesia

Asistente de consulta local basado en **RAG (Retrieval-Augmented Generation)** orientado a responder preguntas sobre grupos y semilleros de investigación de **Ingeniería Catastral y Geodesia** de la **Universidad Distrital Francisco José de Caldas**, a partir de documentos PDF oficiales.

El sistema carga archivos PDF desde una carpeta local, fragmenta el contenido respetando numeraciones y párrafos, genera embeddings locales con **HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)**, indexa los fragmentos en **FAISS** y consulta un modelo local en **Ollama** (`llama3.2:3b-instruct-q4_K_M`) mediante **LangChain (LCEL)** con instrucciones estrictas para ceñirse al contexto documental.

---

## Arquitectura y funcionamiento

1. **Carga documental (`PyPDFDirectoryLoader`):** Lee todos los archivos `.pdf` ubicados en el directorio `pdfs/`. Si la carpeta no existe, el script la crea automáticamente y solicita agregar documentos.
2. **Segmentación (`RecursiveCharacterTextSplitter`):** Divide el texto en fragmentos (`chunk_size=1200`, `chunk_overlap=200`) utilizando separadores regex (`\n\d+\.\s`), saltos de párrafo y línea para conservar la estructura temática.
3. **Vectorización e índice (`HuggingFaceEmbeddings` + `FAISS`):** Convierte los fragmentos en vectores densos y construye un índice vectorial en memoria con recuperación de los `k=3` fragmentos más relevantes por consulta.
4. **Generación controlada (`Ollama` + `PromptTemplate` + LCEL):** Envía el contexto recuperado y la pregunta al modelo `llama3.2:3b-instruct-q4_K_M` con un rol técnico y directo, indicando responder *"No tengo información en mis documentos sobre esto"* cuando el dato no figura en los PDFs.

---

## Estructura del proyecto

```text
Chatbot-RAG-en-Antigravity-y-Llama-para-Semilleros-de-Ingenier-a-Catastral-y-Geodesia/
├── main.py              # (o script principal con el pipeline RAG y bucle de chat en consola)
├── pdfs/                # Carpeta de entrada para PDFs oficiales de grupos y semilleros
└── README.md
