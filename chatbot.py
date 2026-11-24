import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ==============================================================================
# PASO 1 y 2: CONFIGURACIÓN DEL CHATBOT RAG CON OLLAMA
# ==============================================================================

# Carpeta donde debes poner todos tus archivos PDF
PDF_DIRECTORY = "pdfs"

def crear_base_conocimiento():
    """
    Lee los PDFs, los divide en fragmentos y crea la base de datos vectorial.
    """
    print("1. Cargando documentos PDF...")
    # Cargar todos los PDFs de la carpeta
    loader = PyPDFDirectoryLoader(PDF_DIRECTORY)
    documentos = loader.load()
    
    if not documentos:
        print("⚠️ No se encontraron PDFs en la carpeta 'pdfs/'. Por favor, añade algunos.")
        return None

    print(f"   Se cargaron {len(documentos)} páginas en total.")

    print("2. Dividiendo el texto en fragmentos (chunks)...")
    # Utilizamos un separador basado en expresiones regulares (regex) para
    # respetar tu secuencia de 1., 2., etc. Así los temas no se cortan por la mitad.
    text_splitter = RecursiveCharacterTextSplitter(
        separators=[
            r"\n\d+\.\s",  # Busca un salto de línea seguido de un número y un punto (ej: "\n1. ")
            "\n\n",        # Párrafos
            "\n",          # Líneas
            " ",           # Palabras
            ""             # Caracteres
        ],
        is_separator_regex=True,
        chunk_size=1200,   # Un poco más grande para abarcar bien cada sección
        chunk_overlap=200
    )
    textos_fragmentados = text_splitter.split_documents(documentos)

    print("3. Creando Base de Datos Vectorial (FAISS)...")
    # Utilizamos un modelo local y ligero de HuggingFace para convertir el texto en vectores
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Crear y guardar la base de datos vectorial en memoria
    vectorstore = FAISS.from_documents(textos_fragmentados, embeddings)
    print("✅ Base de conocimiento creada exitosamente.")
    return vectorstore

def configurar_chatbot(vectorstore):
    """
    Configura el LLM (Ollama) y el Prompt estructurado con la personalidad solicitada.
    """
    print("4. Conectando con Ollama Local...")
    # Aquí puedes cambiar "llama3" por el modelo que tengas descargado en Ollama (ej. mistral, qwen)
    llm = Ollama(model="llama3.2:3b-instruct-q4_K_M")

    print("5. Definiendo la personalidad (System Prompt)...")
    # Definir el Prompt con el perfil de experto, tono directo y contexto
    template_experto = """Eres un experto absoluto en los grupos de investigación, semilleros de investigación de Ingeniería Catastral y Geodesia de la Universidad Distrital Francisco José de Caldas.
Tu tono debe ser extremadamente directo, técnico y al grano. No uses saludos excesivos ni introducciones largas.
Responde a la pregunta basándote ÚNICAMENTE en la siguiente información de contexto (que proviene de PDFs oficiales).
Si la información no está en el contexto, di "No tengo información en mis documentos sobre esto" y no inventes respuestas.

Contexto:
{context}

Pregunta del usuario: {question}

Respuesta directa:"""
    
    PROMPT = PromptTemplate(
        template=template_experto, 
        input_variables=["context", "question"]
    )

    # Crear el motor final que une el LLM, el Prompt y el buscador usando LCEL (Modern LangChain)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Trae los 3 fragmentos más relevantes
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
    
    return qa_chain

if __name__ == "__main__":
    # Asegurarnos de que existe la carpeta
    if not os.path.exists(PDF_DIRECTORY):
        os.makedirs(PDF_DIRECTORY)
        print(f"He creado la carpeta '{PDF_DIRECTORY}'. Por favor, pon tus PDFs ahí y vuelve a ejecutar el script.")
    else:
        # Iniciar el proceso
        vector_db = crear_base_conocimiento()
        
        if vector_db:
            chatbot = configurar_chatbot(vector_db)
            print("\n" + "="*50)
            print("🤖 Chatbot Experto en Ing. Catastral y Geodesia Listo")
            print("Escribe 'salir' para terminar.")
            print("="*50 + "\n")
            
            # Ciclo de chat en la terminal
            while True:
                pregunta = input("Tú: ")
                if pregunta.lower() in ['salir', 'exit', 'quit']:
                    print("¡Hasta luego!")
                    break
                
                print("Procesando...")
                respuesta = chatbot.invoke(pregunta)
                print(f"\nExperto: {respuesta}\n")
