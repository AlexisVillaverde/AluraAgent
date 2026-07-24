import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_cohere import ChatCohere
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Cargar variables de entorno (API Keys)
load_dotenv()

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="BimBam Buy - Asistente Interno", page_icon="🛒")
st.title("Asistente Corporativo: BimBam Buy 🛒")
st.markdown("Pregunta cualquier duda sobre reembolsos, garantías, envíos o políticas internas.")

with st.sidebar:
    st.header("🤖 Sobre el Agente")
    st.write(
        "Este asistente de Inteligencia Artificial está diseñado para ayudar a todos los "
        "colaboradores de **BimBam Buy** a encontrar respuestas rápidas y precisas basadas "
        "estrictamente en la documentación oficial e interna de la empresa."
    )
    
    st.divider() # Línea separadora visual
    
    st.subheader("💡 Preguntas de ejemplo")
    st.markdown("Puedes intentar copiar y pegar alguna de estas consultas:")
    
    # Cajas de información para destacar las preguntas
    st.info("¿Cuál es el tiempo máximo para solicitar un reembolso?")
    st.info("¿Qué situaciones no cubre la garantía de los productos?")
    st.info("¿Cómo funciona el programa de afiliados para los clientes?")
    st.info("¿Cuáles son los costos y tiempos de envío estándar?")

# --- INICIALIZACIÓN DEL SISTEMA RAG ---
@st.cache_resource
def iniciar_sistema_rag():
    # 1. Cargar la misma función de embeddings usada en ingest.py
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    
    # 2. Conectar a la base de datos vectorial existente
    vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6}) # Trae los 3 fragmentos más relevantes
    
    # 3. Configurar el LLM (el que redactará la respuesta)
    clave_cohere = os.getenv("COHERE_API_KEY")
    
    if not clave_cohere:
        st.error("No se encontró la clave COHERE_API_KEY. Revisa tu archivo .env")
        st.stop()
        
    # Usamos 'command-r', que es el modelo de Cohere optimizado para sistemas RAG
    llm = ChatCohere(
        cohere_api_key=clave_cohere
    )

    # 4. Crear el Prompt (Instrucciones para la IA)
    system_prompt = (
        "Eres un asistente de recursos humanos y operaciones para la empresa BimBam Buy. "
        "Usa los siguientes fragmentos de contexto recuperado para responder a la pregunta del colaborador. "
        "Si no sabes la respuesta basándote en el contexto, di que no tienes esa información en los documentos. "
        "Sé profesional, claro y conciso.\n\n"
        "Contexto:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # 5. Unir todo en una cadena de ejecución (Chain)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

rag_chain = iniciar_sistema_rag()

# --- LÓGICA DE LA INTERFAZ DE CHAT ---
# Guardar el historial de la conversación en la sesión
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

# Mostrar mensajes anteriores
for mensaje in st.session_state.mensajes:
    with st.chat_message(mensaje["rol"]):
        st.markdown(mensaje["contenido"])

# Caja de texto para nueva pregunta
pregunta_usuario = st.chat_input("Ej: ¿Cuál es el tiempo máximo para solicitar un reembolso?")

if pregunta_usuario:
    # Mostrar la pregunta del usuario
    st.session_state.mensajes.append({"rol": "user", "contenido": pregunta_usuario})
    with st.chat_message("user"):
        st.markdown(pregunta_usuario)
        
    # Generar y mostrar la respuesta de la IA
    with st.chat_message("assistant"):
        with st.spinner("Buscando en los documentos de BimBam Buy..."):
            respuesta = rag_chain.invoke({"input": pregunta_usuario})
            texto_respuesta = respuesta["answer"]
            st.markdown(texto_respuesta)
            
    # Guardar respuesta en el historial
    st.session_state.mensajes.append({"rol": "assistant", "contenido": texto_respuesta})