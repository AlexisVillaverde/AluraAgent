# 🛒 BimBam Buy - Asistente Corporativo de IA

## 📖 Descripción general del proyecto
Este proyecto consiste en el desarrollo de un agente de Inteligencia Artificial corporativo, accesible para todos los colaboradores de la empresa hipotética **BimBam Buy**. Su objetivo principal es funcionar como una base de conocimiento conversacional centralizada, capaz de responder de forma ágil y precisa a preguntas relacionadas con políticas internas, reembolsos, garantías, métodos de pago y logística de envíos, basándose exclusivamente en los documentos internos oficiales de la organización. 

Este proyecto fue desarrollado como parte del desafío de Inteligencia Artificial de Alura.

---

## 🏗️ Arquitectura de la solución implementada
El agente está construido sobre una arquitectura **RAG (Retrieval-Augmented Generation)**, dividida en dos procesos principales:

1. **Pipeline de Ingesta (Preparación de datos):** 
   Los documentos corporativos en formato PDF son procesados utilizando `PyPDFDirectoryLoader`. El texto extraído se divide en fragmentos (chunks) utilizando `RecursiveCharacterTextSplitter` para mantener el contexto semántico. Posteriormente, estos fragmentos se transforman en representaciones vectoriales matemáticas y se almacenan localmente en una base de datos vectorial para consultas rápidas.
2. **Motor de Recuperación y Generación (Inferencia):** 
   Cuando un usuario realiza una consulta a través de la interfaz web, el sistema vectoriza la pregunta, busca los 6 fragmentos más relevantes (Top-K) en la base de datos vectorial, y los inyecta como contexto en un modelo de lenguaje de gran escala (LLM). El LLM sintetiza la información recuperada y redacta una respuesta natural, veraz y fundamentada estrictamente en la documentación.
```mermaid
   graph TD
    %% Estilos de los nodos
    classDef database fill:#f9f6f0,stroke:#333,stroke-width:2px;
    classDef user fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef llm fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;
    classDef ui fill:#fff3e0,stroke:#e65100,stroke-width:2px;

    subgraph Fase 1: Ingesta de Documentos
        A[PDFs de BimBam Buy] -->|PyPDF Loader| B(División de Texto)
        B -->|RecursiveCharacterTextSplitter| C(Generación de Vectores)
        C -->|HuggingFace Embeddings<br>paraphrase-multilingual| D[(ChromaDB)]
        class D database;
    end

    subgraph Fase 2: Interfaz y Motor RAG
        U((Colaborador)) -->|Pregunta| UI[Streamlit App]
        UI -->|Convierte a vector| C2(HuggingFace Embeddings)
        C2 -->|Busca similitud| D
        D -.->|Devuelve Top-6 Fragmentos| RAG[LangChain<br>Retrieval Chain]
        UI -->|Pregunta Original| RAG
        RAG -->|Contexto + Pregunta| LLM[Cohere API<br>command-r-plus]
        LLM -->|Respuesta Generada| UI
        
        class U user;
        class LLM llm;
        class UI ui;
    end
    ```


---

## 🛠️ Tecnologías y herramientas utilizadas
* **Lenguaje Core:** Python 3.12
* **Framework RAG:** LangChain
* **Modelo de Lenguaje (LLM):** Cohere, optimizado para tareas corporativas y RAG.
* **Modelo de Embeddings:** HuggingFace (`paraphrase-multilingual-MiniLM-L12-v2`), seleccionado específicamente por su alto rendimiento en español.
* **Base de Datos Vectorial:** ChromaDB (Persistente local).
* **Interfaz de Usuario (Frontend):** Streamlit.
* **Infraestructura y Despliegue:** Oracle Cloud Infrastructure (OCI) / Streamlit Community Cloud.

---
🌐 Demo Interactiva en la Nube
Accede a la aplicación desde tu navegador web:
👉 Probar Demo en Vivo en Streamlit Cloud(https://aluraagent-ccusak6w2xsx45zyxyr2kp.streamlit.app/) 🚀
---


## 🚀 Instrucciones para ejecutar el proyecto

Sigue estos pasos para clonar y ejecutar el agente de IA en tu máquina local:

**1. Clonar el repositorio:**
```bash
git clone 
cd tu-repositorio

**2. Crear y activar el entorno virtual:**
# En Windows:
python -m venv .venv
.venv\Scripts\activate

# En Linux/macOS:
python3 -m venv .venv
source .venv/bin/activate

**3. Instalar las dependencias requeridas:**
```bash
pip install -r requirements.txt

**4. Configurar las variables de entorno:**
Crea un archivo llamado .env en la raíz del proyecto y agrega tu clave de acceso de Cohere:

COHERE_API_KEY=tu_clave_de_cohere_aqui

**5. Procesar los documentos (Pipeline de Ingesta):**
Ejecuta este script una única vez (o cada que agregues nuevos PDFs a la carpeta data/) para generar la base de datos vectorial:

```bash
python ingest.py

**6. Levantar la aplicación web:**

```bash
streamlit run app.py

La interfaz gráfica se abrirá automáticamente en tu navegador web en la dirección http://localhost:8501.

❓ Ejemplos de preguntas que el agente puede responder
El agente está capacitado para contestar consultas como las siguientes, basándose en la documentación interna de BimBam Buy:

"¿Cuál es el tiempo en días para solicitar un retracto de compra o reembolso?"

"¿Qué situaciones y daños específicos no cubre la garantía de los productos?"

"¿Cómo funciona el programa de afiliados y cuáles son las comisiones?"

"¿Cuáles son los costos y tiempos de envío estándar para el interior del país?"

"¿Qué métodos de pago son aceptados actualmente en la plataforma?"

💬 Ejemplos de respuestas generadas por el agente
Usuario: ¿Cuál es el tiempo en días para un retracto o reembolso?

Agente BimBam Buy: De acuerdo con la Política de Reembolsos, el cliente puede solicitar una devolución por retracto de compra dentro de los 10 días corridos posteriores a la recepción del pedido, siempre y cuando el producto cumpla con los requisitos de elegibilidad documentados. Por otro lado, si el producto llegó dañado, el caso debe reportarse dentro de las primeras 48 horas para aplicar a un reemplazo rápido.

Usuario: ¿Qué pasa si el cliente pierde su recibo de compra, aún aplica la garantía?

Agente BimBam Buy: No tengo esa información exacta en los documentos actuales. Los manuales especifican los tiempos y condiciones de los daños, pero no detallan explícitamente el procedimiento a seguir en caso de extravío del comprobante o recibo de compra. Te recomiendo escalar esta consulta directamente con el supervisor del área de Soporte al Cliente.