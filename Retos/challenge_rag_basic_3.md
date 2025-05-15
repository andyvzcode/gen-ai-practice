# Reto Básico 3: Construyendo tu Primer Sistema RAG (Retrieval Augmented Generation)

**Nivel de Dificultad:** Básico

**Herramientas Principales Involucradas:** Llama Index (o Langchain con un almacén de vectores), Python, Modelo de Lenguaje (ej. OpenAI).

**Conceptos Clave Abordados:** Retrieval Augmented Generation (RAG), Almacenes de Vectores (Vector Stores), Embeddings, Recuperación de Información (Retrieval), Aumentación de Contexto (Context Augmentation), Generación de Texto con LLMs.

**Objetivos Específicos del Reto:**

*   Comprender el patrón arquitectónico de RAG y su importancia para mejorar las respuestas de los LLMs.
*   Entender cómo RAG combina la recuperación de información de una base de conocimiento con la capacidad de generación de un LLM.
*   Utilizar Llama Index (o Langchain) para implementar un pipeline RAG simple.
*   Ingerir un conjunto de documentos y crear un índice (almacén de vectores) para una búsqueda semántica eficiente.
*   Implementar la fase de recuperación para encontrar los fragmentos de información más relevantes de los documentos indexados en base a una consulta del usuario.
*   Aprender a aumentar el prompt enviado al LLM con el contexto recuperado.
*   Generar una respuesta informada y contextualizada utilizando el LLM, basándose en la información recuperada y la consulta original.
*   Observar la diferencia en la calidad de las respuestas cuando se utiliza RAG en comparación con un LLM sin acceso a datos específicos.

**Introducción Conceptual y Relevancia:**

Retrieval Augmented Generation (RAG) es un paradigma poderoso en el desarrollo de aplicaciones con LLMs. Los LLMs, aunque vastos en su conocimiento preentrenado, a menudo carecen de información específica de un dominio, datos actualizados después de su entrenamiento, o conocimiento sobre documentos privados. RAG aborda esto "aumentando" el conocimiento del LLM en tiempo de ejecución. El proceso típicamente implica dos etapas principales: primero, una fase de **Recuperación** donde, dada una consulta del usuario, el sistema busca y recupera fragmentos de información relevante de una base de conocimiento externa (por ejemplo, tus propios documentos, una base de datos, etc.). Segundo, una fase de **Generación** donde esta información recuperada se proporciona como contexto adicional al LLM junto con la consulta original. El LLM luego genera una respuesta que está fundamentada en estos datos recuperados, lo que lleva a respuestas más precisas, detalladas, actualizadas y menos propensas a la "alucinación" (generar información incorrecta o inventada). Este reto te guiará en la construcción de un sistema RAG básico, una habilidad fundamental para crear aplicaciones de IA verdaderamente útiles y basadas en datos.

**Requisitos Previos:**

*   Conocimientos básicos de Python.
*   Haber completado el "Reto Básico 2: Indexando y Consultando tus Primeros Documentos con Llama Index" (o tener una comprensión sólida de sus conceptos, especialmente la carga de datos y la creación de índices/motores de consulta).
*   Una cuenta de OpenAI y una clave de API configurada de forma segura (como se describe en retos anteriores).

**Instrucciones Detalladas Paso a Paso:**

Para este reto, continuaremos utilizando Llama Index, ya que su `VectorStoreIndex` y `query_engine` implementan intrínsecamente el patrón RAG. El objetivo aquí es entender explícitamente este proceso.

**Paso 1: Revisión de la Configuración y Dependencias**

Asegúrate de tener `llama-index`, `openai` y `python-dotenv` instalados:

```bash
pip install llama-index openai python-dotenv
```

También, verifica que tu archivo `.env` con la `OPENAI_API_KEY` esté presente y sea accesible por tu script.

**Paso 2: Preparación de los Datos (Puede Reutilizar o Ampliar)**

Puedes reutilizar el directorio `data` y el archivo `mi_documento.txt` del reto anterior. Para hacer este reto un poco más interesante, considera añadir otro archivo de texto con información diferente o expandir `mi_documento.txt`.

Por ejemplo, crea `mi_documento_2.txt` en la carpeta `data` con el siguiente contenido:

```text
Langchain es otro framework popular para construir aplicaciones con LLMs.
Ofrece componentes modulares como cadenas, agentes y herramientas de memoria.
Se puede integrar con Llama Index para flujos de trabajo RAG más complejos.
Crew AI y LangGraph son herramientas para crear sistemas de agentes multi-colaborativos.
```

**Paso 3: Cargar, Indexar y Crear el Motor de Consulta (Repaso de Llama Index)**

Este paso es muy similar al Reto Básico 2. Crearemos un script `mi_primer_rag.py`.

```python
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openai import OpenAI

# Cargar variables de entorno
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("No se encontró la clave de API de OpenAI. Asegúrate de que esté configurada en tu archivo .env")

# Configurar el LLM globalmente para Llama Index
Settings.llm = OpenAI(model="gpt-3.5-turbo", api_key=api_key)
print("LLM de OpenAI configurado.")

# Cargar documentos
data_dir = "./data"
try:
    documents = SimpleDirectoryReader(data_dir).load_data()
    if not documents:
        print(f"No se encontraron documentos en {data_dir}")
        exit()
    print(f"Se cargaron {len(documents)} documento(s).")
except Exception as e:
    print(f"Error cargando documentos: {e}")
    exit()

# Crear el índice (VectorStoreIndex)
print("Creando el índice...")
try:
    index = VectorStoreIndex.from_documents(documents)
    print("Índice creado.")
except Exception as e:
    print(f"Error creando el índice: {e}")
    exit()
```

**Paso 4: Entendiendo la Fase de Recuperación (Retrieval)**

Cuando usas `index.as_query_engine()`, Llama Index crea automáticamente un recuperador (retriever) y lo combina con un sintetizador de respuestas. Vamos a hacerlo un poco más explícito para entender las partes.

Un recuperador se encarga de encontrar los nodos (fragmentos de tus documentos) más relevantes para una consulta.

```python
# ... (código anterior)

# 1. Configurar el Recuperador (Retriever)
# Puedes especificar cuántos documentos principales (top_k) recuperar.
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=3,  # Recuperar los 3 fragmentos más similares
)

# Probar el recuperador con una consulta
consulta_test = "¿Qué es Langchain?"
nodos_recuperados = retriever.retrieve(consulta_test)

print(f"\n--- Fase de Recuperación para: '{consulta_test}' ---")
if nodos_recuperados:
    print(f"Se recuperaron {len(nodos_recuperados)} nodos:")
    for i, nodo in enumerate(nodos_recuperados):
        print(f"\nNodo {i+1} (Score: {nodo.score:.4f}):")
        print(nodo.get_content()[:250] + "...") # Imprime los primeros 250 caracteres del nodo
else:
    print("No se recuperaron nodos.")
print("---------------------------------------------")
```

Ejecuta tu script. Deberías ver los fragmentos de texto de tus documentos que el recuperador consideró más relevantes para la pregunta "¿Qué es Langchain?". Esta es la "R" de RAG.

**Paso 5: Entendiendo la Fase de Generación Aumentada (Augmented Generation)**

Una vez que tienes los nodos relevantes, estos se utilizan para "aumentar" el prompt que se envía al LLM. El LLM luego genera una respuesta basada en tu consulta original Y esta información contextual adicional.

Llama Index tiene un `ResponseSynthesizer` que maneja esto. Se combina con el recuperador para formar un `RetrieverQueryEngine`, que es lo que `index.as_query_engine()` crea convenientemente.

```python
# ... (código anterior)

# 2. Configurar el Sintetizador de Respuestas (Response Synthesizer)
# Este componente toma los nodos recuperados y la consulta original, y genera una respuesta usando el LLM.
response_synthesizer = get_response_synthesizer(
    # response_mode="refine" # Puedes experimentar con diferentes modos
)

# 3. Construir el Motor de Consulta (Query Engine) explícitamente
# Este es el motor RAG que combina el recuperador y el sintetizador.
query_engine_rag = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
)

# Realizar una consulta al motor RAG
pregunta_rag = "Explícame brevemente qué es Langchain y cómo se relaciona con Llama Index."
print(f"\n--- Fase de Generación Aumentada para: '{pregunta_rag}' ---")

try:
    respuesta_rag = query_engine_rag.query(pregunta_rag)
    print(f"Respuesta del sistema RAG:\n{respuesta_rag}")

    # Los nodos fuente también están disponibles en la respuesta
    # print("\nNodos fuente utilizados para la respuesta:")
    # for nodo_fuente in respuesta_rag.source_nodes:
    #     print(f"Score: {nodo_fuente.score:.4f}, Contenido: {nodo_fuente.text[:100]}...")

except Exception as e:
    print(f"Error durante la consulta RAG: {e}")
print("-----------------------------------------------------")
```

**Paso 6: Comparar con un LLM sin RAG (Opcional pero Ilustrativo)**

Para apreciar realmente el valor de RAG, puedes hacer la misma pregunta directamente al LLM sin el contexto recuperado.

```python
# ... (código anterior)

# Comparación: Preguntar directamente al LLM sin RAG
llm_directo = Settings.llm # Usamos el LLM configurado globalmente
pregunta_directa_llm = "Explícame brevemente qué es Langchain y cómo se relaciona con Llama Index."

print(f"\n--- Consulta Directa al LLM (sin RAG) para: '{pregunta_directa_llm}' ---")
try:
    respuesta_directa_llm = llm_directo.complete(pregunta_directa_llm)
    print(f"Respuesta directa del LLM:\n{respuesta_directa_llm.text}")
except Exception as e:
    print(f"Error durante la consulta directa al LLM: {e}")
print("-------------------------------------------------------------")
```

**Paso 7: Analizar los Resultados**

Ejecuta tu script `mi_primer_rag.py` completo. Observa:
1.  Los nodos recuperados en el Paso 4.
2.  La respuesta generada por el sistema RAG en el Paso 5. Debería basarse en la información de tus archivos `mi_documento.txt` y `mi_documento_2.txt`.
3.  La respuesta generada por el LLM directamente en el Paso 6. Compara esta respuesta con la del sistema RAG. ¿Es la respuesta RAG más específica o precisa según tus documentos?

**Recursos y Documentación Adicional:**

*   Llama Index - Documentación sobre Motores de Consulta (Query Engines): [https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/root.html](https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/root.html)
*   Llama Index - Documentación sobre Recuperadores (Retrievers): [https://docs.llamaindex.ai/en/stable/module_guides/querying/retriever/root.html](https://docs.llamaindex.ai/en/stable/module_guides/querying/retriever/root.html)
*   Llama Index - Documentación sobre Sintetizadores de Respuesta (Response Synthesizers): [https://docs.llamaindex.ai/en/stable/module_guides/querying/response_synthesizer/root.html](https://docs.llamaindex.ai/en/stable/module_guides/querying/response_synthesizer/root.html)
*   Artículo de Blog de Langchain sobre RAG: Aunque este reto usa Llama Index, los conceptos de RAG son universales. Langchain también tiene excelentes capacidades RAG. [https://blog.langchain.dev/retrieval-augmented-generation-rag/](https://blog.langchain.dev/retrieval-augmented-generation-rag/)

**Criterios de Evaluación y Verificación:**

*   Tu script `mi_primer_rag.py` se ejecuta sin errores.
*   El script muestra los nodos recuperados que son relevantes para tu consulta de prueba.
*   El sistema RAG genera respuestas que utilizan la información de tus documentos locales.
*   Puedes articular la diferencia entre la respuesta del sistema RAG y la respuesta de un LLM sin RAG para la misma pregunta, notando cómo RAG utiliza el contexto de tus documentos.

**Posibles Extensiones o Retos Adicionales:**

*   Experimenta con el parámetro `similarity_top_k` en el `VectorIndexRetriever`. ¿Cómo afecta a la respuesta cambiar el número de fragmentos recuperados?
*   Prueba diferentes `response_mode` en `get_response_synthesizer` (por ejemplo, "compact", "tree_summarize") y observa cómo cambia la respuesta y el proceso de síntesis.
*   Intenta implementar un RAG similar utilizando Langchain. Esto implicaría usar un `VectorStore` de Langchain (como FAISS o Chroma), un `RetrievalQA` chain, o construirlo manualmente con un `RunnablePassthrough` y `RunnableParallel` para pasar los documentos recuperados al prompt.
*   Añade más documentos sobre temas variados y prueba la capacidad del sistema RAG para responder preguntas sobre ellos.
