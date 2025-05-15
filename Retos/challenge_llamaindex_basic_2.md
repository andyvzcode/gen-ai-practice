# Reto Básico 2: Indexando y Consultando tus Primeros Documentos con Llama Index

**Nivel de Dificultad:** Básico

**Herramientas Principales Involucradas:** Llama Index, Python, Modelo de Lenguaje (ej. OpenAI)

**Conceptos Clave Abordados:** Ingesta de Datos (Data Ingestion), Documentos (Documents), Nodos (Nodes), Indexación (Indexing), Vectores de Embeddings, Motores de Consulta (Query Engines), LLMs.

**Objetivos Específicos del Reto:**

*   Instalar la biblioteca Llama Index (ahora también conocida como `llama-index`).
*   Comprender el propósito de Llama Index para conectar LLMs con datos externos.
*   Aprender a cargar datos desde archivos locales (por ejemplo, archivos de texto) en objetos `Document` de Llama Index.
*   Entender el concepto de indexación y cómo Llama Index crea estructuras de datos para una recuperación eficiente.
*   Construir un índice simple (por ejemplo, `VectorStoreIndex`) a partir de los documentos cargados.
*   Configurar y utilizar un motor de consulta (`query_engine`) para hacer preguntas sobre los datos indexados.
*   Observar cómo Llama Index utiliza un LLM para sintetizar respuestas basadas en la información recuperada.
*   Configurar de forma segura tu clave de API de OpenAI si se utiliza como LLM subyacente.

**Introducción Conceptual y Relevancia:**

Llama Index es un framework de datos para aplicaciones LLM que permite ingerir, estructurar y acceder a datos privados o específicos de un dominio. Mientras que los LLMs preentrenados tienen un vasto conocimiento general, no conocen tus datos personales o la información más reciente no incluida en su entrenamiento. Llama Index resuelve esto proporcionando herramientas para cargar tus datos (desde APIs, PDFs, documentos de texto, bases de datos, etc.), indexarlos de manera inteligente (a menudo convirtiendo el texto en representaciones numéricas llamadas embeddings y almacenándolos en índices vectoriales) y luego consultarlos utilizando el poder de los LLMs para obtener respuestas contextualizadas. Este reto te introducirá al flujo de trabajo fundamental de Llama Index: cargar datos, crear un índice y consultarlo. Dominar estos pasos es crucial para construir aplicaciones RAG (Retrieval Augmented Generation) y otras aplicaciones LLM que operan sobre tus propios datos.

**Requisitos Previos:**

*   Conocimientos básicos de Python.
*   Una cuenta de OpenAI y una clave de API (si usas modelos de OpenAI). Consíguela en [platform.openai.com](https://platform.openai.com/).
*   Haber completado (o entender los conceptos de) la configuración de la clave API como se describe en el "Reto Básico 1: Creando tu Primera Cadena (Chain) con Langchain".

**Instrucciones Detalladas Paso a Paso:**

**Paso 1: Instalación de las Bibliotecas Necesarias**

Necesitarás instalar `llama-index` y, si planeas usar los modelos de OpenAI, también `openai` y `python-dotenv`.

```bash
pip install llama-index openai python-dotenv
```

*   `llama-index`: La biblioteca principal de Llama Index.
*   `openai`: Para interactuar con la API de OpenAI (si se usa como LLM).
*   `python-dotenv`: Para gestionar tu clave de API de forma segura.

**Paso 2: Preparación de tus Datos**

Llama Index puede ingerir datos de muchas fuentes. Para este reto básico, crearemos un simple archivo de texto.

1.  Crea un directorio llamado `data` en la raíz de tu proyecto.
2.  Dentro del directorio `data`, crea un archivo de texto llamado `mi_documento.txt`.
3.  Añade algo de texto a `mi_documento.txt`. Por ejemplo:
    ```text
    La historia de la inteligencia artificial (IA) es fascinante.
    Comenzó con ideas filosóficas sobre máquinas pensantes.
    Alan Turing fue un pionero clave en este campo.
    Hoy en día, los Modelos de Lenguaje Grandes como GPT están transformando muchas industrias.
    Llama Index ayuda a conectar estos modelos con datos personalizados.
    ```

**Paso 3: Configuración de tu Clave de API de OpenAI (Si es Necesario)**

Si vas a utilizar un LLM de OpenAI (que es el predeterminado para muchas funcionalidades en Llama Index si no se especifica lo contrario), necesitas configurar tu clave de API. Sigue el mismo procedimiento que en el Reto Básico 1 de Langchain:

1.  Asegúrate de tener un archivo `.env` en el directorio raíz de tu proyecto.
2.  Añade tu clave de API al archivo `.env`:
    ```
    OPENAI_API_KEY="TU_CLAVE_DE_API_DE_OPENAI"
    ```

**Paso 4: Creación de tu Script de Llama Index**

Crea un archivo Python (por ejemplo, `mi_primer_indice.py`) e importa las clases necesarias. También cargaremos la clave de API.

```python
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI # Si usas OpenAI

# Cargar variables de entorno
load_dotenv()

# Configurar el LLM (Opcional, LlamaIndex puede usar OpenAI por defecto si la clave está disponible)
# Esto es explícito para mayor claridad y para permitir cambiar a otros LLMs más adelante.
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Advertencia: No se encontró la clave de API de OpenAI. Algunas funciones pueden no estar disponibles o usar LLMs por defecto.")
else:
    Settings.llm = OpenAI(model="gpt-3.5-turbo", api_key=api_key)
    # También puedes configurar el modelo de embedding si lo deseas, por ejemplo:
    # from llama_index.embeddings.openai import OpenAIEmbedding
    # Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002", api_key=api_key)
    print("LLM de OpenAI configurado.")

print("Llama Index listo.")
```

*Nota sobre `Settings`*: Llama Index utiliza un objeto `Settings` global para configurar componentes como el LLM y el modelo de embedding. Esto permite una configuración sencilla que se aplica a todo el pipeline.

**Paso 5: Cargar tus Documentos**

Llama Index proporciona "Readers" (Lectores) para cargar datos de diversas fuentes. `SimpleDirectoryReader` es una forma fácil de cargar todos los archivos de un directorio.

```python
# ... (código anterior)

# Especificar el directorio que contiene tus datos
data_dir = "./data"

# Cargar los documentos desde el directorio
try:
    documents = SimpleDirectoryReader(data_dir).load_data()
    if not documents:
        print(f"No se encontraron documentos en el directorio: {data_dir}")
        print("Asegúrate de que 'mi_documento.txt' exista en la carpeta 'data'.")
        exit()
    print(f"Se cargaron {len(documents)} documento(s) desde '{data_dir}'.")
    # Cada 'documento' en la lista 'documents' es un objeto Document de Llama Index.
    # Puedes inspeccionar el contenido si lo deseas:
    # for doc in documents:
    #     print(f"---
{doc.text[:200]}...\n---")
except Exception as e:
    print(f"Error al cargar documentos: {e}")
    exit()
```

**Paso 6: Crear un Índice a partir de los Documentos**

Una vez que los documentos están cargados, puedes construir un índice. Un `VectorStoreIndex` es común: convierte tus documentos en embeddings numéricos y los almacena para una búsqueda semántica eficiente.

```python
# ... (código anterior)

# Crear un índice a partir de los documentos cargados
# Esto procesará los documentos, los dividirá en nodos (chunks), generará embeddings y los almacenará.
print("Creando el índice... Esto puede tardar un momento la primera vez.")
try:
    index = VectorStoreIndex.from_documents(documents)
    print("Índice creado exitosamente.")
except Exception as e:
    # Esto podría ocurrir si el LLM o el modelo de embedding no están configurados correctamente
    # o si hay problemas de conectividad con la API de OpenAI.
    print(f"Error al crear el índice: {e}")
    print("Asegúrate de que tu clave de API de OpenAI esté configurada correctamente en .env y sea válida.")
    exit()
```

**Paso 7: Crear un Motor de Consulta (Query Engine)**

Para interactuar con tu índice y hacer preguntas, necesitas un motor de consulta.

```python
# ... (código anterior)

# Crear un motor de consulta a partir del índice
query_engine = index.as_query_engine()
print("Motor de consulta listo.")
```

**Paso 8: Realizar Consultas y Obtener Respuestas**

Ahora puedes hacer preguntas a tu motor de consulta. El motor recuperará los fragmentos de texto más relevantes de tu índice y luego usará el LLM configurado para sintetizar una respuesta basada en esa información.

```python
# ... (código anterior)

# Realizar una consulta
pregunta = "¿Quién fue un pionero clave en la IA?"
print(f"\nPregunta: {pregunta}")

try:
    response = query_engine.query(pregunta)
    print(f"Respuesta: {response}")

    # Puedes inspeccionar los nodos fuente que Llama Index recuperó para generar la respuesta:
    # print("\nNodos fuente recuperados:")
    # for node in response.source_nodes:
    #     print(f"ID del Nodo: {node.node_id}, Score: {node.score:.4f}")
    #     print(f"Texto del Nodo:\n{node.text[:150]}...")
    #     print("---")

except Exception as e:
    print(f"Error al realizar la consulta: {e}")

# Otra pregunta
pregunta_2 = "¿Para qué sirve Llama Index?"
print(f"\nPregunta: {pregunta_2}")
try:
    response_2 = query_engine.query(pregunta_2)
    print(f"Respuesta: {response_2}")
except Exception as e:
    print(f"Error al realizar la consulta: {e}")

```

**Recursos y Documentación Adicional:**

*   Documentación de Llama Index - Inicio Rápido (Python): [https://docs.llamaindex.ai/en/stable/getting_started/starter_example.html](https://docs.llamaindex.ai/en/stable/getting_started/starter_example.html)
*   Documentación de Llama Index sobre Carga de Datos (Data Loaders): [https://docs.llamaindex.ai/en/stable/module_guides/loading/connector.html](https://docs.llamaindex.ai/en/stable/module_guides/loading/connector.html)
*   Documentación de Llama Index sobre Índices: [https://docs.llamaindex.ai/en/stable/module_guides/indexing/indexing.html](https://docs.llamaindex.ai/en/stable/module_guides/indexing/indexing.html)
*   Documentación de Llama Index sobre Motores de Consulta: [https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/root.html](https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/root.html)

**Criterios de Evaluación y Verificación:**

*   Tu script de Python (`mi_primer_indice.py`) se ejecuta sin errores.
*   El script carga correctamente los datos del archivo `mi_documento.txt`.
*   Se crea un índice sin errores (esto implica que los embeddings se generaron, lo que a su vez requiere una configuración de LLM/embedding funcional, por ejemplo, con la clave de API de OpenAI).
*   El motor de consulta responde a tus preguntas basándose en el contenido de `mi_documento.txt`.
*   Las respuestas son coherentes con la información presente en tu documento.

**Posibles Extensiones o Retos Adicionales:**

*   Añade más archivos de texto al directorio `data` y observa cómo `SimpleDirectoryReader` los carga todos.
*   Prueba con diferentes tipos de preguntas para ver cómo responde el motor de consulta.
*   Investiga otros tipos de índices en Llama Index (por ejemplo, `SummaryIndex` o `KeywordTableIndex`) y cómo podrían ser útiles para diferentes tareas.
*   Intenta cargar datos desde una fuente diferente, como un archivo PDF (esto podría requerir instalar dependencias adicionales como `pip install llama-index-readers-file`).
