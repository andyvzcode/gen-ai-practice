# Reto Básico 1: Creando tu Primera Cadena (Chain) con Langchain

**Nivel de Dificultad:** Básico

**Herramientas Principales Involucradas:** Langchain, Python, Modelo de Lenguaje (ej. OpenAI)

**Conceptos Clave Abordados:** Modelos de Lenguaje (LLMs), Prompts, Cadenas (Chains), Plantillas de Prompt (Prompt Templates), Claves de API.

**Objetivos Específicos del Reto:**

*   Instalar las bibliotecas Langchain y OpenAI.
*   Configurar de forma segura tu clave de API de OpenAI.
*   Comprender el concepto fundamental de una "Cadena" (Chain) en Langchain como una secuencia de llamadas a componentes.
*   Entender el rol de los Modelos de Lenguaje (LLMs) como componentes centrales en Langchain.
*   Aprender a crear y utilizar Plantillas de Prompt (`PromptTemplate`) para formatear entradas a los LLMs.
*   Construir y ejecutar una cadena simple (`LLMChain`) que combine un LLM con una plantilla de prompt.
*   Pasar datos de entrada a la cadena y observar la salida generada por el LLM.

**Introducción Conceptual y Relevancia:**

Langchain es un potente framework diseñado para simplificar el desarrollo de aplicaciones que utilizan Modelos de Lenguaje Grandes (LLMs). En lugar de interactuar directamente con las APIs de los LLMs de forma aislada, Langchain proporciona abstracciones y herramientas para construir aplicaciones más complejas y modulares. El concepto central en Langchain es la "Cadena" (Chain), que representa una secuencia de llamadas, ya sea a un LLM, a una herramienta (como una búsqueda en Google) o a otra cadena. Este primer reto se enfoca en la cadena más fundamental: una `LLMChain`, que conecta una plantilla de prompt con un LLM. Dominar este concepto es el primer paso esencial para construir aplicaciones de IA más sofisticadas con Langchain, como chatbots, sistemas de resumen de texto, o agentes que pueden razonar y actuar.

**Requisitos Previos:**

*   Conocimientos básicos de Python (variables, funciones, tipos de datos).
*   Una cuenta de OpenAI y una clave de API. Puedes obtenerla desde [platform.openai.com](https://platform.openai.com/).

**Instrucciones Detalladas Paso a Paso:**

**Paso 1: Instalación de las Bibliotecas Necesarias**

Primero, necesitas instalar las bibliotecas de Langchain y OpenAI. Abre tu terminal o línea de comandos y ejecuta el siguiente comando pip:

```bash
pip install langchain openai python-dotenv
```

*   `langchain`: La biblioteca principal de Langchain.
*   `openai`: La biblioteca cliente de Python para interactuar con la API de OpenAI.
*   `python-dotenv`: Una utilidad para gestionar variables de entorno, útil para manejar tu clave de API de forma segura.

**Paso 2: Configuración de tu Clave de API de OpenAI**

Para usar los modelos de OpenAI a través de Langchain, necesitas configurar tu clave de API. Es una **mala práctica** escribir tu clave directamente en el código. En su lugar, la guardaremos en un archivo `.env`.

1.  Crea un archivo llamado `.env` en el mismo directorio donde crearás tu script de Python.
2.  Abre el archivo `.env` con un editor de texto y añade la siguiente línea, reemplazando `TU_CLAVE_DE_API_DE_OPENAI` con tu clave real:
    ```
    OPENAI_API_KEY="TU_CLAVE_DE_API_DE_OPENAI"
    ```
3.  Guarda el archivo `.env`.

Ahora, en tu script de Python, puedes cargar esta clave de forma segura.

**Paso 3: Creación de tu Primer Script de Langchain**

Crea un archivo Python (por ejemplo, `mi_primera_cadena.py`) y comienza importando las clases necesarias y cargando tu clave de API:

```python
import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Recuperar la clave de API de OpenAI de las variables de entorno
api_key = os.getenv("OPENAI_API_KEY")

# Verificar si la clave de API se cargó correctamente (opcional pero recomendado)
if not api_key:
    raise ValueError("No se encontró la clave de API de OpenAI. Asegúrate de que esté configurada en tu archivo .env")

print("Clave de API cargada exitosamente.")
```

**Paso 4: Inicializar el Modelo de Lenguaje (LLM)**

Langchain se integra con varios proveedores de LLMs. En este reto, usaremos OpenAI. Debes inicializar el LLM, especificando el modelo que deseas usar (por ejemplo, `text-davinci-003`, aunque modelos más nuevos como los de la familia GPT-3.5 o GPT-4 son recomendables si tienes acceso) y la clave de API.

```python
# ... (código anterior)

# Inicializar el LLM de OpenAI
# El parámetro 'temperature' controla la aleatoriedad de la salida. 0 es más determinista.
llm = OpenAI(openai_api_key=api_key, temperature=0.7)

print("LLM inicializado.")
```

**Paso 5: Crear una Plantilla de Prompt (PromptTemplate)**

Los prompts son las instrucciones que le das al LLM. Langchain facilita la creación de prompts dinámicos usando `PromptTemplate`. Esta plantilla puede tener variables que se rellenarán más tarde.

```python
# ... (código anterior)

# Definir la plantilla del prompt
# Esta plantilla espera una variable de entrada llamada 'producto'.
prompt_template_texto = "Sugiere un nombre creativo para una empresa que fabrica {producto}."

# Crear una instancia de PromptTemplate
prompt = PromptTemplate(
    input_variables=["producto"],
    template=prompt_template_texto
)

print(f"Plantilla de prompt creada: {prompt.template}")
```

**Paso 6: Construir la Cadena (LLMChain)**

Ahora, combinaremos el LLM y la plantilla de prompt en una `LLMChain`. Esta cadena tomará la entrada del usuario, la formateará usando la plantilla de prompt y luego pasará el prompt formateado al LLM para obtener una respuesta.

```python
# ... (código anterior)

# Crear la LLMChain
cadena = LLMChain(llm=llm, prompt=prompt)

print("LLMChain creada.")
```

**Paso 7: Ejecutar la Cadena y Obtener Resultados**

Finalmente, puedes ejecutar la cadena proporcionando un valor para la variable de entrada definida en tu plantilla de prompt (en este caso, `producto`).

```python
# ... (código anterior)

# Definir el producto para el cual queremos un nombre de empresa
nombre_producto = "calcetines de lana coloridos"

# Ejecutar la cadena con la entrada
# La entrada se pasa como un diccionario donde las claves coinciden con las input_variables del prompt
respuesta = cadena.run(nombre_producto)

# Imprimir la respuesta del LLM
print(f"\nPara un producto: {nombre_producto}")
print(f"Nombre de empresa sugerido: {respuesta.strip()}")
```

**Paso 8: Experimentar (Opcional pero Recomendado)**

Intenta cambiar el `nombre_producto` a diferentes cosas y observa cómo cambian las sugerencias del LLM. También puedes experimentar con el parámetro `temperature` en la inicialización de `OpenAI` (valores más altos como 0.9 darán respuestas más creativas pero menos predecibles, valores más bajos como 0.2 darán respuestas más enfocadas y deterministas).

**Recursos y Documentación Adicional:**

*   Documentación de Langchain sobre Cadenas (Chains): [https://python.langchain.com/docs/modules/chains/](https://python.langchain.com/docs/modules/chains/)
*   Documentación de Langchain sobre LLMs: [https://python.langchain.com/docs/modules/model_io/llms/](https://python.langchain.com/docs/modules/model_io/llms/)
*   Documentación de Langchain sobre Plantillas de Prompt: [https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/](https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/)
*   Modelos de OpenAI: [https://platform.openai.com/docs/models](https://platform.openai.com/docs/models)

**Criterios de Evaluación y Verificación:**

*   Tu script de Python (`mi_primera_cadena.py`) se ejecuta sin errores.
*   El script carga correctamente la clave de API de OpenAI desde el archivo `.env`.
*   El script imprime el nombre de empresa sugerido por el LLM.
*   La salida del LLM es coherente con el producto que proporcionaste como entrada.

**Posibles Extensiones o Retos Adicionales:**

*   Modifica la plantilla de prompt para pedir algo diferente, por ejemplo, un eslogan para el producto en lugar de un nombre de empresa.
*   Investiga cómo usar un modelo de LLM diferente de OpenAI (si tienes acceso a otros, o explora los modelos gratuitos que Langchain podría soportar).
*   Intenta crear una plantilla de prompt que acepte múltiples variables de entrada (por ejemplo, `producto` y `publico_objetivo`) y modifica la cadena para que las utilice.
