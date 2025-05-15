# Reto Intermedio 1: Creando tu Primer Agente Cíclico con LangGraph

**Nivel de Dificultad:** Intermedio

**Herramientas Principales Involucradas:** LangGraph, Langchain, Python, Modelo de Lenguaje (ej. OpenAI)

**Conceptos Clave Abordados:** Grafos de Estado (State Graphs), Nodos (Nodes), Aristas (Edges), Aristas Condicionales (Conditional Edges), Gestión de Estado (State Management), Ciclos en Agentes, Herramientas (Tools), Orquestación de LLMs.

**Objetivos Específicos del Reto:**

*   Comprender el propósito de LangGraph para construir aplicaciones LLM con estado y ciclos.
*   Aprender a definir un grafo de estado que represente el flujo de trabajo de un agente.
*   Implementar nodos en el grafo como funciones de Python o `Runnable`s de Langchain que modifican el estado.
*   Definir aristas (edges) para conectar nodos y dirigir el flujo de información y control.
*   Implementar aristas condicionales para permitir que el agente tome decisiones y siga diferentes caminos basados en el estado actual.
*   Construir un agente simple que pueda realizar múltiples pasos, potencialmente en un ciclo (por ejemplo, usar una herramienta, evaluar el resultado y decidir el siguiente paso).
*   Gestionar y actualizar el estado de la aplicación a medida que fluye a través del grafo.
*   Compilar y ejecutar el grafo de LangGraph.

**Introducción Conceptual y Relevancia:**

LangGraph es una extensión de Langchain diseñada para construir aplicaciones LLM robustas y con estado, especialmente aquellas que requieren ciclos o flujos de control complejos, como los agentes. Mientras que las `Chain`s de Langchain son excelentes para secuencias lineales, muchos agentes necesitan la capacidad de llamar a herramientas repetidamente, reflexionar sobre los resultados y decidir dinámicamente qué hacer a continuación. LangGraph modela estas aplicaciones como grafos de estado. Cada "nodo" en el grafo es una función o un `Runnable` que realiza una acción (como llamar a un LLM, a una herramienta, o simplemente procesar datos). Las "aristas" conectan estos nodos, definiendo las posibles transiciones. Lo más importante es que LangGraph permite aristas condicionales, donde la siguiente transición se elige en tiempo de ejecución basándose en el estado actual, permitiendo bucles y lógica compleja. Este reto te guiará en la construcción de un agente simple con un ciclo de decisión, una capacidad fundamental para crear agentes de IA más autónomos y sofisticados.

**Requisitos Previos:**

*   Conocimientos sólidos de Python y programación orientada a objetos.
*   Haber completado el "Reto Básico 1: Creando tu Primera Cadena (Chain) con Langchain" y comprender los conceptos de LLMs, Prompts y Chains.
*   Familiaridad con el concepto de herramientas (Tools) en Langchain (aunque se introducirá una simple aquí).
*   Una cuenta de OpenAI y una clave de API configurada de forma segura.

**Instrucciones Detalladas Paso a Paso:**

Vamos a construir un agente simple que puede decidir si responder directamente a una pregunta o si necesita usar una "herramienta de búsqueda" (simulada) para encontrar más información antes de responder. Si usa la herramienta, volverá a evaluar si necesita más información o si ya puede responder.

**Paso 1: Instalación de las Bibliotecas Necesarias**

Asegúrate de tener `langgraph`, `langchain`, `langchain-openai` (si usas OpenAI), y `python-dotenv`.

```bash
pip install langgraph langchain langchain-openai python-dotenv
```

**Paso 2: Configuración de la Clave de API y el LLM**

Como en retos anteriores, configura tu clave de API de OpenAI en un archivo `.env`.

```python
import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence, Literal
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode # Para facilitar la creación de nodos de herramientas
from langchain_core.tools import tool # Para crear herramientas fácilmente

# Cargar variables de entorno
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("No se encontró la clave de API de OpenAI. Asegúrate de que esté configurada en tu archivo .env")

# Configurar el LLM
# Usaremos un modelo con capacidad de llamar a herramientas (tool calling)
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, openai_api_key=api_key)

print("LLM y dependencias listas.")
```

**Paso 3: Definición del Estado del Grafo (AgentState)**

El estado es un diccionario (o una clase `TypedDict`) que se pasa entre los nodos del grafo. Cada nodo puede leer y modificar este estado.

```python
# ... (código anterior)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # `operator.add` significa que cuando se actualiza `messages`, los nuevos mensajes se añaden a los existentes.
    # Esto es útil para mantener un historial de conversación.
```

**Paso 4: Definición de las Herramientas (Tools)**

Crearemos una herramienta de búsqueda simulada muy simple.

```python
# ... (código anterior)

@tool
def magic_search_tool(query: str) -> str:
    """Simula una búsqueda mágica. Si la consulta es sobre 'LangGraph', devuelve información específica.
    Para otras consultas, indica que no encontró nada específico."""
    print(f"---> Herramienta 'magic_search_tool' llamada con la consulta: {query}")
    if "langgraph" in query.lower():
        return "LangGraph es una biblioteca para construir aplicaciones LLM con estado y ciclos. Es útil para agentes complejos."
    return f"No se encontró información específica para '{query}' en la búsqueda mágica."

tools = [magic_search_tool]

# Langchain y LangGraph pueden trabajar con herramientas de forma más estructurada.
# Aquí, vincularemos las herramientas al LLM para que sepa cuándo y cómo llamarlas.
llm_with_tools = llm.bind_tools(tools)

print("Herramientas definidas y vinculadas al LLM.")
```

**Paso 5: Definición de los Nodos del Grafo**

Los nodos son funciones que toman el estado actual y devuelven una actualización del estado.

1.  **`call_model` (Agente Principal)**: Este nodo llamará al LLM. El LLM podría responder directamente o decidir llamar a una herramienta.
2.  **`call_tool` (Ejecutor de Herramientas)**: Este nodo se encarga de ejecutar la herramienta que el LLM decidió llamar.

```python
# ... (código anterior)

# Nodo 1: Agente que llama al LLM
def call_model(state: AgentState):
    """Llama al LLM con el historial de mensajes actual. 
    El LLM puede responder o solicitar una llamada a herramienta."""
    print("---NODO: call_model---")
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    # Devolvemos la respuesta del LLM para añadirla al estado
    return {"messages": [response]}

# Nodo 2: Ejecutor de herramientas
# LangGraph proporciona un `ToolNode` preconstruido que simplifica esto.
# Tomará el último mensaje (que debería ser una solicitud de herramienta del AIMessage)
# y ejecutará la herramienta, devolviendo un ToolMessage.
tool_node = ToolNode(tools)

print("Nodos definidos.")
```

**Paso 6: Definición de las Aristas Condicionales**

Necesitamos una función que decida qué hacer después de que el LLM (en `call_model`) haya respondido. ¿Debería terminar, o debería llamar a una herramienta?

```python
# ... (código anterior)

def should_continue(state: AgentState) -> Literal["action", "end"]:
    """Decide la siguiente acción basada en la última respuesta del LLM."""
    print("---NODO CONDICIONAL: should_continue---")
    last_message = state["messages"][-1]
    # Si el LLM hizo una llamada a herramienta, entonces vamos al nodo de acción (tool_node)
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        print("Decisión: LLM solicitó una herramienta. Ir a 'action'.")
        return "action"
    # Si no hay llamada a herramienta, el LLM ha respondido, entonces terminamos.
    print("Decisión: LLM respondió. Ir a 'end'.")
    return "end"

print("Lógica condicional definida.")
```

**Paso 7: Construcción del Grafo**

Ahora ensamblamos los nodos y las aristas en un `StateGraph`.

```python
# ... (código anterior)

# Crear el grafo de estado
workflow = StateGraph(AgentState)

# Añadir los nodos al grafo
workflow.add_node("agent", call_model) # Nodo donde el agente (LLM) decide
workflow.add_node("action", tool_node)  # Nodo que ejecuta la acción/herramienta

# Definir el punto de entrada del grafo
workflow.set_entry_point("agent")

# Añadir las aristas condicionales
workflow.add_conditional_edges(
    "agent", # Nodo de origen
    should_continue, # Función que decide la ruta
    {
        "action": "action", # Si should_continue devuelve "action", ir al nodo "action"
        "end": END          # Si should_continue devuelve "end", terminar el grafo
    }
)

# Añadir una arista desde el nodo de acción de vuelta al agente
# Después de ejecutar una herramienta, queremos que el agente procese el resultado de la herramienta.
workflow.add_edge("action", "agent")

# Compilar el grafo en una aplicación ejecutable
app = workflow.compile()
print("Grafo compilado y listo.")
```

**Paso 8: Ejecutar el Grafo**

Podemos interactuar con el grafo enviándole mensajes.

```python
# ... (código anterior)

# Ejecutar el grafo con una entrada inicial
inputs1 = {"messages": [HumanMessage(content="Hola, ¿cómo estás?")]}
try:
    print("\n--- Ejecución 1: Saludo Simple ---")
    for event in app.stream(inputs1):
        for key, value in event.items():
            print(f"Evento del grafo: Nodo='{key}', Estado actualizado='{value}'")
        print("---")
    # La respuesta final estará en el último estado del nodo 'agent' o 'end'
    final_state_1 = app.invoke(inputs1)
    print(f"Respuesta final (Saludo): {final_state_1['messages'][-1].content}")
except Exception as e:
    print(f"Error en la ejecución 1: {e}")


# Ejecutar con una pregunta que podría requerir la herramienta
inputs2 = {"messages": [HumanMessage(content="¿Qué sabes sobre LangGraph?")]}
try:
    print("\n--- Ejecución 2: Pregunta sobre LangGraph (debería usar herramienta) ---")
    for event in app.stream(inputs2, {"recursion_limit": 5}): # Añadir límite de recursión
        for key, value in event.items():
            print(f"Evento del grafo: Nodo='{key}', Estado actualizado='{value}'")
        print("---")
    final_state_2 = app.invoke(inputs2, {"recursion_limit": 5})
    print(f"Respuesta final (LangGraph): {final_state_2['messages'][-1].content}")
except Exception as e:
    print(f"Error en la ejecución 2: {e}")


# Ejecutar con una pregunta que la herramienta no cubre
inputs3 = {"messages": [HumanMessage(content="¿Cuál es la capital de Francia?")]}
try:
    print("\n--- Ejecución 3: Pregunta general (no debería usar herramienta específica) ---")
    for event in app.stream(inputs3, {"recursion_limit": 5}):
        for key, value in event.items():
            print(f"Evento del grafo: Nodo='{key}', Estado actualizado='{value}'")
        print("---")
    final_state_3 = app.invoke(inputs3, {"recursion_limit": 5})
    print(f"Respuesta final (Capital): {final_state_3['messages'][-1].content}")
except Exception as e:
    print(f"Error en la ejecución 3: {e}")

```

**Recursos y Documentación Adicional:**

*   Documentación de LangGraph: [https://python.langchain.com/docs/langgraph/](https://python.langchain.com/docs/langgraph/)
*   Introducción a LangGraph: [https://python.langchain.com/docs/langgraph/introduction/](https://python.langchain.com/docs/langgraph/introduction/)
*   Ejemplos de LangGraph (especialmente agentes con herramientas): [https://python.langchain.com/docs/langgraph/how-tos/agent_executor/](https://python.langchain.com/docs/langgraph/how-tos/agent_executor/) (Aunque este ejemplo construye uno más simple desde cero).
*   Langchain Tool Calling: [https://python.langchain.com/docs/modules/tools/toolkits/](https://python.langchain.com/docs/modules/tools/toolkits/)

**Criterios de Evaluación y Verificación:**

*   Tu script se ejecuta sin errores.
*   Cuando haces una pregunta simple (como "Hola"), el agente responde directamente sin llamar a la herramienta.
*   Cuando preguntas sobre "LangGraph", el `magic_search_tool` es invocado (verás el `print` de la herramienta), y la respuesta final del agente incluye la información de la herramienta.
*   Cuando haces una pregunta general que la herramienta no cubre específicamente (como "¿Cuál es la capital de Francia?"), el agente puede intentar usar la herramienta, obtener una respuesta genérica de ella, y luego el LLM debería generar una respuesta final (posiblemente indicando que la herramienta no ayudó mucho o respondiendo desde su conocimiento general).
*   Puedes seguir el flujo de ejecución a través de los `print` de los nodos y la lógica condicional.

**Posibles Extensiones o Retos Adicionales:**

*   Añade más herramientas al agente y modifica la lógica condicional `should_continue` o el nodo `call_model` para manejar múltiples herramientas.
*   Implementa un contador de reintentos en el estado para evitar bucles infinitos si la herramienta falla repetidamente o no proporciona la información necesaria.
*   Modifica el estado para incluir un "plan" o una "pregunta intermedia" que el agente pueda refinar después de cada llamada a herramienta.
*   Integra un recuperador RAG (como el del Reto Básico 3) como una de las herramientas disponibles para el agente LangGraph.
