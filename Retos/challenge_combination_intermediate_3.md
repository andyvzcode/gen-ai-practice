# Reto Intermedio 3: Agente de Investigación Asistido por RAG y Orquestado con LangGraph

**Nivel de Dificultad:** Intermedio

**Herramientas Principales Involucradas:** LangGraph, Llama Index (para RAG), Langchain, Python, Modelo de Lenguaje (ej. OpenAI)

**Conceptos Clave Abordados:** Integración de Sistemas RAG como Herramientas, Agentes Basados en Grafos de Estado, Planificación de Tareas por un Agente, Ejecución de Herramientas (incluyendo RAG), Síntesis de Información de Múltiples Fuentes, Ciclos de Razonamiento y Acción.

**Objetivos Específicos del Reto:**

*   Comprender cómo integrar un sistema de Retrieval Augmented Generation (RAG) construido con Llama Index como una herramienta (Tool) para un agente más complejo.
*   Diseñar y construir un agente utilizando LangGraph que pueda planificar y ejecutar una serie de pasos para responder a una pregunta compleja.
*   Implementar un nodo en LangGraph que permita al agente consultar el sistema RAG (Llama Index Query Engine).
*   Gestionar el estado del agente en LangGraph para acumular información, mantener el historial de la conversación y rastrear los pasos de investigación.
*   Crear un flujo de trabajo cíclico donde el agente pueda consultar el RAG, evaluar la respuesta, y decidir si necesita más información (otra consulta al RAG o a otra herramienta) o si puede sintetizar una respuesta final.
*   Sintetizar una respuesta final utilizando un LLM, basada en la información recopilada a través de las interacciones con el sistema RAG.

**Introducción Conceptual y Relevancia:**

En los retos anteriores, exploramos RAG con Llama Index para consultar bases de conocimiento y LangGraph para construir agentes con flujos de control complejos. Este reto une esos dos mundos. A menudo, un agente necesita acceder a un cuerpo de conocimiento específico para realizar sus tareas. En lugar de que el LLM del agente intente recordar toda la información (lo cual es propenso a errores o información desactualizada), podemos proporcionarle una herramienta RAG. El agente, orquestado por LangGraph, puede entonces decidir cuándo y cómo usar esta herramienta RAG para buscar información, y luego usar esa información para razonar y generar respuestas o tomar acciones. Este patrón es extremadamente poderoso para construir agentes de investigación, asistentes de soporte técnico que consultan manuales, o cualquier aplicación donde un agente necesite fundamentar sus respuestas en un conjunto de documentos específico. Aprenderás a encapsular tu motor de consulta de Llama Index como una herramienta de Langchain y a integrarlo en un ciclo de razonamiento de LangGraph.

**Requisitos Previos:**

*   Conocimientos sólidos de Python.
*   Haber completado y comprendido:
    *   "Reto Básico 2: Indexando y Consultando tus Primeros Documentos con Llama Index" (para la parte RAG).
    *   "Reto Básico 3: Construyendo tu Primer Sistema RAG"
    *   "Reto Intermedio 1: Creando tu Primer Agente Cíclico con LangGraph"
*   Familiaridad con la creación de `Tool`s en Langchain.
*   Una cuenta de OpenAI y una clave de API configurada de forma segura.

**Instrucciones Detalladas Paso a Paso:**

Construiremos un agente que investiga un tema. Primero, intentará responder desde su conocimiento general. Si no puede, o si necesita más detalles, usará una herramienta RAG (construida con Llama Index) para buscar en un conjunto de documentos. Luego, sintetizará una respuesta final.

**Paso 1: Instalación de Bibliotecas**

Asegúrate de tener `langgraph`, `llama-index`, `langchain`, `langchain-openai`, `python-dotenv`.

```bash
pip install langgraph llama-index langchain langchain-openai python-dotenv
```

**Paso 2: Preparación de Datos para RAG y Configuración de API**

1.  Crea un directorio `data_rag_agent` en tu proyecto.
2.  Dentro de `data_rag_agent`, crea algunos archivos de texto con información sobre un tema específico. Por ejemplo, sobre "Energías Renovables".
    *   `solar.txt`: "La energía solar se obtiene del sol. Los paneles fotovoltaicos convierten la luz solar en electricidad. Es una fuente limpia y abundante."
    *   `eolica.txt`: "La energía eólica utiliza turbinas de viento para generar electricidad. Es efectiva en áreas con vientos consistentes. Su impacto visual es una consideración."
    *   `geotermica.txt`: "La energía geotérmica aprovecha el calor del interior de la Tierra. Es una fuente constante pero su disponibilidad geográfica es limitada."
3.  Configura tu `OPENAI_API_KEY` en un archivo `.env`.

**Paso 3: Script Inicial y Configuración del LLM**

```python
import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence, Literal, List, Union
import operator

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool, BaseTool
from langchain_openai import ChatOpenAI

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings as LlamaSettings
from llama_index.llms.openai import OpenAI as LlamaOpenAI # LLM para LlamaIndex
from llama_index.embeddings.openai import OpenAIEmbedding as LlamaOpenAIEmbedding # Embeddings para LlamaIndex
from llama_index.core.query_engine import BaseQueryEngine

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Cargar variables de entorno
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("No se encontró la clave de API de OpenAI.")

# LLM para LangGraph (Agente principal)
agent_llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0, openai_api_key=api_key)

# Configuración para Llama Index (puede ser diferente si se desea)
LlamaSettings.llm = LlamaOpenAI(model="gpt-3.5-turbo", api_key=api_key)
LlamaSettings.embed_model = LlamaOpenAIEmbedding(model="text-embedding-ada-002", api_key=api_key)

print("LLMs y dependencias listas.")
```

**Paso 4: Creación del Sistema RAG (Llama Index) y la Herramienta RAG**

Primero, creamos el motor de consulta de Llama Index. Luego, lo envolvemos en una `Tool` de Langchain.

```python
# ... (código anterior)

# Crear el motor de consulta RAG de Llama Index
def create_rag_query_engine(data_path: str) -> BaseQueryEngine:
    print(f"Cargando documentos desde: {data_path}")
    documents = SimpleDirectoryReader(data_path).load_data()
    if not documents:
        raise ValueError(f"No se encontraron documentos en {data_path}")
    print(f"Creando índice RAG para {len(documents)} documentos...")
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(similarity_top_k=2) # Recuperar 2 fragmentos
    print("Motor de consulta RAG creado.")
    return query_engine

rag_data_path = "./data_rag_agent"
rag_query_engine = create_rag_query_engine(rag_data_path)

# Envolver el motor de consulta RAG en una Langchain Tool
@tool
def rag_search_tool(query: str) -> str:
    """Busca información en la base de conocimiento sobre energías renovables. 
    Úsalo para preguntas específicas sobre energía solar, eólica o geotérmica."""
    print(f"---> Herramienta RAG llamada con la consulta: {query}")
    response = rag_query_engine.query(query)
    return str(response)

tools = [rag_search_tool]
agent_llm_with_tools = agent_llm.bind_tools(tools)

print("Herramienta RAG definida y vinculada al LLM del agente.")
```

**Paso 5: Definición del Estado del Grafo (AgentState)**

El estado contendrá los mensajes, la pregunta original y la información recopilada.

```python
# ... (código anterior)

class ResearchAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    original_question: str
    research_steps: Annotated[List[str], operator.add]
    final_answer: Union[str, None]
```

**Paso 6: Definición de los Nodos del Grafo**

1.  `planner_agent_node`: El LLM principal que decide qué hacer (responder, usar herramienta RAG).
2.  `rag_tool_node`: Ejecuta la herramienta RAG.
3.  `final_answer_node`: Genera la respuesta final (podría ser el mismo LLM que el planner, pero con un prompt diferente).

```python
# ... (código anterior)

# Nodo 1: Agente Planificador/Decisor
def planner_agent_node(state: ResearchAgentState):
    print("---NODO: planner_agent_node---")
    # El prompt podría ser más sofisticado, indicando que ya ha intentado X pasos
    current_messages = state["messages"]
    if not any(isinstance(msg, SystemMessage) for msg in current_messages):
        # Añadir un SystemMessage si no existe para guiar mejor al LLM
        system_prompt = SystemMessage(content=(
            "Eres un asistente de investigación. Tu objetivo es responder a la pregunta del usuario. "
            "Primero intenta responder desde tu conocimiento general. Si no puedes o necesitas detalles específicos "
            "sobre energías renovables (solar, eólica, geotérmica), debes usar la herramienta 'rag_search_tool'. "
            "Después de usar la herramienta, evalúa si tienes suficiente información o si necesitas volver a usarla. "
            "Una vez que tengas suficiente información, indica que estás listo para dar una respuesta final."
        ))
        current_messages = [system_prompt] + list(current_messages)

    response = agent_llm_with_tools.invoke(current_messages)
    return {"messages": [response], "research_steps": [f"Planner LLM: {response.content}"]}

# Nodo 2: Ejecutor de la herramienta RAG (usando ToolNode preconstruido)
rag_tool_node = ToolNode([rag_search_tool]) # Pasamos la lista de herramientas

# Nodo 3: Nodo para generar la respuesta final (opcional, o el planner puede hacerlo)
# Por simplicidad, dejaremos que el planner_agent_node también maneje la generación de la respuesta final
# cuando decida que tiene suficiente información (es decir, no llama a una herramienta).

print("Nodos del grafo definidos.")
```

**Paso 7: Definición de las Aristas Condicionales**

Esta función decidirá si continuar con la herramienta RAG o si el agente está listo para finalizar.

```python
# ... (código anterior)

MAX_ITERATIONS = 3 # Para evitar bucles infinitos

def should_use_rag_or_end(state: ResearchAgentState) -> Literal["use_rag_tool", "end_research"]:
    print("---NODO CONDICIONAL: should_use_rag_or_end---")
    last_message = state["messages"][-1]
    current_iteration = len([step for step in state.get("research_steps", []) if "Planner LLM:" in step or "Tool Result:" in step])

    if current_iteration >= MAX_ITERATIONS:
        print(f"Límite de iteraciones ({MAX_ITERATIONS}) alcanzado. Finalizando.")
        return "end_research"

    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        # Específicamente verificar si la herramienta es rag_search_tool
        if any(tc.name == rag_search_tool.name for tc in last_message.tool_calls):
            print("Decisión: LLM solicitó la herramienta RAG. Ir a 'use_rag_tool'.")
            return "use_rag_tool"
    
    print("Decisión: LLM no solicitó la herramienta RAG o es una respuesta. Finalizando para síntesis.")
    return "end_research"

print("Lógica condicional definida.")
```

**Paso 8: Construcción del Grafo**

```python
# ... (código anterior)

workflow = StateGraph(ResearchAgentState)

workflow.add_node("planner_agent", planner_agent_node)
workflow.add_node("rag_tool_executor", rag_tool_node)

workflow.set_entry_point("planner_agent")

workflow.add_conditional_edges(
    "planner_agent",
    should_use_rag_or_end,
    {
        "use_rag_tool": "rag_tool_executor",
        "end_research": END 
    }
)

# Después de ejecutar la herramienta RAG, volvemos al agente planificador para que evalúe
workflow.add_edge("rag_tool_executor", "planner_agent")

app = workflow.compile()
print("Grafo de investigación compilado.")
```

**Paso 9: Ejecutar el Grafo de Investigación**

```python
# ... (código anterior)

initial_question = "¿Cuáles son las ventajas de la energía solar según la base de conocimiento?"
inputs = {
    "messages": [HumanMessage(content=initial_question)],
    "original_question": initial_question,
    "research_steps": [],
    "final_answer": None
}

print(f"\n--- Iniciando Agente de Investigación para: 
{initial_question} ---")

# Usar stream para ver los eventos y el estado en cada paso
for event in app.stream(inputs, {"recursion_limit": 10}):
    for key, value in event.items():
        print(f"Evento del Grafo: Nodo='{key}'")
        # print(f"Estado actualizado: {value}") # Puede ser muy verboso
        if "messages" in value:
            print(f"  Último mensaje: {value['messages'][-1].pretty_repr()[:300]}...")
        if "research_steps" in value and value["research_steps"]:
             print(f"  Último paso de investigación: {value['research_steps'][-1]}")
    print("---")

final_state = app.invoke(inputs, {"recursion_limit": 10})
final_llm_response_message = final_state["messages"][-1]

print("\n--- Resultado Final de la Investigación ---")
print(f"Pregunta Original: {final_state['original_question']}")
print(f"Respuesta del Agente: {final_llm_response_message.content}")

# Segunda prueba con una pregunta más general
initial_question_2 = "Háblame sobre las energías renovables en general."
inputs_2 = {
    "messages": [HumanMessage(content=initial_question_2)],
    "original_question": initial_question_2,
    "research_steps": [],
    "final_answer": None
}
print(f"\n--- Iniciando Agente de Investigación para: {initial_question_2} ---")
for event in app.stream(inputs_2, {"recursion_limit": 10}):
    for key, value in event.items():
        print(f"Evento del Grafo: Nodo='{key}'")
    print("---")
final_state_2 = app.invoke(inputs_2, {"recursion_limit": 10})
print(f"Respuesta del Agente: {final_state_2['messages'][-1].content}")

```

**Recursos y Documentación Adicional:**

*   Revisar la documentación de LangGraph sobre agentes con herramientas y ciclos.
*   Revisar la documentación de Llama Index sobre `QueryEngine` y cómo se pueden personalizar.
*   Langchain - Documentación sobre Herramientas (Tools): [https://python.langchain.com/docs/modules/tools/](https://python.langchain.com/docs/modules/tools/)

**Criterios de Evaluación y Verificación:**

*   El script se ejecuta sin errores.
*   El sistema RAG (Llama Index) se inicializa correctamente con los documentos proporcionados.
*   Cuando se hace una pregunta que requiere información de los documentos RAG (ej., "ventajas de la energía solar"), el `planner_agent_node` decide usar la `rag_search_tool`.
*   El `rag_tool_node` ejecuta la consulta contra el motor de Llama Index y devuelve los resultados.
*   El `planner_agent_node` recibe los resultados de la herramienta RAG y los utiliza para formular una respuesta final (o decide hacer otra consulta si es necesario, dentro del límite de iteraciones).
*   Para preguntas generales que no están cubiertas por los documentos RAG, el agente podría intentar responder desde su conocimiento general o indicar que la herramienta RAG no proporcionó información relevante.
*   Puedes seguir el flujo de decisiones y la acumulación de información a través de la salida detallada.

**Posibles Extensiones o Retos Adicionales:**

*   Añade más herramientas al `planner_agent_node` (por ejemplo, una herramienta de búsqueda web general además de la herramienta RAG específica) y mejora la lógica de `should_use_rag_or_end` para decidir qué herramienta usar o si usar múltiples herramientas secuencialmente.
*   Implementa un nodo de "síntesis" explícito que tome toda la información recopilada en `state['research_steps']` o los mensajes de herramientas y genere un informe final cohesivo.
*   Permite que el agente haga múltiples llamadas a la herramienta RAG si la primera no es suficiente, refinando su consulta cada vez.
*   Integra este agente LangGraph como un `Tool` para un agente de Crew AI, creando una jerarquía de agentes donde una tripulación puede delegar tareas de investigación complejas a este especialista en RAG orquestado por LangGraph.
