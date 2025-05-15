# Reto Avanzado (Agentes): Diseño de Agentes Complejos con Memoria Persistente y Herramientas Dinámicas

**Nivel de Dificultad:** Avanzado

**Herramientas Principales Involucradas:** Langchain, LangGraph, Python, (Opcional) FAISS o similar para memoria vectorial, APIs externas para herramientas.

**Conceptos Clave Abordados:** Arquitecturas de Agentes Avanzadas (ej. ReAct, Plan-and-Execute), Memoria Persistente (Buffer, Vectorial, Base de Datos), Gestión de Conversaciones Largas, Selección Dinámica de Herramientas, Creación de Herramientas Personalizadas, Manejo de Errores en Herramientas, Evaluación de Agentes (conceptual).

**Objetivos Específicos del Reto:**

*   Diseñar e implementar un agente capaz de realizar tareas complejas que requieran múltiples pasos y el uso de diversas herramientas.
*   Integrar un mecanismo de memoria persistente que permita al agente recordar interacciones pasadas a través de múltiples sesiones o ejecuciones (por ejemplo, usando un archivo, una base de datos simple, o un almacén de vectores para memoria semántica).
*   Implementar la capacidad para que el agente seleccione y utilice herramientas de forma dinámica basándose en la tarea actual y el estado de la conversación.
*   Crear al menos una herramienta personalizada compleja que el agente pueda utilizar (por ejemplo, una herramienta que interactúe con una API externa o realice un procesamiento de datos específico).
*   Implementar un manejo de errores robusto para el uso de herramientas, permitiendo al agente reintentar o buscar alternativas si una herramienta falla.
*   Utilizar LangGraph para orquestar el flujo de control del agente, permitiendo ciclos de pensamiento, acción y observación más complejos.
*   Reflexionar sobre cómo se podría evaluar el rendimiento y la fiabilidad del agente desarrollado.

**Introducción Conceptual y Relevancia:**

Los agentes de IA están evolucionando rápidamente de simples ejecutores de tareas a sistemas más autónomos capaces de razonar, planificar y adaptarse. Para construir agentes verdaderamente útiles y robustos, es crucial ir más allá de los ejemplos básicos y abordar desafíos como la memoria a largo plazo, la toma de decisiones sofisticada sobre el uso de herramientas y la capacidad de manejar conversaciones o tareas que se extienden en el tiempo.

Este reto te sumergirá en el diseño de un agente avanzado. Explorarás cómo dotar a tu agente de una memoria que persista entre interacciones, permitiéndole aprender de experiencias pasadas o continuar tareas interrumpidas. También te enfrentarás al desafío de crear y gestionar un conjunto de herramientas, donde el agente no solo las usa, sino que puede decidir dinámicamente cuál es la más apropiada para la situación actual. LangGraph será una herramienta clave para modelar los complejos flujos de control internos de dicho agente.

**Requisitos Previos:**

*   Haber completado los retos intermedios de Langchain, LangGraph y, preferiblemente, Crew AI y RAG.
*   Comprensión sólida de los conceptos de agentes en Langchain (tipos de agentes, herramientas, ejecutores).
*   Experiencia con LangGraph para construir flujos de estado.
*   Conocimientos de Python, incluyendo manejo de archivos, y opcionalmente, interacción con bases de datos simples (como SQLite) o APIs.
*   Familiaridad con el concepto de embeddings y almacenes de vectores si se opta por memoria semántica persistente.

**Instrucciones Detalladas Paso a Paso:**

**Paso 1: Definición del Agente y su Propósito**

1.  **Define un Propósito Específico:** Elige una tarea o un dominio para tu agente. Ejemplos:
    *   Un agente de investigación personal que puede buscar información en la web, resumirla, guardarla y responder preguntas sobre temas investigados previamente.
    *   Un asistente de planificación de proyectos que ayuda a definir tareas, asignar prioridades y hacer seguimiento del progreso, recordando el estado del proyecto entre sesiones.
    *   Un agente de soporte técnico que puede consultar una base de conocimiento (RAG), guiar al usuario a través de pasos de solución de problemas y recordar interacciones previas con ese usuario.
2.  **Identifica las Capacidades Clave:** ¿Qué necesita ser capaz de hacer tu agente? ¿Qué información necesita recordar?

**Paso 2: Diseño de la Arquitectura del Agente con LangGraph**

1.  **Define el Estado del Agente (`AgentState`):** Este será más complejo que en retos anteriores. Considera incluir:
    *   Historial de la conversación (mensajes de entrada, respuestas del agente, observaciones de herramientas).
    *   Estado interno del agente (ej. plan actual, tarea actual, información recopilada).
    *   Contenido de la memoria persistente (o un puntero/identificador a ella).
2.  **Diseña los Nodos del Grafo:**
    *   Un nodo para recibir la entrada del usuario.
    *   Un nodo "pensador" (LLM) que decide la siguiente acción (usar una herramienta, responder directamente, pedir aclaración).
    *   Nodos para ejecutar cada herramienta (o un `ToolNode` genérico de LangGraph).
    *   Un nodo para actualizar la memoria persistente.
    *   Un nodo para generar la respuesta final al usuario.
3.  **Diseña las Aristas Condicionales:** La lógica de transición será crucial. Por ejemplo:
    *   Después de pensar, ¿el agente necesita una herramienta? ¿Cuál?
    *   Después de usar una herramienta, ¿necesita pensar de nuevo o puede responder?
    *   ¿Cuándo se actualiza la memoria?

**Paso 3: Implementación de la Memoria Persistente**

Elige un mecanismo de persistencia. Aquí algunas opciones, de simple a más compleja:

1.  **Archivo de Texto/JSON Simple:**
    *   Guarda el historial de conversación o notas clave en un archivo.
    *   Carga este archivo al inicio de una nueva sesión.
    *   **Pros:** Fácil de implementar.
    *   **Contras:** No escalable, difícil para búsquedas semánticas.
2.  **Base de Datos SQLite:**
    *   Crea tablas para almacenar mensajes, entidades extraídas, resúmenes, etc.
    *   **Pros:** Más estructurado, permite consultas SQL.
    *   **Contras:** Requiere conocimientos de SQL.
3.  **Almacén de Vectores (FAISS, ChromaDB) para Memoria Semántica:**
    *   Guarda embeddings de fragmentos de conversación o información importante.
    *   Permite al agente recuperar recuerdos relevantes basados en la similitud semántica con la consulta actual.
    *   **Pros:** Potente para recordar información contextualmente relevante.
    *   **Contras:** Más complejo de configurar e integrar; puede requerir un modelo de embedding.

**Implementación:**

*   Crea funciones para `cargar_memoria()` y `guardar_memoria(estado_actual)`.
*   Integra estas funciones en los nodos apropiados de tu grafo LangGraph (por ejemplo, cargar al inicio, guardar después de cada interacción significativa o al final de la sesión).

**Paso 4: Creación y Selección Dinámica de Herramientas**

1.  **Define un Conjunto de Herramientas:**
    *   Incluye herramientas estándar de Langchain (ej. `DuckDuckGoSearchRun`).
    *   **Crea al menos una Herramienta Personalizada Compleja:**
        *   Ejemplo: Una herramienta que interactúa con una API específica (noticias, clima, una base de datos de productos, etc.).
        *   Ejemplo: Una herramienta que realiza un análisis de datos sobre un archivo CSV proporcionado por el usuario.
        *   Asegúrate de que tu herramienta personalizada tenga una descripción clara para que el LLM sepa cuándo usarla.
        ```python
        from langchain_core.tools import tool

        @tool
        def mi_herramienta_api_compleja(parametro1: str, parametro2: int) -> str:
            """Esta herramienta interactúa con la API X para obtener información sobre Y 
            basándose en parametro1 y parametro2. Es útil cuando necesitas datos específicos de X."""
            # Lógica para llamar a la API, procesar la respuesta
            # return "Resultado de la API"
            pass
        ```
2.  **Selección Dinámica de Herramientas:**
    *   El nodo "pensador" de tu agente debe ser capaz de decidir qué herramienta (si alguna) usar.
    *   Esto se puede lograr haciendo que el LLM genere una llamada a función (si usas un modelo que lo soporte como OpenAI) o un formato estructurado que indique la herramienta y sus argumentos.
    *   Langchain tiene mecanismos para esto (ej. `convert_to_openai_function` para herramientas, o agentes que parsean la salida del LLM para llamadas a herramientas).

**Paso 5: Manejo de Errores en Herramientas**

1.  **Captura de Excepciones:** En la ejecución de tus herramientas (especialmente las personalizadas), envuelve las partes críticas en bloques `try...except`.
2.  **Informar al Agente:** Si una herramienta falla, la observación devuelta al agente debe indicar el error de forma clara.
3.  **Lógica de Reintento/Alternativa:** El nodo "pensador" del agente debe ser capaz de:
    *   Reintentar la herramienta (quizás con parámetros diferentes).
    *   Probar una herramienta alternativa.
    *   Informar al usuario que no pudo completar la solicitud debido al fallo de la herramienta.
    *   Esto se modela con aristas condicionales en LangGraph basadas en el resultado de la ejecución de la herramienta.

**Paso 6: Orquestación con LangGraph**

*   Ensambla todos los componentes (estado, nodos, herramientas, memoria) en un `StateGraph`.
*   Presta especial atención a los ciclos: `(Usuario Input) -> Pensar -> (Herramienta o Respuesta) -> (Actualizar Memoria) -> Pensar/Responder`.
*   Utiliza `CompiledStateGraph.stream(inputs)` para depurar y observar el flujo de estado.

**Paso 7: Prueba y Refinamiento**

*   Prueba tu agente con una variedad de entradas y escenarios.
*   Verifica que la memoria persistente funcione correctamente (la información se guarda y se carga entre sesiones).
*   Observa si el agente selecciona las herramientas adecuadas y maneja los errores.
*   Refina los prompts de tu agente, las descripciones de las herramientas y la lógica del grafo según sea necesario.

**Paso 8: Reflexión sobre la Evaluación (Conceptual)**

No necesitas implementar un framework de evaluación complejo, pero reflexiona sobre cómo lo harías:

*   **Métricas:** ¿Qué define el éxito para tu agente? (ej. finalización de tareas, calidad de la respuesta, eficiencia, capacidad de usar la memoria correctamente).
*   **Conjunto de Datos de Prueba:** ¿Cómo crearías un conjunto de escenarios de prueba representativos?
*   **Evaluación Humana vs. Automática:** ¿Qué aspectos se pueden evaluar automáticamente (ej. si una herramienta fue llamada) y cuáles requieren juicio humano?
*   Herramientas como LangSmith o Ragas (para componentes RAG) podrían ser útiles aquí.

**Ejemplo de Estructura de Código (Muy Simplificado):**

```python
# (Importaciones: TypedDict, LangGraph, LLM, herramientas, etc.)

# class AgentState(TypedDict):
#     messages: Annotated[list, add_messages]
#     persistent_memory_content: str # o una estructura más compleja
#     # ... otros campos de estado

# def load_memory_node(state: AgentState):
#     # Cargar desde archivo/DB
#     # return {"persistent_memory_content": ...}

# def save_memory_node(state: AgentState):
#     # Guardar state["persistent_memory_content"] o partes relevantes del historial
#     # return {}

# def think_node(state: AgentState):
#     # LLM decide la acción, actualiza messages
#     # return {"messages": ...}

# def tool_node(state: AgentState):
#     # Ejecuta la herramienta decidida en think_node, maneja errores
#     # return {"messages": ... # con la observación de la herramienta}

# # ... otros nodos y funciones de condición

# workflow = StateGraph(AgentState)
# workflow.add_node("loader", load_memory_node)
# workflow.add_node("thinker", think_node)
# # ... añadir más nodos y aristas
# workflow.set_entry_point("loader")
# # ... definir el flujo, incluyendo el guardado de memoria

# app = workflow.compile()

# # Simular una sesión
# inputs = {"messages": [("user", "Hola, agente")]}
# for event in app.stream(inputs):
#     print(event)

# # Simular otra sesión (la memoria debería cargarse)
# inputs_new_session = {"messages": [("user", "¿Recuerdas de qué hablamos antes?")]}
# for event in app.stream(inputs_new_session):
#     print(event)
```

**Recursos y Documentación Adicional:**

*   Documentación de Langchain sobre Agentes y Memoria.
*   Documentación de LangGraph (especialmente ejemplos con ciclos y herramientas).
*   Artículos sobre arquitecturas de agentes como ReAct (Reason+Act).
*   Tutoriales sobre cómo integrar Langchain con bases de datos o almacenes de vectores.

**Criterios de Evaluación y Verificación:**

*   El agente se ejecuta y puede completar una tarea compleja que definiste, utilizando múltiples pasos y herramientas.
*   La memoria persistente funciona: el agente puede referenciar o utilizar información de interacciones/sesiones anteriores.
*   El agente demuestra la selección dinámica de al menos una herramienta personalizada y una herramienta estándar.
*   El agente maneja al menos un escenario de error de una herramienta de forma graceful (ej. informando al usuario o intentando una alternativa).
*   El flujo del agente está claramente definido y orquestado usando LangGraph.
*   Puedes discutir los desafíos encontrados y las decisiones de diseño que tomaste, especialmente en relación con la memoria y el manejo de herramientas.
*   Puedes articular un plan básico sobre cómo evaluarías el rendimiento de tu agente.

**Este es un reto ambicioso. Concéntrate en implementar un subconjunto funcional de estas características si el tiempo es limitado, pero asegúrate de comprender los conceptos detrás de todas ellas.**
