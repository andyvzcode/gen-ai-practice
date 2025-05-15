# Reto Intermedio 2: Creando tu Primera Tripulación (Crew) de Agentes con Crew AI

**Nivel de Dificultad:** Intermedio

**Herramientas Principales Involucradas:** Crew AI, Langchain, Python, Modelo de Lenguaje (ej. OpenAI)

**Conceptos Clave Abordados:** Agentes Colaborativos, Roles, Metas (Goals), Tareas (Tasks), Herramientas (Tools) para Agentes, Procesos Secuenciales y Jerárquicos, Delegación de Tareas, Orquestación de Múltiples Agentes.

**Objetivos Específicos del Reto:**

*   Comprender el concepto de sistemas multi-agente y cómo Crew AI facilita su creación.
*   Aprender a definir Agentes con roles específicos, metas claras y backstories (contexto).
*   Aprender a definir Tareas detalladas que los agentes deben realizar, incluyendo las herramientas que pueden usar y el resultado esperado.
*   Configurar una Tripulación (Crew) que consista en múltiples agentes y las tareas que deben ejecutar.
*   Entender los diferentes procesos de ejecución de tareas en Crew AI (por ejemplo, secuencial).
*   Asignar herramientas (Tools de Langchain) a los agentes para que puedan interactuar con el mundo exterior o realizar cálculos complejos.
*   Ejecutar la tripulación y observar cómo los agentes colaboran para lograr un objetivo común.
*   Analizar el resultado producido por la tripulación.

**Introducción Conceptual y Relevancia:**

Crew AI es un framework diseñado para orquestar agentes autónomos de IA que colaboran para resolver tareas complejas. En lugar de depender de un único agente monolítico, Crew AI permite definir múltiples agentes especializados, cada uno con su propio rol, conjunto de herramientas y objetivos. Estos agentes trabajan juntos en una "tripulación", delegando subtareas y compartiendo información para alcanzar una meta general. Este enfoque fomenta la modularidad, la especialización y puede llevar a soluciones más robustas y sofisticadas para problemas que requieren diversas habilidades o perspectivas. Por ejemplo, podrías tener un agente investigador, un agente escritor y un agente crítico trabajando juntos para producir un informe de alta calidad. Este reto te introducirá a los componentes fundamentales de Crew AI: Agentes, Tareas y la Tripulación misma, permitiéndote construir tu primer equipo de agentes colaborativos.

**Requisitos Previos:**

*   Conocimientos sólidos de Python.
*   Haber completado el "Reto Básico 1: Creando tu Primera Cadena (Chain) con Langchain" y comprender los conceptos de LLMs y Prompts.
*   Es útil, aunque no estrictamente necesario, haber revisado el "Reto Intermedio 1: Creando tu Primer Agente Cíclico con LangGraph" para tener una idea de la complejidad de los agentes individuales, aunque Crew AI abstrae gran parte de esa complejidad interna del agente.
*   Familiaridad con el concepto de `Tool` en Langchain.
*   Una cuenta de OpenAI y una clave de API configurada de forma segura.

**Instrucciones Detalladas Paso a Paso:**

Vamos a crear una tripulación simple con dos agentes: un "Investigador de Viajes" y un "Redactor de Itinerarios". El objetivo será planificar un viaje de fin de semana a una ciudad específica.

**Paso 1: Instalación de las Bibliotecas Necesarias**

Necesitarás `crewai` y sus dependencias, incluyendo `langchain-community` para algunas herramientas y `langchain-openai` si usas OpenAI.

```bash
pip install crewai langchain-community langchain-openai python-dotenv duckduckgo-search
```
*   `duckduckgo-search`: Lo usaremos para una herramienta de búsqueda simple que los agentes pueden utilizar.

**Paso 2: Configuración de la Clave de API y el LLM**

Configura tu clave de API de OpenAI en un archivo `.env` como en los retos anteriores. Crew AI utiliza LLMs (a través de Langchain) para potenciar a sus agentes.

```python
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
# Para este ejemplo, usaremos una herramienta de búsqueda simple de DuckDuckGo
# Crew AI se integra bien con herramientas de Langchain
from langchain_community.tools import DuckDuckGoSearchRun

# Cargar variables de entorno
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("No se encontró la clave de API de OpenAI. Asegúrate de que esté configurada en tu archivo .env")

# Configurar el LLM que usarán los agentes
# Puedes usar el mismo para todos o diferentes LLMs por agente si es necesario
llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.2, openai_api_key=api_key)
# Nota: Algunos modelos más pequeños o rápidos podrían tener dificultades con roles complejos o el uso de herramientas.
# GPT-4 o Claude 3 suelen funcionar bien.

print("LLM y dependencias listas.")
```

**Paso 3: Definición de Herramientas (Tools)**

Los agentes de Crew AI pueden usar herramientas de Langchain. Crearemos una instancia de la herramienta de búsqueda DuckDuckGo.

```python
# ... (código anterior)

search_tool = DuckDuckGoSearchRun()

print("Herramientas definidas.")
```

**Paso 4: Definición de los Agentes**

Un Agente en Crew AI se define por su `role`, `goal`, `backstory`, las `tools` que puede usar, y opcionalmente el `llm` específico que utilizará (si no, usará el LLM predeterminado de la tripulación o uno globalmente configurado).

```python
# ... (código anterior)

# Agente 1: Investigador de Viajes
researcher_agent = Agent(
    role="Investigador de Destinos de Viaje Senior",
    goal="Descubrir atracciones imperdibles y joyas ocultas en {city} para un viaje de fin de semana.",
    backstory=(
        "Eres un investigador de viajes de renombre con un ojo agudo para los detalles y una pasión por descubrir "
        "experiencias únicas. Tienes una habilidad especial para encontrar información que va más allá de las típicas "
        "guías turísticas. Tu objetivo es proporcionar una lista concisa pero inspiradora de lugares y actividades."
    ),
    tools=[search_tool],
    llm=llm,
    verbose=True, # Muestra los pensamientos y acciones del agente
    allow_delegation=False # Para este agente, no permitiremos que delegue tareas
)

# Agente 2: Redactor de Itinerarios
planner_agent = Agent(
    role="Redactor Experto de Itinerarios de Viaje",
    goal="Crear un itinerario de fin de semana atractivo, bien estructurado y factible para {city}, "
         "basado en la investigación proporcionada.",
    backstory=(
        "Eres un talentoso escritor de viajes que se especializa en transformar listas de atracciones en "
        "itinerarios cautivadores y prácticos. Sabes cómo equilibrar actividades, tiempos de viaje y "
        "descansos para crear una experiencia memorable. Tu itinerario debe ser fácil de seguir e inspirador."
    ),
    tools=[], # Este agente no necesita herramientas directas, se basará en la información del investigador
    llm=llm,
    verbose=True,
    allow_delegation=False
)

print("Agentes definidos.")
```

**Paso 5: Definición de las Tareas (Tasks)**

Una Tarea describe el trabajo que un agente específico debe realizar. Incluye una `description` (que puede usar variables como `{city}`), el `agent` asignado, y el `expected_output`.

```python
# ... (código anterior)

# Tarea 1: Investigación del Destino
research_task = Task(
    description=(
        "Investiga a fondo la ciudad de {city}. Identifica las 5 principales atracciones turísticas "
        "y al menos 3 joyas ocultas o experiencias locales únicas. Considera una variedad de intereses "
        "(historia, arte, gastronomía, naturaleza, etc.)."
    ),
    agent=researcher_agent,
    expected_output=(
        "Un informe conciso que liste las 5 atracciones principales y 3 joyas ocultas para {city}, "
        "con una breve descripción (1-2 frases) para cada una. El informe debe ser claro y fácil de entender."
    )
)

# Tarea 2: Creación del Itinerario
# Esta tarea dependerá del resultado de la tarea de investigación.
# Crew AI maneja el paso de contexto entre tareas secuenciales.
planning_task = Task(
    description=(
        "Usando la investigación proporcionada sobre {city} (atracciones principales y joyas ocultas), "
        "crea un itinerario detallado para un viaje de fin de semana (Sábado y Domingo). "
        "El itinerario debe ser atractivo, incluir horarios sugeridos, y ser realista en términos de logística. "
        "Asegúrate de que el tono sea inspirador."
    ),
    agent=planner_agent,
    expected_output=(
        "Un itinerario de fin de semana completo y bien formateado para {city}, presentado día por día (Sábado, Domingo). "
        "Debe incluir nombres de lugares, breves descripciones de actividades y sugerencias de horarios. "
        "El resultado final debe ser un único bloque de texto Markdown."
    ),
    # context=[research_task] # Opcionalmente, puedes definir explícitamente el contexto de tareas anteriores.
                              # CrewAI a menudo lo infiere bien en procesos secuenciales.
)

print("Tareas definidas.")
```

**Paso 6: Creación y Configuración de la Tripulación (Crew)**

La Tripulación (Crew) une a los agentes y las tareas. También se define el `process` (por ejemplo, `Process.sequential` donde las tareas se ejecutan una tras otra).

```python
# ... (código anterior)

# Crear la tripulación con los agentes y las tareas
trip_crew = Crew(
    agents=[researcher_agent, planner_agent],
    tasks=[research_task, planning_task],
    process=Process.sequential, # Las tareas se ejecutarán en el orden en que se listan
    verbose=2 # Nivel de verbosidad: 0 (silencioso), 1 (básico), 2 (detallado)
)

print("Tripulación creada.")
```

**Paso 7: Ejecutar la Tripulación (Kickoff)**

Ahora, podemos poner en marcha la tripulación. Necesitaremos proporcionar las variables de entrada que nuestras tareas y agentes esperan (en este caso, `{city}`).

```python
# ... (código anterior)

# Ciudad para la que queremos planificar el viaje
city_to_plan = "París"

print(f"\n--- Iniciando la Tripulación para planificar un viaje a: {city_to_plan} ---")

# Ejecutar la tripulación con las entradas necesarias
try:
    trip_result = trip_crew.kickoff(inputs={'city': city_to_plan})

    print("\n\n--- Resultado Final de la Tripulación ---")
    print(trip_result)

except Exception as e:
    print(f"Error durante la ejecución de la tripulación: {e}")

# Otro ejemplo
city_to_plan_2 = "Kioto"
print(f"\n--- Iniciando la Tripulación para planificar un viaje a: {city_to_plan_2} ---")
try:
    trip_result_2 = trip_crew.kickoff(inputs={'city': city_to_plan_2})
    print("\n\n--- Resultado Final de la Tripulación (Kioto) ---")
    print(trip_result_2)
except Exception as e:
    print(f"Error durante la ejecución de la tripulación para Kioto: {e}")

```

**Recursos y Documentación Adicional:**

*   Documentación Oficial de Crew AI: [https://docs.crewai.com/](https://docs.crewai.com/)
*   Crew AI - Guía de Inicio Rápido: [https://docs.crewai.com/getting-started/creating-a-crew/](https://docs.crewai.com/getting-started/creating-a-crew/)
*   Crew AI - Definición de Agentes: [https://docs.crewai.com/core-concepts/Agents/](https://docs.crewai.com/core-concepts/Agents/)
*   Crew AI - Definición de Tareas: [https://docs.crewai.com/core-concepts/Tasks/](https://docs.crewai.com/core-concepts/Tasks/)
*   Crew AI - Herramientas (Tools): [https://docs.crewai.com/core-concepts/Tools/](https://docs.crewai.com/core-concepts/Tools/)

**Criterios de Evaluación y Verificación:**

*   Tu script se ejecuta sin errores.
*   El agente `researcher_agent` utiliza la herramienta `search_tool` (verás su actividad si `verbose=True` para el agente y `verbose=2` para la crew).
*   El agente `planner_agent` recibe la información del investigador y la utiliza para crear el itinerario.
*   El resultado final (`trip_result`) es un itinerario de viaje coherente y bien formateado para la ciudad especificada, basado en la investigación simulada.
*   Puedes seguir el flujo de trabajo de los agentes y las tareas a través de la salida detallada (verbose output).

**Posibles Extensiones o Retos Adicionales:**

*   Añade un tercer agente, por ejemplo, un "Crítico de Itinerarios" que revise el plan del `planner_agent` y sugiera mejoras. Esto podría requerir un proceso más complejo o la delegación de tareas.
*   Experimenta con diferentes LLMs para tus agentes. ¿Cómo afecta la calidad del resultado?
*   Proporciona a los agentes herramientas más sofisticadas (por ejemplo, herramientas para buscar vuelos, hoteles, o herramientas personalizadas que hayas creado con Langchain).
*   Intenta cambiar el `process` de la tripulación a uno jerárquico (si es aplicable a tu caso de uso) o explora cómo manejar tareas que pueden ejecutarse en paralelo.
*   Implementa un mecanismo de memoria compartida más explícito para la tripulación si necesitas que los agentes accedan a un conjunto de conocimientos común más allá del paso de resultados de tareas.
