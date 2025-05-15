# Reto Intermedio (Embeddings): Profundizando en Embeddings de Texto

**Nivel de Dificultad:** Intermedio

**Herramientas Principales Involucradas:** Python, NumPy, Scikit-learn, Matplotlib (opcional para visualización), Modelos de Embedding (ej. Sentence Transformers, OpenAI)

**Conceptos Clave Abordados:** Embeddings de Texto, Espacio Vectorial Semántico, Modelos de Embedding (Transformers, Word2Vec conceptualmente), Generación de Embeddings, Similitud del Coseno, Distancia Euclidiana, Reducción de Dimensionalidad (PCA, t-SNE para visualización), Casos de Uso (Búsqueda Semántica, Clustering, Clasificación).

**Objetivos Específicos del Reto:**

*   Comprender en profundidad qué son los embeddings de texto y por qué son fundamentales en el NLP moderno.
*   Aprender a generar embeddings de texto utilizando bibliotecas populares como Sentence Transformers o la API de OpenAI.
*   Calcular la similitud semántica entre textos utilizando sus embeddings (por ejemplo, similitud del coseno).
*   Aplicar embeddings para una tarea simple de búsqueda semántica (encontrar la frase más similar en un pequeño corpus).
*   Visualizar embeddings de alta dimensión en 2D o 3D utilizando técnicas de reducción de dimensionalidad como PCA o t-SNE (opcional, pero muy ilustrativo).
*   Discutir cómo los embeddings pueden ser utilizados como características para tareas de machine learning downstream (clustering, clasificación).

**Introducción Conceptual y Relevancia:**

Los embeddings de texto son el pilar de muchas aplicaciones modernas de Procesamiento de Lenguaje Natural (NLP) e Inteligencia Artificial. Son representaciones vectoriales densas (listas de números de punto flotante) de palabras, frases, oraciones o documentos completos en un espacio de alta dimensión. La característica fundamental de estos embeddings es que capturan el **significado semántico** del texto: textos con significados similares tendrán embeddings que están "cerca" uno del otro en este espacio vectorial, mientras que textos con significados diferentes estarán más alejados.

Esta capacidad de representar el significado como geometría en un espacio vectorial abre la puerta a una amplia gama de aplicaciones, desde la búsqueda semántica (encontrar documentos relevantes para una consulta aunque no compartan palabras clave exactas), la agrupación (clustering) de documentos por tema, la clasificación de texto, hasta la generación de texto condicionado y los sistemas de recomendación.

Modelos como Word2Vec, GloVe, y más recientemente, los basados en Transformers (BERT, RoBERTa, Sentence Transformers, GPT, etc.) han revolucionado la forma en que generamos y utilizamos estos embeddings. Este reto se centrará en utilizar modelos preentrenados para generar embeddings y explorar algunas de sus propiedades y aplicaciones básicas.

**Requisitos Previos:**

*   Conocimientos sólidos de Python y NumPy.
*   Comprensión básica de conceptos de machine learning (vectores, características).
*   Familiaridad con la instalación y uso de bibliotecas de Python (`pip install`).
*   (Opcional para visualización) Conocimientos básicos de Matplotlib y Scikit-learn.

**Instrucciones Detalladas Paso a Paso:**

**Paso 1: Instalación de las Bibliotecas Necesarias**

Necesitarás `numpy`, `sentence-transformers` (una biblioteca popular para generar embeddings de alta calidad), y opcionalmente `scikit-learn` y `matplotlib` para la reducción de dimensionalidad y visualización.

```bash
pip install numpy sentence-transformers scikit-learn matplotlib
# Si planeas usar embeddings de OpenAI, también necesitarás:
# pip install openai
```

**Paso 2: Configuración Inicial y Generación de Embeddings**

Crea un script Python (por ejemplo, `embeddings_deep_dive.py`).

```python
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

print("Bibliotecas importadas.")

# 1. Cargar un modelo de embedding preentrenado
# Sentence Transformers ofrece muchos modelos. 'all-MiniLM-L6-v2' es un buen compromiso entre calidad y velocidad.
print("Cargando modelo de embedding (all-MiniLM-L6-v2)...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Modelo cargado.")

# 2. Frases de ejemplo para generar embeddings
sentences = [
    "El rápido zorro marrón salta sobre el perro perezoso.",
    "La inteligencia artificial está cambiando nuestro mundo.",
    "Amo la programación y el desarrollo de software.",
    "Los modelos de lenguaje pueden generar texto coherente.",
    "El clima de hoy es soleado y cálido.",
    "Un canino ágil brinca sobre el can pasivo.", # Semánticamente similar a la primera
    "La IA tiene un gran impacto en la sociedad.", # Semánticamente similar a la segunda
    "Disfruto codificando y creando aplicaciones.", # Semánticamente similar a la tercera
]

# 3. Generar los embeddings
print("\nGenerando embeddings para las frases...")
embeddings = model.encode(sentences)
# Los embeddings son arrays de NumPy
print(f"Forma de la matriz de embeddings: {embeddings.shape}") # (num_sentences, embedding_dimension)
print(f"Dimensión de cada embedding: {embeddings.shape[1]}")
# print("Primer embedding:", embeddings[0])
```

**Paso 3: Cálculo de Similitud Semántica**

Una vez que tenemos los embeddings, podemos calcular cuán similares son dos frases midiendo la "proximidad" de sus vectores. La similitud del coseno es una métrica común para esto.

*   **Similitud del Coseno:** Mide el coseno del ángulo entre dos vectores. Un valor de 1 significa que los vectores apuntan exactamente en la misma dirección (máxima similitud), 0 significa que son ortogonales (sin similitud), y -1 significa que apuntan en direcciones opuestas (máxima disimilitud).
*   **Distancia Euclidiana:** La distancia "recta" entre dos puntos (vectores). Menor distancia implica mayor similitud.

```python
# ... (código anterior)

# 4. Calcular la similitud semántica
print("\n--- Cálculo de Similitud Semántica ---")

# Comparar la primera frase con todas las demás, incluyendo ella misma
emb1 = embeddings[0].reshape(1, -1) # El embedding de la primera frase

# Similitud del Coseno
cosine_sim_matrix = cosine_similarity(emb1, embeddings)
print("\nSimilitud del Coseno entre la frase 1 y todas las demás:")
for i, sentence in enumerate(sentences):
    print(f"  Con 
'{sentence}": {cosine_sim_matrix[0, i]:.4f}")

# Distancia Euclidiana
# Nota: Menor distancia = mayor similitud
euclidean_dist_matrix = euclidean_distances(emb1, embeddings)
print("\nDistancia Euclidiana entre la frase 1 y todas las demás:")
for i, sentence in enumerate(sentences):
    print(f"  Con 
'{sentence}": {euclidean_dist_matrix[0, i]:.4f}")

# Encontrar la frase más similar a la primera (excluyéndola a ella misma)
print(f"\nBuscando la frase más similar a: 
'{sentences[0]}"")
similarities_to_first = cosine_similarity(embeddings[0].reshape(1, -1), embeddings[1:])
most_similar_index_in_rest = np.argmax(similarities_to_first)
# El índice devuelto es relativo a embeddings[1:], así que sumamos 1 para el índice original
original_index_of_most_similar = most_similar_index_in_rest + 1

print(f"La frase más similar es: 
'{sentences[original_index_of_most_similar]}" 
(Similitud Coseno: {similarities_to_first[0, most_similar_index_in_rest]:.4f})")
```

Deberías observar que `sentences[0]` ("El rápido zorro marrón salta sobre el perro perezoso.") es más similar a `sentences[5]` ("Un canino ágil brinca sobre el can pasivo.").

**Paso 4: Búsqueda Semántica Simple**

Podemos usar este principio para implementar una búsqueda semántica simple: dada una consulta, encontrar la frase más similar en nuestro corpus.

```python
# ... (código anterior)

# 5. Búsqueda Semántica Simple
print("\n--- Búsqueda Semántica Simple ---")
corpus_sentences = [
    "El cambio climático es un desafío global urgente.",
    "La energía solar es una fuente de energía renovable y limpia.",
    "Los bosques tropicales son vitales para la biodiversidad del planeta.",
    "La deforestación contribuye a la pérdida de hábitats.",
    "Reciclar ayuda a reducir la contaminación ambiental."
]

print("Corpus de búsqueda:")
for s in corpus_sentences: print(f"  - {s}")

corpus_embeddings = model.encode(corpus_sentences)

query = "¿Cómo podemos proteger el medio ambiente?"
query_embedding = model.encode([query])

# Calcular similitudes del coseno entre la consulta y el corpus
similarities = cosine_similarity(query_embedding, corpus_embeddings)

# Encontrar el índice de la frase más similar
most_similar_idx = np.argmax(similarities)

print(f"\nConsulta: 
'{query}"")
print(f"Frase más similar encontrada en el corpus: 
'{corpus_sentences[most_similar_idx]}"")
print(f"Similitud del Coseno: {similarities[0, most_similar_idx]:.4f}")
```

**Paso 5: Visualización de Embeddings con Reducción de Dimensionalidad (Opcional)**

Los embeddings suelen tener cientos de dimensiones, lo que los hace imposibles de visualizar directamente. Podemos usar técnicas de reducción de dimensionalidad como PCA (Análisis de Componentes Principales) o t-SNE (t-distributed Stochastic Neighbor Embedding) para proyectarlos en 2D o 3D.

```python
# ... (código anterior)

# 6. Visualización de Embeddings (Opcional)
print("\n--- Visualización de Embeddings (Opcional) ---")

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    # Usaremos los embeddings generados en el Paso 2
    # Reducción a 2D usando PCA
    print("Aplicando PCA para reducir a 2 dimensiones...")
    pca = PCA(n_components=2)
    embeddings_2d_pca = pca.fit_transform(embeddings)

    # Reducción a 2D usando t-SNE (puede ser más lento pero a menudo mejor para visualización)
    print("Aplicando t-SNE para reducir a 2 dimensiones... (puede tardar un poco)")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(embeddings)-1)) # Ajustar perplexity
    embeddings_2d_tsne = tsne.fit_transform(embeddings)

    # Graficar
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(embeddings_2d_pca[:, 0], embeddings_2d_pca[:, 1])
    for i, txt in enumerate(sentences):
        plt.annotate(f"  {i}: {txt[:20]}...", (embeddings_2d_pca[i, 0], embeddings_2d_pca[i, 1]))
    plt.title("Embeddings en 2D usando PCA")
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")

    plt.subplot(1, 2, 2)
    plt.scatter(embeddings_2d_tsne[:, 0], embeddings_2d_tsne[:, 1])
    for i, txt in enumerate(sentences):
        plt.annotate(f"  {i}: {txt[:20]}...", (embeddings_2d_tsne[i, 0], embeddings_2d_tsne[i, 1]))
    plt.title("Embeddings en 2D usando t-SNE")
    plt.xlabel("Componente t-SNE 1")
    plt.ylabel("Componente t-SNE 2")

    plt.tight_layout()
    # Para guardar la figura:
    # plt.savefig("embeddings_visualization.png")
    # print("Visualización guardada como embeddings_visualization.png")
    # Para mostrar la figura en entornos interactivos (como Jupyter):
    plt.show() 
    # En un script normal, plt.show() bloqueará la ejecución hasta que se cierre la ventana.
    # Puedes comentarlo si ejecutas el script de forma no interactiva y solo guardas la imagen.

except ImportError:
    print("Skipping visualization: scikit-learn o matplotlib no están instalados.")
except Exception as e:
    print(f"Error durante la visualización: {e}")

```

Observa cómo las frases semánticamente similares tienden a agruparse en el espacio 2D.

**Paso 6: Discusión: Embeddings como Características para Machine Learning**

Los embeddings de texto no solo son útiles para la búsqueda de similitud. También son excelentes **características** para alimentar modelos de machine learning tradicionales para tareas como:

*   **Clasificación de Texto:** Entrenar un clasificador (ej. SVM, Regresión Logística, Red Neuronal) para predecir la categoría de un texto (ej. análisis de sentimiento, detección de spam, clasificación de temas) utilizando sus embeddings como entrada.
*   **Clustering de Documentos:** Agrupar documentos similares utilizando algoritmos de clustering (ej. K-Means, DBSCAN) sobre sus embeddings. Esto puede ayudar a descubrir temas latentes en un corpus.
*   **Sistemas de Recomendación:** Recomendar artículos, productos o contenido a los usuarios basándose en la similitud de los embeddings de los ítems que han consumido o les han gustado.
*   **Detección de Anomalías:** Identificar textos que son semánticamente diferentes del resto de un conjunto de datos.

**Recursos y Documentación Adicional:**

*   Sentence Transformers Documentation: [https://www.sbert.net/](https://www.sbert.net/)
*   OpenAI Embeddings API: [https://platform.openai.com/docs/guides/embeddings](https://platform.openai.com/docs/guides/embeddings)
*   Blog de Jay Alammar - The Illustrated Word2vec: [http://jalammar.github.io/illustrated-word2vec/](http://jalammar.github.io/illustrated-word2vec/)
*   Blog de Jay Alammar - The Illustrated Transformer: [http://jalammar.github.io/illustrated-transformer/](http://jalammar.github.io/illustrated-transformer/)
*   Scikit-learn - Manifold learning (t-SNE, etc.): [https://scikit-learn.org/stable/modules/manifold.html](https://scikit-learn.org/stable/modules/manifold.html)
*   Scikit-learn - PCA: [https://scikit-learn.org/stable/modules/decomposition.html#pca](https://scikit-learn.org/stable/modules/decomposition.html#pca)

**Criterios de Evaluación y Verificación:**

*   Tu script se ejecuta sin errores.
*   Los embeddings se generan para las frases de ejemplo.
*   Puedes calcular e interpretar correctamente la similitud del coseno y la distancia euclidiana entre embeddings.
*   La búsqueda semántica simple identifica correctamente la frase más similar a una consulta dada dentro de un pequeño corpus.
*   (Opcional) Si implementas la visualización, puedes generar y observar los gráficos 2D de los embeddings, notando cómo se agrupan los textos semánticamente similares.
*   Puedes articular al menos dos casos de uso donde los embeddings de texto se utilizan como características para tareas de machine learning.

**Posibles Extensiones o Retos Adicionales:**

*   Experimenta con diferentes modelos de embedding de Sentence Transformers (o de OpenAI si tienes una clave de API) y observa cómo cambian los resultados de similitud o las visualizaciones.
*   Implementa una tarea de clasificación de texto simple (por ejemplo, usando un conjunto de datos como el de clasificación de sentimientos de IMDB) donde primero generas embeddings para los textos y luego entrenas un clasificador simple de Scikit-learn.
*   Intenta agrupar (clustering) los embeddings de un conjunto de frases usando K-Means de Scikit-learn y analiza los grupos resultantes.
*   Investiga cómo se manejan los embeddings para textos que exceden la longitud máxima de entrada de un modelo de embedding (por ejemplo, estrategias de chunking y promediado de embeddings de chunks).

