# 🌐 Transformer: Traductor de Inglés a Español desde Cero

Este repositorio contiene la implementación completa, desde cero, de un modelo Transformer basado en la arquitectura Encoder-Decoder descrita en el artículo original
[*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017). 
El objetivo principal de este proyecto es entrenar un modelo de Traducción Automática Neuronal (NMT) capaz de traducir texto de inglés a español.

---

## 📖 ¿Qué es este proyecto?

Este proyecto no utiliza librerías de alto nivel sino que el Transformer ha sido programado componente por componente (Self-Attention, Multi-Head Attention, Positional Encoding, etc.) 
mediante Pytorch para comprender en profundidad el comportamiento y el flujo de de los datos  que hacen posible el correcto funcionamiento del Transformer.

### ¿Qué es un Transformer?
Antes de 2017, la traducción automática estaba dominada por Redes Neuronales Recurrentes (RNNs) y LSTMs, las cuales procesaban el texto palabra por palabra, siendo lentas y perdiendo el contexto en frases largas. 

El **Transformer** revolucionó la Inteligencia Artificial al eliminar la recurrencia y utilizar únicamente **Mecanismos de Atención** (*Self-Attention* y *Cross-Attention*). Esto permite al modelo:
1. Procesar todas las palabras de una frase simultáneamente (paralelización).
2. Entender qué palabras de una oración están relacionadas entre sí, independientemente de la distancia que las separe.

---

## 🏗️ Arquitectura y Configuración

El proyecto replica la arquitectura clásica Encoder-Decoder. El **Encoder** procesa la frase en inglés y extrae su significado profundo, mientras que el **Decoder** toma esa información y 
genera la traducción al español, prestando atención a las partes relevantes del texto original paso a paso.

![Arquitectura del Transformer original](ruta/a/tu/imagen/transformer_architecture.png)
*> Imagen de la arquitectura original extraída del paper "Attention Is All You Need".*

### Parámetros del Modelo
Para este entrenamiento, el modelo ha sido instanciado con la siguiente configuración técnica:

* **`CONTEXT_LENGTH = 64`**: Longitud máxima de las secuencias de entrada y salida (en tokens).
* **`D_EMBEDDING = 512`**: Dimensión de los vectores de embedding y de las capas ocultas del modelo.
* **`ATTENTION_HEADS = 8`**: Número de "cabezas" en el mecanismo de Multi-Head Attention, permitiendo al modelo enfocarse en 8 aspectos gramaticales/semánticos diferentes a la vez.
* **`NUMBER_ENCODERS = 6`**: Cantidad de capas apiladas en el bloque del Encoder.
* **`NUMBER_DECODERS = 6`**: Cantidad de capas apiladas en el bloque del Decoder.

---

## 📚 Datasets de Entrenamiento

Para lograr que el modelo aprenda a traducir, el entrenamiento se ha realizado utilizando una combinación de dos corpus paralelos:

1. **Tatoeba:** Un gran conjunto de oraciones cotidianas y traducciones colaborativas. Aporta al modelo la capacidad de entender lenguaje natural, coloquial y frases cortas del día a día.
2. **Parlamento Europeo (Europarl):** Transcripciones oficiales de las sesiones del Parlamento Europeo. Proporciona al modelo una gramática estructurada, vocabulario rico, formal y estructuras de oraciones más complejas.

---

## 🔡 Tokenización y Vocabulario Custom (BPE)

En lugar de depender de tokenizadores preentrenados genéricos (como `cl100k_base` de OpenAI o el de GPT-2), este proyecto implementa **su propio tokenizador entrenado desde cero**
sobre nuestro corpus bilingüe. Esto mejora significativamente el aprendizaje del algoritmo al estar adaptado específicamente al inglés y al español.

**Características del Tokenizador:**
* **Algoritmo:** Byte-Pair Encoding (BPE) a nivel de bytes (`ByteLevel`).
* **Tamaño del vocabulario:** 32.000 tokens (vocabulario compartido para ambos idiomas).
* **Tokens especiales:** * `<PAD>`: Para rellenar secuencias cortas.
  * `<START>`: Indica el inicio de la traducción.
  * `<END>`: Indica el final de la secuencia generada.
  * `<UNK>`: Para palabras fuera del vocabulario.
  
Contar con un vocabulario específico para el par inglés-español, en lugar de depender de tokenizadores universales o a nivel de carácter, optimiza el espacio de embeddings.
Esto no solo facilita enormemente el aprendizaje del Transformer, sino que aporta una generalización mucho más robusta ante textos no vistos

---

## 📊 Comportamiento del Modelo y Resultados

A continuación se muestra cómo se comporta el modelo tras el entrenamiento, ilustrando su capacidad de aprendizaje y traducción.

### 1. Evolución del Entrenamiento (Loss)
La siguiente gráfica muestra cómo el modelo fue reduciendo su error (Loss) época tras época, aprendiendo progresivamente el mapeo entre el vocabulario en inglés y su correspondencia en español.

![Gráfica de Loss del entrenamiento](ruta/a/tu/imagen/loss_curve.png)
*> Curva de aprendizaje mostrando la convergencia del modelo.*

### 2. Mapas de Atención (Cross-Attention)
La verdadera "magia" del Transformer es visible aquí. Este mapa de calor muestra en qué palabras de la frase original en inglés (columnas) se fijó el Decoder al generar cada palabra en español (filas). Se puede observar cómo el modelo aprende a invertir el orden de los adjetivos y sustantivos (por ejemplo, *blue car* -> *coche azul*).

![Mapa de calor de atención](ruta/a/tu/imagen/attention_map.png)
*> Visualización de los pesos de atención durante la traducción de una frase.*

### 3. Ejemplos de Inferencia
Resultados del modelo traduciendo frases que nunca antes había visto en los datasets de entrenamiento:

![Ejemplos de output en consola](ruta/a/tu/imagen/inference_examples.png)
*> Captura del script de inferencia realizando traducciones en vivo.*

---
