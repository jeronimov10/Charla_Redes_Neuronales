# AI Chatbot - Redes Neuronales

Un chatbot interactivo basado en redes neuronales desarrollado con PyTorch, diseñado como herramienta educativa para charlas sobre redes neuronales. Permite que los estudiantes hagan preguntas sobre conceptos de deep learning durante la presentación.

## Sobre este Proyecto

Este proyecto demuestra cómo funcionan las redes neuronales de forma práctica. El chatbot está entrenado para responder preguntas sobre:

- Conceptos fundamentales de redes neuronales
- Funciones de activación (ReLU, Sigmoid, Tanh)
- Procesos de entrenamiento y backpropagation
- Arquitecturas de redes (CNN, RNN, Transformers)
- Temas avanzados como overfitting, regularización y hiperparámetros
- Aplicaciones reales de IA

El chatbot utiliza un clasificador de red neuronal simple para entender las preguntas de los estudiantes y proporcionar respuestas educativas relevantes.

## Funcionalidades

- **Clasificación de intenciones**: Usa una red neuronal para clasificar preguntas en diferentes categorías
- **18+ intents**: Cubre las preguntas más comunes sobre redes neuronales
- **Procesamiento de Lenguaje Natural**: Tokenización y lemmatización de texto en español
- **Interfaz de línea de comandos**: Simple y fácil de usar durante la charla

## Requisitos

- Python 3.8+
- PyTorch
- NLTK
- NumPy

## Instalación

### 1. Clonar o descargar el proyecto

```bash
cd AI_Chatbot_Torch
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Descargar recursos de NLTK (Primera vez)

Si es la primera vez que usas el proyecto, descomenta estas líneas en `main.py`:

```python
nltk.download('punkt_tab')
nltk.download('wordnet')
```

Luego comenta las líneas de nuevo y guarda.

## Orden de Ejecución

### Opción A: Usar modelo ya entrenado (RECOMENDADO)

Si ya tienes el modelo entrenado (`chatbot_model.pth` y `dimensions.json`):

```bash
python main.py
```

Y el chatbot estará listo para usar inmediatamente.

### Opción B: Entrenar el modelo desde cero

Si necesitas entrenar un nuevo modelo:

1. Abre `main.py`
2. Descomenta el bloque de ENTRENAMIENTO:

```python
assistant = ChatbotAssistant('intents.json', function_mappings = {})
assistant.parse_intents()
assistant.prepare_data()
assistant.train_model(batch_size=8, lr=0.001, epochs=100)
assistant.save_model('chatbot_model.pth', 'dimensions.json')
```

3. Comenta el bloque de MODO INTERACTIVO
4. Ejecuta:

```bash
python main.py
```

5. Una vez finalizado el entrenamiento, descomenta el bloque de MODO INTERACTIVO y comenta el de entrenamiento
6. Ejecuta nuevamente:

```bash
python main.py
```

## Cómo Funciona

### Arquitectura de la Red Neuronal

```
Input (Bag of Words) 
    ↓
[Linear 128 neuronas] → ReLU → Dropout
    ↓
[Linear 64 neuronas] → ReLU → Dropout
    ↓
[Linear Output] → Clasificación de Intent
```

### Flujo de Procesamiento

1. **Tokenización y Lemmatización**: La pregunta se divide en palabras y se normalizan
2. **Bag of Words**: Se crea un vector binario basado en el vocabulario conocido
3. **Predicción**: La red neuronal clasifica la pregunta en una categoría (intent)
4. **Respuesta**: Se devuelve una respuesta aleatoria de las opciones disponibles

### Componentes Principales

- `ChatBotModel`: Arquitectura de la red neuronal
- `ChatbotAssistant`: Clase principal que maneja todo el ciclo de vida
  - `parse_intents()`: Carga y procesa el archivo de intents
  - `prepare_data()`: Prepara datos para el entrenamiento
  - `train_model()`: Entrena la red
  - `load_model()`: Carga un modelo preentrenado
  - `process_message()`: Procesa una pregunta del usuario

## Archivos del Proyecto

```
AI_Chatbot_Torch/
├── main.py                 # Código principal
├── intents.json           # Base de conocimiento (intents y respuestas)
├── requirements.txt       # Dependencias del proyecto
├── chatbot_model.pth      # Modelo entrenado (se crea después de entrenar)
├── dimensions.json        # Dimensiones del modelo (se crea después de entrenar)
└── README.md             # Este archivo
```

## Uso Durante la Charla

1. Abre una terminal en la carpeta del proyecto
2. Ejecuta:
   ```bash
   python main.py
   ```
3. El chatbot mostrará un mensaje de bienvenida
4. Los estudiantes pueden hacer preguntas como:
   - "Que son las redes neuronales?"
   - "Como funciona una neurona artificial?"
   - "Que es dropout?"
   - "Diferencia entre ReLU y sigmoid?"
   - "Como se entrena una red?"

5. Para salir, escribe `/salir` o presiona `Ctrl+C`

## Intents Disponibles

| Tag | Ejemplo de Pregunta | Tema |
|-----|-------------------|------|
| greeting | "Hola" | Saludos |
| thanks | "Gracias" | Agradecimientos |
| what_is_nn | "Que son las redes neuronales?" | Definición |
| how_nn_work | "Como funcionan las redes?" | Funcionamiento |
| neuron_structure | "Que es una neurona artificial?" | Estructura |
| activation_functions | "Que es ReLU?" | Funciones de activación |
| loss_functions | "Que es la funcion de perdida?" | Loss functions |
| learning_process | "Que es backpropagation?" | Aprendizaje |
| overfitting | "Como evitar overfitting?" | Regularización |
| deep_learning | "Que es deep learning?" | Arquitecturas |
| nlp | "Que es NLP?" | Procesamiento de Lenguaje |
| image_recognition | "Reconocimiento de imagenes" | Visión |
| datasets | "Como preparar datos?" | Preparación de datos |
| hyperparameters | "Que es learning rate?" | Hiperparámetros |
| pytorch_vs_tensorflow | "Diferencia entre PyTorch y TensorFlow?" | Frameworks |
| applications | "Donde se usa IA?" | Aplicaciones |

## Parámetros del Modelo

- **Capas ocultas**: 128 → 64 neuronas
- **Función de activación**: ReLU
- **Regularización**: Dropout (0.5)
- **Optimizador**: Adam
- **Learning Rate**: 0.001
- **Batch Size**: 8
- **Epochs**: 100

Estos parámetros pueden ajustarse en `main.py` para mejorar el rendimiento.

## Troubleshooting

### Error: "FileNotFoundError: chatbot_model.pth"
Necesitas entrenar el modelo primero. Ve a la sección "Opción B" en "Orden de Ejecución".

### Error de encoding en intents.json
El archivo ha sido limpiado de acentos para evitar problemas. Si necesitas agregar nuevos intents, evita characters especiales.

### El chatbot no reconoce mis preguntas
Intenta hacer preguntas más similares a los ejemplos en el archivo `intents.json`. El modelo funciona por similitud.

## Mejoras Futuras

- Agregar más intents y respuestas
- Implementar confidence threshold para respuestas
- Agregar logs de conversación
- Interfaz gráfica con Tkinter o Flask
- Integración con API de OpenAI para respuestas mejoradas

## Licencia

Este proyecto es de código abierto para fines educativos.

## Autor

Creado como herramienta educativa para charlas de redes neuronales.

---

**¡Disfruta de tu charla y que los estudiantes aprendan sobre redes neuronales de forma interactiva!**
