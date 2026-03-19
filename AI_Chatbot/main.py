import os
import json
import random

import nltk

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# nltk.download('punkt_tab') toca correr esto si es la primera vez que uno usa nltk
# nltk.download('wordnet') toca correr esto si es la primera vez usando nltk

class ChatBotModel(nn.Module):
    

    def __init__(self, input_size, output_size):
        super(ChatBotModel, self).__init__()

        self.fc1 = nn.Linear(input_size, 128) #Primera capa oculta / pasa inputs size a 128 neuronas
        self.fc2 = nn.Linear(128, 64) # Segunda capa oculta / pasa 128 neuronas a 64 neuronas
        self.fc3 = nn.Linear(64, output_size) # Tercera capa oculta / pasa 64 neuronas a output size
        self.relu = nn.ReLU() # Funcion de activacion (ReLu) la usamos para romper la linealidad
        self.dropout = nn.Dropout(0.5) # Dropout

    def forward(self, x): # Forward Propagation
        x = self.relu(self.fc1(x))
        x = self.dropout(x)

        x = self.relu(self.fc2(x))
        x = self.dropout(x)

        x = self.fc3(x)
        return x
    
class ChatbotAssistant():

    def __init__(self, intents_path, function_mappings = None):

        self.model = None
        self.intents_path = intents_path

        self.documents = []
        self.vocabulary = []

        self.intents = []
        self.intents_responses = {}

        self.function_mappings = function_mappings if function_mappings else {}

        self.X = None
        self.y = None

    @staticmethod
    def tokenize_and_lemmatize(text):

        lemmatizer = nltk.WordNetLemmatizer()

        words = nltk.word_tokenize(text)
        words = [lemmatizer.lemmatize(token.lower()) for token in words]

        return words
    
    
    def bag_of_words(self, words):
        return [1 if word in words else 0 for word in self.vocabulary] # Voy por todas las palabras que conozco y doy 1 si la conozco 0 si no
    def parse_intents(self):
        lemmatizer = nltk.WordNetLemmatizer()

        if os.path.exists(self.intents_path):
            with open(self.intents_path, 'r') as f:
                intents_data = json.load(f)

            for intent in intents_data['intents']:
                if intent['tag'] not in self.intents:
                    self.intents.append(intent['tag'])
                    self.intents_responses[intent['tag']] = intent['responses']

                for pattern in intent['patterns']:
                    pattern_words = self.tokenize_and_lemmatize(pattern)
                    self.vocabulary.extend(pattern_words)
                    self.documents.append((pattern_words, intent['tag']))

            self.vocabulary = sorted(set(self.vocabulary))

    def prepare_data(self):
        bags = []
        indices = []

        for document in self.documents: # Transforma las palabras en 1 o 0
            words = document[0]
            bag = self.bag_of_words(words)
            
            intent_index = self.intents.index(document[1]) # Saco el tag

            bags.append(bag)
            indices.append(intent_index)

        self.X = np.array(bags)
        self.y = np.array(indices)

    def train_model(self, batch_size, lr, epochs):
        X_tensor = torch.tensor(self.X, dtype=torch.float32) # Representacion del bag of words de todas las palabras
        y_tensor = torch.tensor(self.y, dtype=torch.long) # La correcta clasificacion

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = ChatBotModel(self.X.shape[1], len(self.intents)) # El input size depende de cuantas palabras vemos osea le tamaño del individual bag del bag of words y el output size es el tamaño de los intents

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            running_loss = 0.0


            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss

            print(f"Epoch {epoch+1}: Loss: {running_loss / len(loader)}")

    def save_model(self, model_path, dimensions_path):
        torch.save(self.model.state_dict(), model_path)
        
        with open(dimensions_path, 'w') as f:
            json.dump({'input_size': self.X.shape[1], 'output_size': len(self.intents)}, f)

    def load_model(self, model_path, dimensions_path):

        with open(dimensions_path, 'r') as f:
            dimensions = json.load(f)

        self.model = ChatBotModel(dimensions['input_size'], dimensions['output_size'])
        self.model.load_state_dict(torch.load(model_path, weights_only=True))

    def process_message(self, input_message):
        words = self.tokenize_and_lemmatize(input_message)
        bag = self.bag_of_words(words)

        bag_tensor = torch.tensor([bag], dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(bag_tensor)

        predicted_class_index = torch.argmax(predictions, dim=1).item()
        predicted_intent = self.intents[predicted_class_index]

        if self.function_mappings:
            if predicted_intent in self.function_mappings:
                self.function_mappings[predicted_intent]()

        if self.intents_responses[predicted_intent]:
            return random.choice(self.intents_responses[predicted_intent])
        else:
            return None



if __name__ == '__main__':
    # Ensure we always use paths relative to this script, regardless of the current working directory.
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    intents_file = os.path.join(BASE_DIR, 'intents.json')
    model_file = os.path.join(BASE_DIR, 'chatbot_model.pth')
    dimensions_file = os.path.join(BASE_DIR, 'dimensions.json')

    # ENTRENAMIENTO (descomenta si necesitas entrenar nuevamente)
    # assistant = ChatbotAssistant(intents_file, function_mappings={})
    # assistant.parse_intents()
    # assistant.prepare_data()
    # assistant.train_model(batch_size=8, lr=0.001, epochs=100)
    # assistant.save_model(model_file, dimensions_file)

    # CHATBOT
    print("\n" + "="*60)
    print("CHATBOT DE REDES NEURONALES")
    print("="*60)
    print("Hola estudiantes!")
    print("Hago preguntas sobre redes neuronales, deep learning e IA")
    print("Escribe: /salir para salir\n")
    print("="*60 + "\n")
    
    assistant = ChatbotAssistant(intents_file, function_mappings={})
    assistant.parse_intents()
    assistant.load_model(model_file, dimensions_file)

    while True:
        try:
            message = input("Tu pregunta: ").strip()

            if not message:
                continue
                
            if message.lower() == '/salir' or message.lower() == 'salir':
                print("\n¡Gracias por tus preguntas! Sigue aprendiendo\n")
                break

            response = assistant.process_message(message)
            if response:
                print(f"Respuesta: {response}\n")
            else:
                print("Respuesta: No tengo respuesta para eso\n")
                
        except KeyboardInterrupt:
            print("\n\n¡Gracias por tus preguntas! Sigue aprendiendo\n")
            break
        except Exception as e:
            print(f"Error: {e}\n")






