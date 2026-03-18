import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

X = np.array([
    [30, 40, 20],
    [28, 45, 25],
    [35, 30, 10],
    [22, 80, 90],
    [24, 75, 85],
    [20, 90, 95],
    [27, 60, 50],
    [26, 65, 70]
], dtype=float)

y = np.array([
    [0],
    [0],
    [0],
    [1],
    [1],
    [1],
    [0],
    [1]
], dtype=float)

X[:, 0] = X[:, 0] / 40
X[:, 1] = X[:, 1] / 100
X[:, 2] = X[:, 2] / 100

np.random.seed(42)

weights = np.random.uniform(-1, 1, (3, 1))
bias = np.random.uniform(-1, 1, (1,))
learning_rate = 0.5
epochs = 10000

for epoch in range(epochs):
    z = np.dot(X, weights) + bias
    output = sigmoid(z)

    error = y - output
    d_output = error * sigmoid_derivative(output)

    weights += learning_rate * np.dot(X.T, d_output)
    bias += learning_rate * np.sum(d_output)

    if epoch % 1000 == 0:
        loss = np.mean(error ** 2)
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

print("\nPesos finales:")
print(weights)
print("Bias final:")
print(bias)

print("\nPredicciones en los datos de entrenamiento:")
for i in range(len(X)):
    prediction = 1 if output[i][0] >= 0.5 else 0
    print(f"Entrada {i+1}: predicción={prediction}, valor real={int(y[i][0])}, prob={output[i][0]:.4f}")

print("\n" + "="*60)
print("PREDICCTOR DE LLUVIA - Sistema de Red Neuronal")
print("="*60)

print("\nRangos de datos esperados:")
print("- Temperatura: 0-40°C")
print("- Humedad: 0-100%")
print("- Nubosidad: 0-100%")
print("- Lluvia: 0 (no llovió) o 1 (llovió)")

print("\n" + "-"*60)
print("Ingrese los datos de los últimos 3 días")
print("-"*60)

datos_dias = []

for dia in range(1, 4):
    print(f"\nDÍA {dia} (hace {4-dia} días):")
    
    while True:
        try:
            temp = float(input(f"  Temperatura (0-40°C): "))
            if not (0 <= temp <= 40):
                print("La temperatura debe estar entre 0 y 40°C")
                continue
            
            humedad = float(input(f"  Humedad (0-100%): "))
            if not (0 <= humedad <= 100):
                print("La humedad debe estar entre 0 y 100%")
                continue
            
            nubosidad = float(input(f"  Nubosidad (0-100%): "))
            if not (0 <= nubosidad <= 100):
                print("La nubosidad debe estar entre 0 y 100%")
                continue
            
            lluvia = float(input(f"  ¿Llovió? (0=No, 1=Sí): "))
            if lluvia not in [0, 1]:
                print("Debes ingresar 0 o 1")
                continue
            
            datos_dias.append([temp, humedad, nubosidad, int(lluvia)])
            print(f"Datos del día {dia} guardados")
            break
        
        except ValueError:
            print("Por favor ingresa valores numéricos válidos")


print("\n" + "-"*60)
print("Procesando datos...")
print("-"*60)

nueva_entrada = np.array(datos_dias, dtype=float)
print("\nDatos ingresados (sin normalizar):")
for i, dia in enumerate(datos_dias):
    print(f"  Día {i+1}: Temp={dia[0]}C, Humedad={dia[1]}%, Nubosidad={dia[2]}%, Lluvia={dia[3]}")

nueva_entrada_norm = nueva_entrada.copy()
nueva_entrada_norm[:, 0] = nueva_entrada_norm[:, 0] / 40
nueva_entrada_norm[:, 1] = nueva_entrada_norm[:, 1] / 100
nueva_entrada_norm[:, 2] = nueva_entrada_norm[:, 2] / 100

entrada_modelo = nueva_entrada_norm[:, :3]

z_new = np.dot(entrada_modelo, weights) + bias
pred_new = sigmoid(z_new)

print("\n" + "="*60)
print("PREDICCIÓN DE LLUVIA PARA MAÑANA")
print("="*60)
print(f"\nProbabilidad de lluvia: {pred_new[0][0]*100:.2f}%")
print()

if pred_new[0][0] >= 0.5:
    print("PREDICCIÓN FINAL: SÍ VA A LLOVER")
else:
    print("PREDICCIÓN FINAL: NO VA A LLOVER")

print("\n" + "="*60)