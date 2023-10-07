import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar el dataset
casas = pd.read_csv('casas.csv')  # Reemplaza 'nombre_del_archivo.csv' con el nombre de tu archivo CSV

# Crear una instancia de LabelEncoder
label_encoder = LabelEncoder()

# Aplicar el etiquetado a la columna 'Departamento'
casas['Departamento'] = label_encoder.fit_transform(casas['Departamento'])

# Verificar el resultado
print(casas.head())
#0 es IT y 1 es RH

# Crear gráficos de dispersión para ver la relación entre las variables numéricas y 'Valor_casa'
variables_numericas = ['Salario', 'Hijos', 'Departamento']

# Definir una paleta de colores personalizada con colores distintivos
colores = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Puedes agregar más colores si es necesario

# Crear un conjunto de subplots para los gráficos
fig, axes = plt.subplots(2, len(variables_numericas), figsize=(15, 8))

# Crear gráficos de histogramas y dispersión con colores diferentes para cada variable
for i, variable in enumerate(variables_numericas):
    sns.histplot(data=casas, x=variable, kde=True, ax=axes[0, i], color=colores[i])
    axes[0, i].set_title(f'Histograma de {variable}')
    axes[0, i].set_xlabel(variable)
    axes[0, i].set_ylabel('Frecuencia')
    axes[0, i].grid(True)
    
    sns.scatterplot(x=variable, y='Valor_casa', data=casas, alpha=0.5, ax=axes[1, i], color=colores[i])
    axes[1, i].set_title(f'Gráfico de dispersión entre {variable} y Valor_casa')
    axes[1, i].grid(True)

# Ajustar el espacio entre subplots
plt.tight_layout()
plt.show()
    
# Calcular la matriz de correlación
correlation_matrix = casas[variables_numericas + ['Valor_casa']].corr()

# Crear un mapa de calor de correlación
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlación entre Variables Independientes y Valor_casa')
plt.show()

# Dividir el dataset en conjuntos de entrenamiento y prueba
X = casas[variables_numericas]  # Variables independientes
y = casas['Valor_casa']  # Variable objetivo

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verificar los tamaños de los conjuntos
print(f"Tamaño del conjunto de entrenamiento: {X_train.shape[0]} ejemplos")
print(f"Tamaño del conjunto de prueba: {X_test.shape[0]} ejemplos")

# Crear una instancia del modelo de regresión lineal
modelo_regresion = LinearRegression()

# Entrenar el modelo con los datos de entrenamiento
modelo_regresion.fit(X_train, y_train)

# Realizar predicciones en el conjunto de entrenamiento y prueba
y_train_pred = modelo_regresion.predict(X_train)
y_test_pred = modelo_regresion.predict(X_test)

# Calcular el error cuadrático medio (MSE) para ambos conjuntos
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

# Calcular el coeficiente de determinación (R^2) para ambos conjuntos
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

# Imprimir las métricas de evaluación para ambos conjuntos
print("Métricas para el conjunto de entrenamiento:")
print(f"Error Cuadrático Medio (MSE): {mse_train}")
print(f"Coeficiente de Determinación (R^2): {r2_train}")

print("\nMétricas para el conjunto de prueba:")
print(f"Error Cuadrático Medio (MSE): {mse_test}")
print(f"Coeficiente de Determinación (R^2): {r2_test}")

# Datos de los nuevos casos
nuevo_caso1 = [[17000, 3, 1]]  # Primer caso con departamento codificado como 1 (RH)
nuevo_caso2 = [[15400, 1, 0]]  # Segundo caso con departamento codificado como 0 (IT)

# Realizar predicciones para los nuevos casos
prediccion_caso1 = modelo_regresion.predict(nuevo_caso1)
prediccion_caso2 = modelo_regresion.predict(nuevo_caso2)

# Imprimir las predicciones
print("\nPredicciones para los Nuevos Casos:")
print("Predicción para el Primer Caso:")
print(f"Ingreso: ${nuevo_caso1[0][0]:,.2f}, Hijos: {nuevo_caso1[0][1]}, Departamento: {nuevo_caso1[0][2]}")
print(f"Valor estimado de la casa: ${prediccion_caso1[0]:,.2f}")

print("\nPredicción para el Segundo Caso:")
print(f"Ingreso: ${nuevo_caso2[0][0]:,.2f}, Hijos: {nuevo_caso2[0][1]}, Departamento: {nuevo_caso2[0][2]}")
print(f"Valor estimado de la casa: ${prediccion_caso2[0]:,.2f}")
