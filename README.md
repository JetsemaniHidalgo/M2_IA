# M2_IA

Este repositorio contiene un código en Python que realiza un análisis exploratorio de datos y utiliza un modelo de regresión lineal para predecir los valores de las casas en función de varias variables independientes. A continuación, se presenta una descripción de los pasos clave y el funcionamiento del código.

Funcionamiento de los pasos de código:

Carga de Datos:
  El código comienza importando las bibliotecas necesarias, incluyendo Pandas, Scikit-Learn, Seaborn y Matplotlib.
  Carga el conjunto de datos desde el archivo 'casas.csv' utilizando Pandas.

Codificación de Etiquetas
  Utiliza LabelEncoder de Scikit-Learn para codificar la columna 'Departamento' en valores numéricos (0 para 'IT' y 1 para 'RH').
  
Visualización de Datos
  Crea gráficos de histogramas y gráficos de dispersión para analizar la relación entre las variables numéricas ('Salario', 'Hijos', 'Departamento') y la variable objetivo 'Valor_casa'.

Matriz de Correlación
  Calcula y visualiza la matriz de correlación entre las variables numéricas y 'Valor_casa' mediante un mapa de calor.

División de Datos
  Divide el conjunto de datos en conjuntos de entrenamiento y prueba utilizando la función train_test_split de Scikit-Learn.
  
Entrenamiento del Modelo de Regresión Lineal
  Crea una instancia del modelo de regresión lineal y lo entrena utilizando los datos de entrenamiento.
  
Predicciones y Métricas de Evaluación
  Realiza predicciones tanto en el conjunto de entrenamiento como en el conjunto de prueba.
  Calcula el Error Cuadrático Medio (MSE) y el Coeficiente de Determinación (R^2) para evaluar el rendimiento del modelo en ambos conjuntos.

Predicciones para Nuevos Casos
  Realiza predicciones para dos nuevos casos dados como entrada, mostrando el ingreso, el número de hijos y el departamento, junto con el valor estimado de la  casa.

Observaciones personales: Con respecto al proyecto tanto el contenido del curso como las clases por parte del maestro han hecho que el tema sea entendible y poder realizar esta actividad con un mejor resultado, ya que permite pulir ciertos aspectos, como fue el caso de las graficas en una sola hoja, ya que yo anteriormente y en este proyecto lo estube desarrollando una por una, ahora ya puedo hacer las graficas en una sola hoja y esto permite comparar y analizar mejor, considero que debo mejorar en mi buenas practicas para tener un codigo mas limpio y mejor.

Observaciones personales: En relación al proyecto, tanto el contenido del curso como las clases impartidas por el profesor han contribuido significativamente a una mejor comprensión del tema y, como resultado, a la realización exitosa de esta actividad. Este enfoque ha permitido pulir ciertos aspectos, como la creación de gráficas en una sola hoja. Anteriormente, solía desarrollarlas una por una, pero ahora tengo la capacidad de generarlas en una única hoja, lo que facilita la comparación y el análisis. Reconozco que debo de mejorar mis buenas prácticas para mantener un código más limpio y de mayor calidad en el futuro.
