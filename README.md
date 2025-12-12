#  Predicción de Aprobación de Curso de Matemáticas mediante Redes Neuronales

## Equipo de Trabajo
- **Juan Esteban Ortiz** - 2410227-3743
- **Juan David Olaya** - 202410206-3743
- **Pablo Esteban Becerra** - 202243506-3743
- **Fernando Cardona Giraldo** - 202241381-3743
- **Sara Yineth Suarez Reyes** - 202241923-3743

---

##  Descripción del Proyecto

Este proyecto implementa modelos de **redes neuronales multicapa (MLP)** para predecir si un estudiante aprobará un curso de matemáticas basándose en sus características demográficas, familiares y hábitos de estudio.

### Objetivo
Desarrollar y comparar diferentes arquitecturas de redes neuronales para clasificar estudiantes en dos categorías:
- **Aprueba** (approved = 1)
- **No aprueba** (approved = 0)

---

## Estructura del Proyecto

```
ActividadML/
│
├── notebook1.ipynb                    # Actividad 1: Redes Neuronales
├── notebook2.ipynb                    # Actividad 2: Árboles de Decisión
├── student_performance.csv            # Dataset de estudiantes
├── Informe Machine learning.pdf       # Informe completo del proyecto
└── README.md                          # Este archivo
```

---

## Dataset

**Archivo:** `student_performance.csv`

### Características del Dataset:
- **Total de registros:** 1,044 estudiantes
- **Atributos totales:** 17 variables
- **Variable objetivo:** `approved` (binaria: 0 o 1)

### Atributos Numéricos (9):
1. `age` - Edad del estudiante
2. `Medu` - Educación de la madre (0-4)
3. `Fedu` - Educación del padre (0-4)
4. `traveltime` - Tiempo de viaje al colegio
5. `studytime` - Tiempo de estudio semanal
6. `failures` - Número de materias reprobadas
7. `goout` - Frecuencia de salidas
8. `Walc` - Consumo de alcohol fin de semana
9. `health` - Estado de salud (1-5)

### Atributos Categóricos (7):
1. `sex` - Sexo (M/F)
2. `famsize` - Tamaño de familia
3. `Pstatus` - Estado de convivencia de los padres
4. `Mjob` - Ocupación de la madre
5. `Fjob` - Ocupación del padre
6. `internet` - Acceso a internet (yes/no)
7. `romantic` - En relación romántica (yes/no)

---

## Tecnologías y Librerías

```python
- Python 3.x
- scikit-learn      # Modelos de ML y preprocesamiento
- pandas            # Manipulación de datos
- numpy             # Operaciones numéricas
- matplotlib        # Visualización
```

---

## Pipeline de Preprocesamiento

### 1. **Pipeline para Atributos Numéricos**
```python
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),    # Imputación de valores faltantes
    ("scaler", StandardScaler())                       # Normalización (μ=0, σ=1)
])
```

### 2. **Pipeline para Atributos Categóricos**
```python
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),  # Imputación con moda
    ("cat_encoder", OneHotEncoder(sparse_output=False))     # Codificación One-Hot
])
```

### 3. **Pipeline Completo**
```python
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs),
])
```

**Resultado:** 9 atributos numéricos + codificación categórica → matriz de características lista para entrenamiento

---

---

# ACTIVIDAD 1: Redes Neuronales (notebook1.ipynb)

## Modelos Implementados

Se entrenaron **5 modelos de redes neuronales** con diferentes configuraciones:

| Modelo | Función Activación | Solver | Arquitectura | Learning Rate |
|--------|-------------------|--------|--------------|---------------|
| **Modelo 1** | ReLU | LBFGS | (10, 5) | - |
| **Modelo 2** | Identity | SGD | (5, 6, 7) | constant |
| **Modelo 3** | Tanh | Adam | (8, 3, 2, 6) | - |
| **Modelo 4** | Logistic | Adam | (20,) | - |
| **Modelo 5** | ReLU | Adam | (3, 5, 7, 9, 10) | - |

### Parámetros Comunes:
- **División de datos:** 80% entrenamiento / 20% prueba
- **Métrica de evaluación:** Accuracy
- **Max iteraciones:** 1000

---

## Resultados

### Primera Ejecución (sin learning_rate):
- **Mejor modelo:** Modelo 2
- **Accuracy:** 0.8182
- **Configuración:** activation='identity', solver='sgd', hidden_layer_sizes=(5,6,7)

### Optimización con learning_rate:
| Configuración | Accuracy | Observación |
|--------------|----------|-------------|
| `learning_rate='constant'` | **0.8278** |  Mejor resultado |
| `learning_rate='adaptive'` | 0.8086 | Disminución |
| `learning_rate='invscaling'` | 0.7273 | Peor resultado |

### Conclusión Actividad 1:
El **Modelo 2 con learning_rate='constant'** logró el mejor desempeño con **82.78% de accuracy**.

---

# ACTIVIDAD 2: Árboles de Decisión (notebook2.ipynb)

## Descripción

Esta actividad aplica **árboles de decisión** al mismo dataset de estudiantes, comparando diferentes configuraciones de hiperparámetros para optimizar el rendimiento del clasificador.

## Objetivo

Determinar los hiperparámetros óptimos para un árbol de decisión que prediga la aprobación del curso de matemáticas, experimentando con:
- Diferentes profundidades del árbol (`max_depth`)
- Criterios de impureza (`gini` vs `entropy`)
- Número mínimo de muestras para dividir (`min_samples_split`)

## Modelos de Árboles de Decisión

### Experimento 1: Variación de max_depth con criterio Gini

| Modelo | max_depth | Criterio | Accuracy |
|--------|-----------|----------|----------|
| Modelo 1 | 2 | gini | ~0.74 |
| Modelo 2 | 4 | gini | **0.8373** |
| Modelo 3 | 6 | gini | ~0.81 |
| Modelo 4 | 8 | gini | ~0.80 |
| Modelo 5 | 10 | gini | ~0.78 |

### Experimento 2: Variación de max_depth con criterio Entropy

| Modelo | max_depth | Criterio | Accuracy |
|--------|-----------|----------|----------|
| Modelo 1 | 2 | entropy | ~0.74 |
| Modelo 2 | 4 | entropy | **0.8373** |
| Modelo 3 | 6 | entropy | ~0.81 |
| Modelo 4 | 8 | entropy | ~0.80 |
| Modelo 5 | 10 | entropy | ~0.78 |

### Observación Importante:
Ambos criterios (`gini` y `entropy`) producen **resultados idénticos** con `max_depth=4`, logrando **83.73% de accuracy**. Esto indica que ambos métodos encuentran las mismas divisiones óptimas en el árbol.

### Experimento 3: Variación de min_samples_split

Con los mejores hiperparámetros (`max_depth=4`, `criterion='gini'`):

| min_samples_split | Accuracy |
|-------------------|----------|
| 2 | 0.8373 |
| 10 | 0.8373 |
| 20 | 0.8373 |

**Conclusión:** El parámetro `min_samples_split` **no afecta** el accuracy cuando `max_depth=4`, ya que la profundidad máxima limita el crecimiento del árbol antes de que este parámetro entre en acción.

## Configuración Óptima del Árbol

```python
DecisionTreeClassifier(
    max_depth=4,
    criterion='gini',  # o 'entropy' (mismo resultado)
    min_samples_split=2
)
```

**Accuracy alcanzado:** **83.73%**

## 📊 Comparación: Redes Neuronales vs Árboles de Decisión

| Técnica | Mejor Accuracy | Configuración |
|---------|----------------|---------------|
| **Redes Neuronales** | 82.78% | MLPClassifier: (5,6,7), SGD, identity, lr=constant |
| **Árboles de Decisión** | **83.73%** ✅ | DecisionTreeClassifier: max_depth=4, gini |

###  Conclusiones Comparativas:

1. **Los árboles de decisión superan ligeramente** a las redes neuronales (+0.95%)

2. **Simplicidad vs Complejidad:**
   - Árboles: Más simples, interpretables, entrenamiento rápido
   - Redes neuronales: Más complejas, requieren más ajuste de hiperparámetros

3. **Interpretabilidad:** Los árboles permiten visualizar las reglas de decisión

4. **Robustez:** Ambos criterios (gini/entropy) producen el mismo árbol, indicando estabilidad

---

## Cómo Ejecutar el Proyecto

### 1. **Clonar el repositorio**
```bash
git clone https://github.com/Juanes-xd/MachineLearning.git
cd MachineLearning/ActividadML
```

### 2. **Instalar dependencias**
```bash
pip install scikit-learn pandas numpy matplotlib jupyter
```

### 3. **Ejecutar el notebook**
```bash
jupyter notebook notebook1.ipynb
```

### 4. **Ejecutar todas las celdas**
En Jupyter: `Cell > Run All`

---


##  Referencias

- [Documentación scikit-learn - MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
- [Student Performance Dataset](https://archive.ics.uci.edu/ml/datasets/student+performance)


