import numpy as np
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist

# Implementar el Clasificador Euclidiano
class EuclideanClassifier:
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        # Calcula las distancias euclidianas de cada punto a los puntos de entrenamiento
        distances = cdist(X, self.X_train, 'euclidean')
        # Encuentra el índice del punto de entrenamiento más cercano
        nearest_neighbor_indices = np.argmin(distances, axis=1)
        # Asigna la clase del punto de entrenamiento más cercano
        return self.y_train[nearest_neighbor_indices]

# Método de Hold Out 70/30 estratificado
def hold_out_stratified(data, target):
    return train_test_split(data, target, test_size=0.3, stratify=target, random_state=42)

# Método de 10-Fold Cross-Validation estratificado
def cross_validation_stratified(data, target):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    return list(skf.split(data, target))

# Función para evaluar el clasificador
def evaluate_classifier(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred)

# Cargar los conjuntos de datos
iris_data = load_iris()
wine_data = load_wine()

# Conjuntos de datos
iris_X = iris_data.data
iris_y = iris_data.target

wine_X = wine_data.data
wine_y = wine_data.target

# Inicializar el clasificador
clf = EuclideanClassifier()

# Aplicar métodos de validación al conjunto de datos Iris
print("Validación con Iris Dataset:")

# Hold Out 70/30
iris_X_train, iris_X_test, iris_y_train, iris_y_test = hold_out_stratified(iris_X, iris_y)
iris_holdout_accuracy = evaluate_classifier(clf, iris_X_train, iris_X_test, iris_y_train, iris_y_test)
print("Hold Out 70/30 Accuracy:", iris_holdout_accuracy)

# 10-Fold Cross-Validation
iris_splits = cross_validation_stratified(iris_X, iris_y)
iris_cv_accuracies = []
for train_index, test_index in iris_splits:
    X_train, X_test = iris_X[train_index], iris_X[test_index]
    y_train, y_test = iris_y[train_index], iris_y[test_index]
    accuracy = evaluate_classifier(clf, X_train, X_test, y_train, y_test)
    iris_cv_accuracies.append(accuracy)
print("10-Fold Cross-Validation Accuracy:", np.mean(iris_cv_accuracies))

# Aplicar métodos de validación al conjunto de datos Wine
print("\nValidación con Wine Dataset:")

# Hold Out 70/30
wine_X_train, wine_X_test, wine_y_train, wine_y_test = hold_out_stratified(wine_X, wine_y)
wine_holdout_accuracy = evaluate_classifier(clf, wine_X_train, wine_X_test, wine_y_train, wine_y_test)
print("Hold Out 70/30 Accuracy:", wine_holdout_accuracy)

# 10-Fold Cross-Validation
wine_splits = cross_validation_stratified(wine_X, wine_y)
wine_cv_accuracies = []
for train_index, test_index in wine_splits:
    X_train, X_test = wine_X[train_index], wine_X[test_index]
    y_train, y_test = wine_y[train_index], wine_y[test_index]
    accuracy = evaluate_classifier(clf, X_train, X_test, y_train, y_test)
    wine_cv_accuracies.append(accuracy)
print("10-Fold Cross-Validation Accuracy:", np.mean(wine_cv_accuracies))  # Corregido aquí