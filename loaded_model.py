# Zadanie 3.2 model wczytuje się poprawnie, aczkolwiek, przez przeuczenie na poprzednim zbiorze, jego wyniki są mierne.
import sklearn
import pickle

iris = sklearn.datasets.load_iris()
X = iris.data
y = iris.target

with open('random_forest_classifier.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3, random_state=42)
y_pred_loaded = loaded_model.predict(X_test)
print(sklearn.metrics.accuracy_score(y_test, y_pred_loaded))
print(sklearn.metrics.precision_score(y_test, y_pred_loaded, average='weighted', zero_division=0))
print(sklearn.metrics.recall_score(y_test, y_pred_loaded, average='weighted'))
print(sklearn.metrics.f1_score(y_test, y_pred_loaded, average='weighted', zero_division=0))
