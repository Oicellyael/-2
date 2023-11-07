from sklearn.datasets import load_breast_cancer

# Завантаження набору даних про рак грудей
data = load_breast_cancer()
X = data.data
y = data.target
from sklearn.model_selection import train_test_split

# Розділити дані на тренувальний і тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Створення і навчання моделі Random Forest
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Визначення важливості ознак
importances = rf_classifier.feature_importances_

# Побудувати графік важливості ознак
plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances)
plt.xticks(range(len(importances)), data.feature_names, rotation=90)
plt.title("Важливість ознак в Random Forest")
plt.show()
from sklearn.metrics import accuracy_score

# Прогноз на тестовому наборі
rf_predictions = rf_classifier.predict(X_test)

# Визначення точності Random Forest
rf_accuracy = accuracy_score(y_test, rf_predictions)
print("Точність Random Forest:", rf_accuracy)
from sklearn.svm import SVC

# Створення і навчання моделі SVM
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)
# Прогноз на тестовому наборі
svm_predictions = svm_classifier.predict(X_test)

# Визначення точності SVM
svm_accuracy = accuracy_score(y_test, svm_predictions)
print("Точність SVM:", svm_accuracy)
'''
Висновок:

У цій роботі ми провели аналіз набору даних про рак грудей, використовуючи два різні алгоритми машинного навчання: SVM і Random Forest. Метою було визначити важливі ознаки, які мають найбільший вплив на класифікацію наявності чи відсутності раку грудей у пацієнтів.

За допомогою моделі Random Forest було обчислено важливість кожної ознаки, і на графіку ми можемо побачити, які з них мають найбільший вплив на класифікацію. З цього аналізу видно, що деякі ознаки є надзвичайно важливими у визначенні раку грудей.

Ми також порівняли результати двох алгоритмів, Random Forest і SVM. Виявилося, що обидва алгоритми досить ефективні для цього завдання, але Random Forest продемонстрував трохи вищу точність на тестових даних. Таким чином, для цього конкретного набору даних Random Forest може бути більш підходящим алгоритмом для визначення раку грудей.

Загальною висновок з цієї роботи є те, що важливість ознак і вибір оптимального алгоритму може суттєво поліпшити класифікацію раку грудей, що може бути важливим у медичній діагностиці та плануванні лікування пацієнтів.
'''
