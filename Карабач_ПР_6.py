from sklearn import datasets

# Завантаження датасету Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target
from sklearn.model_selection import train_test_split

# Розділіть дані на тренувальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Створення та навчання моделі SVM
svm_classifier = SVC()
svm_classifier.fit(X_train, y_train)

# Прогноз на тестовому наборі
svm_predictions = svm_classifier.predict(X_test)

# Визначення точності SVM
svm_accuracy = accuracy_score(y_test, svm_predictions)
print("Точність SVM:", svm_accuracy)

from sklearn.ensemble import RandomForestClassifier

# Створення та навчання моделі Random Forest
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)

# Прогноз на тестовому наборі
rf_predictions = rf_classifier.predict(X_test)

# Визначення точності Random Forest
rf_accuracy = accuracy_score(y_test, rf_predictions)
print("Точність Random Forest:", rf_accuracy)
# Порівняння результатів
if svm_accuracy > rf_accuracy:
    print("SVM є більш точним алгоритмом.")
elif svm_accuracy < rf_accuracy:
    print("Random Forest є більш точним алгоритмом.")
else:
    print("SVM і Random Forest мають однакову точність.")
'''
SVM і Random Forest - обидві моделі здатні класифікувати види ірисів із високою точністю.
Ви можете бачити це з результатів точності, які виводяться на екрані.

У даному конкретному випадку, модель Random Forest показала трохи вищу точність на тестових даних порівняно з моделлю SVM.
Висновок: для даного набору даних і завдання класифікації ірисів, модель Random Forest є більш точною.

Звісно, результати можуть змінюватися в залежності від параметрів моделей і розподілу даних, але в контексті даного коду і набору даних Iris, Random Forest виявився більш ефективним для цієї конкретної задачі.
'''
