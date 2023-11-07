import pandas as pd

# Завантаження даних з файлу shopping_trends.csv
data = pd.read_csv('shopping_trends.csv')

# Виведення розміру таблиці
print("Розмір таблиці:", data.shape)

# Перевірка наявності пропусків
print("Пропуски в даних:\n", data.isnull().sum())

# Виведення типів стовпців
print("Типи стовпців:\n", data.dtypes)
# Вибір відповідних стовпців
selected_columns = ["Age", "Purchase Amount (USD)", "Review Rating", "Previous Purchases"]

# Створення нового DataFrame, що містить обрані стовпці
selected_data = data[selected_columns]

# Перейменування стовпців
selected_data.columns = ['age', 'purchase_amount_usd', 'review_rating', 'previous_purchases']
from sklearn.preprocessing import StandardScaler

# Створення екземпляру StandardScaler
scaler = StandardScaler()

# Масштабування даних
scaled_data = scaler.fit_transform(selected_data)
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Створення списку для зберігання значень вартості кластерів
cost = []

# Визначення кількості кластерів від 1 до 10 (можна змінити за потребою)
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    cost.append(kmeans.inertia_)

# Побудування графіку вартості кластерів
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), cost, marker='o', linestyle='--')
plt.xlabel('Кількість кластерів')
plt.ylabel('Вартість кластерів')
plt.title('Метод ліктя (Elbow Method)')
plt.show()
# Виберіть оптимальну кількість кластерів (наприклад, якщо ліктьова точка вказує на 3 кластери)
optimal_clusters = 3

# Створення екземпляру KMeans з оптимальною кількістю кластерів
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)


# Кластеризація даних
cluster_labels = kmeans.fit_predict(scaled_data)

# Додайте ідентифікатори кластера до DataFrame з обраними стовпцями
selected_data = selected_data.copy()
selected_data['cluster'] = cluster_labels


# Виведіть перші кілька рядків для перевірки
print(selected_data.head())
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# Створення екземпляру PCA зі зменшеною розмірністю (наприклад, 2)
pca = PCA(n_components=2)

# Застосування PCA до масштабованих даних
reduced_data = pca.fit_transform(scaled_data)
# Створення графіку для відображення кластерів
plt.figure(figsize=(8, 6))

# Створення графіку для кожного кластера
for cluster in range(optimal_clusters):
    plt.scatter(reduced_data[selected_data['cluster'] == cluster][:, 0],
                reduced_data[selected_data['cluster'] == cluster][:, 1],
                label=f'Кластер {cluster + 1}')

# Відображення легенди
plt.legend()

# Додаткові налаштування графіку
plt.title('Візуалізація результатів кластеризації')
plt.xlabel('Перша головна компонента')
plt.ylabel('Друга головна компонента')
plt.grid(True)

# Відображення графіку
plt.show()
'''В ході виконання даної роботи було проведено кластеризацію даних з використанням алгоритму K-means. Основні етапи роботи включали в себе підготовку даних, визначення оптимальної кількості кластерів, кластеризацію даних та візуалізацію результатів.

Підготовка даних включала в себе відбір числових стовпців та їх масштабування, щоб забезпечити однаковий масштаб для всіх ознак.

Для визначення оптимальної кількості кластерів був використаний метод ліктя (Elbow Method), який дозволив визначити кількість кластерів, що найкраще відображають структуру даних.

На підставі оптимальної кількості кластерів була проведена кластеризація даних за допомогою K-means алгоритму, і кожному об'єкту був призначений ідентифікатор кластера.

За допомогою методу головних компонентів (PCA) дані були зменшені до двовимірного простору, і результати кластеризації були візуалізовані на графіку.

Отже, в результаті роботи було успішно визначено кількість та структуру кластерів в наборі даних і створено графічне відображення цих кластерів. Визначення кластерів та їх аналіз може бути корисним для подальших досліджень та прийняття рішень в різних сферах діяльност.'''