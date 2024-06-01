from pyspark.sql import SparkSession
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def main():
    # Инициализация Spark сессии
    spark = SparkSession.builder \
        .appName("IrisModelTraining") \
        .getOrCreate()

    # Загрузка данных Iris
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Инициализация и обучение модели RandomForest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Предсказание на тестовой выборке
    y_pred = model.predict(X_test)

    # Оценка точности модели
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Остановка Spark сессии
    spark.stop()


if __name__ == "__main__":
    main()
