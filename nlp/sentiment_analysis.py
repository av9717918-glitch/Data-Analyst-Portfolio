"""
Анализ тональности отзывов на товары
Используется: pandas, sklearn, nltk
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk

print("=== Анализ тональности отзывов ===")
print("Загрузка данных...")

# Создаем пример данных
data = {
    'review': [
        'Отличный товар, очень доволен',
        'Ужасное качество, не покупайте',
        'Нормально за свои деньги',
        'Супер, рекомендую всем',
        'Разочарован, ожидал большего'
    ],
    'sentiment': [1, 0, 0, 1, 0]  # 1=положительный, 0=отрицательный
}

df = pd.DataFrame(data)
print(f"Данные:\n{df}\n")

# Векторизация текста
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['review'])
y = df['sentiment']

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели
model = MultinomialNB()
model.fit(X_train, y_train)

# Предсказание
y_pred = model.predict(X_test)

# Оценка
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели: {accuracy:.1%}")

# Тестовый отзыв
test_review = ["Хорошее качество за эти деньги"]
test_vector = vectorizer.transform(test_review)
prediction = model.predict(test_vector)[0]
result = "положительный" if prediction == 1 else "отрицательный"
print(f"Отзыв '{test_review[0]}' — {result}")
