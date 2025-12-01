#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Анализ тональности отзывов
Проект по классификации отзывов на позитивные/негативные
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
import string

def main():
    print("=== Анализ тональности отзывов ===\n")
    
    # Пример данных
    data = {
        'text': [
            'Отличный товар, очень доволен покупкой!',
            'Качество ужасное, не рекомендую.',
            'Нормально, но есть недостатки.',
            'Прекрасный сервис, спасибо!',
            'Разочарован, не оправдало ожиданий.'
        ],
        'sentiment': [1, 0, 0, 1, 0]  # 1=позитивный, 0=негативный
    }
    
    df = pd.DataFrame(data)
    print("Данные:")
    print(df)
    print()
    
    # Векторизация текста
    vectorizer = TfidfVectorizer(stop_words=stopwords.words('russian'))
    X = vectorizer.fit_transform(df['text'])
    y = df['sentiment']
    
    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Обучение модели
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    # Предсказания
    y_pred = model.predict(X_test)
    
    # Оценка
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nТочность модели: {accuracy:.2%}")
    print("\nОтчёт по классификации:")
    print(classification_report(y_test, y_pred))
    
    # Пример предсказания
    new_review = ["Хорошее качество за свои деньги"]
    new_vector = vectorizer.transform(new_review)
    prediction = model.predict(new_vector)[0]
    
    sentiment = "позитивный" if prediction == 1 else "негативный"
    print(f"\nПример предсказания для отзыва '{new_review[0]}': {sentiment}")

if __name__ == "__main__":
    main()
EOFcat nlp/sentiment_analysis.py
git add nlp/sentiment_analysis.py
git commit -m "Добавил полноценный код NLP анализа с sklearn"
git push origin main
cat > nlp/sentiment_analysis.py << 'EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Анализ тональности отзывов
Проект по классификации отзывов на позитивные/негативные
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
import string

def main():
    print("=== Анализ тональности отзывов ===\n")
    
    # Пример данных
    data = {
        'text': [
            'Отличный товар, очень доволен покупкой!',
            'Качество ужасное, не рекомендую.',
            'Нормально, но есть недостатки.',
            'Прекрасный сервис, спасибо!',
            'Разочарован, не оправдало ожиданий.'
        ],
        'sentiment': [1, 0, 0, 1, 0]  # 1=позитивный, 0=негативный
    }
    
    df = pd.DataFrame(data)
    print("Данные:")
    print(df)
    print()
    
    # Векторизация текста
    vectorizer = TfidfVectorizer(stop_words=stopwords.words('russian'))
    X = vectorizer.fit_transform(df['text'])
    y = df['sentiment']
    
    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Обучение модели
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    # Предсказания
    y_pred = model.predict(X_test)
    
    # Оценка
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nТочность модели: {accuracy:.2%}")
    print("\nОтчёт по классификации:")
    print(classification_report(y_test, y_pred))
    
    # Пример предсказания
    new_review = ["Хорошее качество за свои деньги"]
    new_vector = vectorizer.transform(new_review)
    prediction = model.predict(new_vector)[0]
    
    sentiment = "позитивный" if prediction == 1 else "негативный"
    print(f"\nПример предсказания для отзыва '{new_review[0]}': {sentiment}")

if __name__ == "__main__":
    main()
