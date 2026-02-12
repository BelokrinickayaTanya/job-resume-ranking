"""
Модуль TF-IDF матчинга (15% веса в финальном скоре)
"""
from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


class TfidfMatcher:
    """
    Матчинг на основе TF-IDF векторизации
    
    Вес в финальном скоре: 15%
    """
    
    def __init__(self, max_features: int = 1000, ngram_range: Tuple[int, int] = (1, 2)):
        """
        Args:
            max_features: Максимальное количество признаков
            ngram_range: Диапазон n-грамм (1,2) = униграммы + биграммы
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            lowercase=True,
            analyzer='word',
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True  # log-scale TF
        )
        self.is_fitted = False
        
    def fit(self, documents: List[str]) -> 'TfidfMatcher':
        """
        Обучение векторизатора на корпусе документов
        
        Args:
            documents: Список текстов для обучения
        """
        if not self.is_fitted:
            self.vectorizer.fit(documents)
            self.is_fitted = True
        return self
    
    def transform(self, text: str) -> np.ndarray:
        """Преобразование текста в TF-IDF вектор"""
        if not self.is_fitted:
            raise ValueError("Модель не обучена. Вызовите fit() сначала.")
        return self.vectorizer.transform([text]).toarray()[0]
    
    def calculate_similarity(
        self, 
        resume_text: str, 
        vacancy_text: str,
        corpus: Optional[List[str]] = None
    ) -> Dict:
        """
        Расчет TF-IDF схожести между резюме и вакансией
        
        Returns:
            {
                'score': 0-100,
                'similarity': float,
                'top_terms': [...],
                'shared_terms': [...]
            }
        """
        # Подготавливаем корпус для IDF
        if corpus is None:
            corpus = [resume_text, vacancy_text]
        
        if not self.is_fitted:
            self.fit(corpus)
        
        # Векторизация
        resume_vec = self.transform(resume_text)
        vacancy_vec = self.transform(vacancy_text)
        
        # Cosine similarity (0-1)
        similarity = self._cosine_similarity(resume_vec, vacancy_vec)
        
        # Нормализация в 0-100
        score = min(100, similarity * 100)
        
        # Получаем важные термины
        top_terms_resume = self._get_top_terms(resume_vec, n=10)
        top_terms_vacancy = self._get_top_terms(vacancy_vec, n=10)
        
        # Общие термины
        shared_terms = set(top_terms_resume) & set(top_terms_vacancy)
        
        return {
            'score': round(score, 1),
            'similarity': round(similarity, 4),
            'top_terms_resume': top_terms_resume[:5],
            'top_terms_vacancy': top_terms_vacancy[:5],
            'shared_terms': list(shared_terms)[:10]
        }
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Вычисление косинусного сходства"""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def _get_top_terms(self, vector: np.ndarray, n: int = 10) -> List[str]:
        """Получение наиболее важных терминов"""
        feature_names = self.vectorizer.get_feature_names_out()
        
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        
        # Индексы топ-N терминов
        top_indices = np.argsort(vector[0])[-n:][::-1]
        
        return [feature_names[i] for i in top_indices if vector[0][i] > 0]
    
    def get_feature_importance(self, text: str) -> Dict[str, float]:
        """Получение важности каждого термина в тексте"""
        vector = self.transform(text)
        feature_names = self.vectorizer.get_feature_names_out()
        
        importance = {}
        for i, weight in enumerate(vector):
            if weight > 0:
                importance[feature_names[i]] = round(weight, 4)
        
        # Сортируем по убыванию
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20])


class AdaptiveTfidfMatcher(TfidfMatcher):
    """
    Адаптивный TF-IDF матчинг с учетом домена IT-рекрутмента
    """
    
    # Технические термины, которые должны иметь повышенный вес
    TECH_BOOST = {
        'python', 'java', 'javascript', 'react', 'angular', 'vue', 'docker',
        'kubernetes', 'aws', 'azure', 'sql', 'nosql', 'mongodb', 'postgresql',
        'c++', 'c#', 'php', 'ruby', 'go', 'rust', 'typescript', 'nodejs',
        'django', 'flask', 'spring', 'hibernate', 'jpa', 'rest', 'graphql'
    }
    
    def calculate_similarity(self, resume_text: str, vacancy_text: str, corpus: Optional[List[str]] = None) -> Dict:
        """Расчет схожести с бустом для технических терминов"""
        
        # Стандартный расчет
        result = super().calculate_similarity(resume_text, vacancy_text, corpus)
        
        # Буст для технических терминов
        resume_boost = self._calculate_tech_boost(resume_text, vacancy_text)
        
        # Комбинируем score
        boosted_score = result['score'] * 0.7 + resume_boost * 30
        
        result['score'] = round(min(100, boosted_score), 1)
        result['tech_boost'] = round(resume_boost, 3)
        
        return result
    
    def _calculate_tech_boost(self, resume_text: str, vacancy_text: str) -> float:
        """Расчет буста на основе технических навыков"""
        resume_lower = resume_text.lower()
        vacancy_lower = vacancy_text.lower()
        
        tech_matches = 0
        tech_total = 0
        
        for tech in self.TECH_BOOST:
            if tech in vacancy_lower:
                tech_total += 1
                if tech in resume_lower:
                    tech_matches += 1
        
        if tech_total == 0:
            return 0.0
        
        return tech_matches / tech_total