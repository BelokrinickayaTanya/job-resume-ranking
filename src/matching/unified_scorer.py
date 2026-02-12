"""
Модуль объединенного скоринга с весами из архитектуры:
- Keyword matching: 80%
- TF-IDF matching: 15%
- Semantic matching: 5%
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime

from .keyword_matcher import KeywordMatcher
from .tfidf_matcher import AdaptiveTfidfMatcher
from .semantic_matcher import SemanticMatcher


@dataclass
class MatchResult:
    """Результат матчинга резюме и вакансии"""
    cv_id: int
    vacancy_id: int
    vacancy_title: str
    
    # Индивидуальные скоры
    keyword_score: float = 0.0
    tfidf_score: float = 0.0
    semantic_score: float = 0.0
    
    # Общий скор
    total_score: float = 0.0
    
    # Детали
    matched_skills: List[str] = field(default_factory=list)
    missing_skills: List[str] = field(default_factory=list)
    match_details: Dict = field(default_factory=dict)
    
    # Время расчета
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class UnifiedScorer:
    """
    Объединенный скоррер с весами из архитектуры:
    Keyword: 60% - точное и синонимичное совпадение навыков
    TF-IDF: 25% - важность терминов в контексте
    Semantic: 15% - семантическая близость текстов
    """
    
    # Веса компонентов (из архитектуры)
    WEIGHTS = {
        'keyword': 0.60,  # 60%
        'tfidf': 0.25,    # 25%
        'semantic': 0.15  # 15%
    }
    
    def __init__(
        self,
        use_adaptive_tfidf: bool = True,
        semantic_model: str = 'mini',
        device: str = 'cpu'
    ):
        """
        Args:
            use_adaptive_tfidf: Использовать AdaptiveTfidfMatcher с тех-бустом
            semantic_model: Модель для семантического матчинга
            device: 'cpu' или 'cuda'
        """
        # Инициализация матчеров
        self.keyword_matcher = KeywordMatcher(fuzzy_threshold=0.85)
        
        if use_adaptive_tfidf:
            self.tfidf_matcher = AdaptiveTfidfMatcher(max_features=2000, ngram_range=(1, 2))
        else:
            self.tfidf_matcher = AdaptiveTfidfMatcher(max_features=2000, ngram_range=(1, 2))
        
        self.semantic_matcher = SemanticMatcher(model_name=semantic_model, device=device)
        
        # Кэш для эмбеддингов
        self.embedding_cache = {}
        
        # Статистика
        self.stats = {
            'total_matches': 0,
            'avg_keyword_score': 0,
            'avg_tfidf_score': 0,
            'avg_semantic_score': 0,
            'avg_total_score': 0
        }
    
    def calculate_score(
        self,
        cv_id: int,
        cv_text: str,
        cv_skills: List[str],
        vacancy_id: int,
        vacancy_title: str,
        vacancy_text: str,
        vacancy_skills: List[str],
        cv_experience: Optional[float] = None,
        vacancy_required_years: Optional[int] = None,
        corpus: Optional[List[str]] = None
    ) -> MatchResult:
        """
        Полный расчет скора для пары резюме-вакансия
        """
        # 1. Keyword matching (80%)
        keyword_result = self.keyword_matcher.calculate_match_score(
            resume_skills=cv_skills,
            vacancy_skills=vacancy_skills,
            vacancy_text=vacancy_text,
            cv_experience=cv_experience,  
            vacancy_required_years=vacancy_required_years
        )
        keyword_score = keyword_result['score']
        
        # 2. TF-IDF matching (15%)
        tfidf_result = self.tfidf_matcher.calculate_similarity(
            resume_text=cv_text,
            vacancy_text=vacancy_text,
            corpus=corpus
        )
        tfidf_score = tfidf_result['score']
        
        # 3. Semantic matching (5%)
        semantic_result = self.semantic_matcher.calculate_similarity(
            resume_text=cv_text,
            vacancy_text=vacancy_text,
            use_chunks=True
        )
        semantic_score = semantic_result['score']
        
        # 4. Взвешенная сумма
        total_score = (
            self.WEIGHTS['keyword'] * keyword_score +
            self.WEIGHTS['tfidf'] * tfidf_score +
            self.WEIGHTS['semantic'] * semantic_score
        )
        
        # Округляем до 1 знака
        total_score = round(total_score, 1)
        
        # Создаем результат
        result = MatchResult(
            cv_id=cv_id,
            vacancy_id=vacancy_id,
            vacancy_title=vacancy_title,
            keyword_score=round(keyword_score, 1),
            tfidf_score=round(tfidf_score, 1),
            semantic_score=round(semantic_score, 1),
            total_score=total_score,
            matched_skills=keyword_result['matched'][:20],
            missing_skills=keyword_result['missing'][:20],
            match_details={
                'keyword': keyword_result,
                'tfidf': tfidf_result,
                'semantic': semantic_result
            }
        )
        
        # Обновляем статистику
        self._update_stats(result)
        
        return result
    
    def calculate_batch(
        self,
        cv_dict: Dict[int, Dict],
        vacancy_dict: Dict[int, Dict],
        corpus: Optional[List[str]] = None
    ) -> List[MatchResult]:
        """
        Пакетный расчет для всех пар резюме-вакансия
        """
        results = []
        total_pairs = len(cv_dict) * len(vacancy_dict)
        
        print(f"🔄 Расчет скоринга для {len(cv_dict)} резюме и {len(vacancy_dict)} вакансий...")
        print(f"📊 Всего пар: {total_pairs}")
        print(f"⚖️ Веса: Keyword={self.WEIGHTS['keyword']*100}%, TF-IDF={self.WEIGHTS['tfidf']*100}%, Semantic={self.WEIGHTS['semantic']*100}%")
        print("-" * 60)
        
        processed = 0
        for cv_id, cv_data in cv_dict.items():
            for vac_id, vac_data in vacancy_dict.items():
                result = self.calculate_score(
                    cv_id=cv_id,
                    cv_text=cv_data['text'],
                    cv_skills=cv_data['skills'],
                    vacancy_id=vac_id,
                    vacancy_title=vac_data['title'],
                    vacancy_text=vac_data['description'],
                    vacancy_skills=vac_data['skills'],
                    corpus=corpus
                )
                results.append(result)
                
                processed += 1
                if processed % 50 == 0:
                    print(f"  ⏳ Обработано {processed}/{total_pairs} пар...")
        
        print(f"  ✅ Обработано {processed}/{total_pairs} пар")
        print("-" * 60)
        
        return results
    
    def rank_vacancies_for_cv(
        self,
        cv_id: int,
        cv_data: Dict,
        vacancies: Dict[int, Dict],
        corpus: Optional[List[str]] = None
    ) -> List[MatchResult]:
        """
        Ранжирование вакансий для конкретного резюме
        """
        results = []
        
        for vac_id, vac_data in vacancies.items():
            result = self.calculate_score(
                cv_id=cv_id,
                cv_text=cv_data['text'],
                cv_skills=cv_data['skills'],
                vacancy_id=vac_id,
                vacancy_title=vac_data['title'],
                vacancy_text=vac_data['description'],
                vacancy_skills=vac_data['skills'],
                corpus=corpus
            )
            results.append(result)
        
        # Сортируем по убыванию total_score
        results.sort(key=lambda x: x.total_score, reverse=True)
        
        return results
    
    def get_ranking_array(self, results: List[MatchResult]) -> List[int]:
        """
        Конвертация результатов в массив рангов (1-5) для сравнения с ground truth
        
        1 = лучшая вакансия (наивысший score)
        5 = худшая вакансия (наименьший score)
        """
        # Сортируем по убыванию total_score
        sorted_results = sorted(results, key=lambda x: x.total_score, reverse=True)
        
        # Создаем массив рангов для vacancy_id 1-5
        ranks = [0] * 5
        
        for position, result in enumerate(sorted_results, 1):
            # vacancy_id уже 1-5
            vacancy_index = result.vacancy_id - 1
            ranks[vacancy_index] = position
        
        return ranks
    
    def _update_stats(self, result: MatchResult):
        """Обновление статистики"""
        self.stats['total_matches'] += 1
        n = self.stats['total_matches']
        
        # Скользящее среднее
        self.stats['avg_keyword_score'] += (result.keyword_score - self.stats['avg_keyword_score']) / n
        self.stats['avg_tfidf_score'] += (result.tfidf_score - self.stats['avg_tfidf_score']) / n
        self.stats['avg_semantic_score'] += (result.semantic_score - self.stats['avg_semantic_score']) / n
        self.stats['avg_total_score'] += (result.total_score - self.stats['avg_total_score']) / n
    
    def print_stats(self):
        """Вывод статистики"""
        print("\n📊 СТАТИСТИКА UNIFIED SCORER")
        print("=" * 50)
        print(f"Всего матчей: {self.stats['total_matches']}")
        print(f"Средний Keyword score: {self.stats['avg_keyword_score']:.1f}")
        print(f"Средний TF-IDF score: {self.stats['avg_tfidf_score']:.1f}")
        print(f"Средний Semantic score: {self.stats['avg_semantic_score']:.1f}")
        print(f"Средний Total score: {self.stats['avg_total_score']:.1f}")
        print("=" * 50)


class VMMethodAdapter:
    """
    Адаптер для Vector Matching метода 
    """
    
    def __init__(self):
        print("✅ VM Method Adapter initialized")
    
    def summarize_text(self, text: str, max_sentences: int = 10) -> str:
        """
        Суммаризация текста - берем первые N предложений
        Используется для VM метода из статьи
        """
        if not text:
            return ""
        
        import re
        # Разбиваем на предложения
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if len(sentences) <= max_sentences:
            return text[:2000]
        
        # Берем первые N предложений
        summary = '. '.join(sentences[:max_sentences]) + '.'
        return summary[:2000]
    
    def get_char_ngrams(self, text: str, n_range: Tuple[int, int] = (1, 3)):
        """
        Character n-gram векторизация
        """
        from sklearn.feature_extraction.text import CountVectorizer
        
        if len(text) > 5000:
            text = text[:5000]
        
        try:
            vectorizer = CountVectorizer(
                analyzer='char',
                ngram_range=n_range,
                lowercase=True,
                max_features=2000
            )
            vec = vectorizer.fit_transform([text])
            return vec
        except Exception as e:
            print(f"⚠️ Vectorization error: {e}")
            from scipy.sparse import csr_matrix
            return csr_matrix((1, 2000))
    
    def l1_distance(self, vec1, vec2) -> float:
        """
        L1 расстояние для спарс-векторов
        """
        try:
            from scipy.spatial.distance import cityblock
            
            if hasattr(vec1, 'toarray'):
                vec1 = vec1.toarray().flatten()
            if hasattr(vec2, 'toarray'):
                vec2 = vec2.toarray().flatten()
            
            min_len = min(len(vec1), len(vec2))
            vec1 = vec1[:min_len]
            vec2 = vec2[:min_len]
            
            return cityblock(vec1, vec2)
        except Exception as e:
            print(f"⚠️ Distance error: {e}")
            return 1000000.0
    
    def get_vector_fixed(self, text: str, max_features: int = 1000) -> np.ndarray:
        """
        Получение вектора ФИКСИРОВАННОЙ размерности
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        try:
            vec = vectorizer.fit_transform([text]).toarray()[0]
            # Нормализация
            if np.linalg.norm(vec) > 0:
                vec = vec / np.linalg.norm(vec)
            return vec
        except:
            return np.zeros(max_features)
    
    def l1_distance_fixed(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        L1 расстояние для векторов ОДИНАКОВОЙ размерности
        """
        from scipy.spatial.distance import cityblock
        # Убеждаемся, что векторы одной длины
        assert len(vec1) == len(vec2), f"Vector dimensions mismatch: {len(vec1)} vs {len(vec2)}"
        return cityblock(vec1, vec2)