"""
Модуль ранжирования вакансий для резюме
"""
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from collections import defaultdict
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime

from ..matching.unified_scorer import UnifiedScorer, MatchResult, VMMethodAdapter


@dataclass
class RankingResult:
    """Результат ранжирования вакансий для одного резюме"""
    cv_id: int
    rankings: List[int]  # [rank_vac1, rank_vac2, rank_vac3, rank_vac4, rank_vac5]
    scores: List[float]  # scores for vacancies 1-5
    method: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return {
            'cv_id': self.cv_id,
            'rankings': self.rankings,
            'scores': self.scores,
            'method': self.method
        }


class VacancyRanker:
    """
    Ранжирование вакансий на основе скоров матчинга
    
    Поддерживает несколько методов ранжирования:
    1. Unified Scoring (Keyword 80% + TF-IDF 15% + Semantic 5%)
    2. VM Method (Vector Matching из статьи)
    3. OKAPI BM25 Baseline
    4. BERT-rank Baseline
    """
    
    def __init__(self):
        self.unified_scorer = UnifiedScorer(
            use_adaptive_tfidf=True,
            semantic_model='mini',
            device='cpu'
        )
        self.vm_method = VMMethodAdapter()
        
    def rank_unified(
        self,
        cv_id: int,
        cv_text: str,
        cv_skills: List[str],
        vacancies: Dict[int, Dict],
        cv_experience: Optional[float] = None,
    ) -> RankingResult:
        """
        Ранжирование методом Unified Scoring (60/25/15)
        """
        results = []

        cv_experience = None
        
        for vac_id, vac_data in vacancies.items():
            # Расчет скора
            result = self.unified_scorer.calculate_score(
                cv_id=cv_id,
                cv_text=cv_text,
                cv_skills=cv_skills,
                vacancy_id=vac_id,
                vacancy_title=vac_data['title'],
                vacancy_text=vac_data['description'],
                vacancy_skills=vac_data['skills'],
                cv_experience=cv_experience,
                vacancy_required_years=vac_data.get('required_years'),
                corpus=None
            )
            results.append(result)
        
        # Сортировка по убыванию total_score
        sorted_results = sorted(results, key=lambda x: x.total_score, reverse=True)
        
        # Формирование массива рангов (1-5)
        rankings = [0] * 5
        scores = [0.0] * 5
        
        for position, result in enumerate(sorted_results, 1):
            vacancy_index = result.vacancy_id - 1
            rankings[vacancy_index] = position
            scores[vacancy_index] = result.total_score
        
        return RankingResult(
            cv_id=cv_id,
            rankings=rankings,
            scores=scores,
            method='unified'
        )
    
    def rank_vm_method(self, cv_id: int, cv_text: str, vacancies: Dict[int, Dict]) -> RankingResult:
        """
        Ранжирование методом Vector Matching
        """
        distances = []
        
        # 1. Вектор резюме (фиксированный размер)
        cv_vec = self.vm_method.get_vector_fixed(cv_text, max_features=2000)
        
        for vac_id, vac_data in vacancies.items():
            # 2. Саммаризация вакансии (BERT)
            vac_summary = self.vm_method.summarize_text(vac_data['description'])
            
            # 3. Вектор вакансии (фиксированный размер)
            vac_vec = self.vm_method.get_vector_fixed(vac_summary, max_features=2000)
            
            # 4. L1 расстояние
            distance = np.sum(np.abs(cv_vec - vac_vec))  # L1 distance
            distances.append((vac_id, distance))
        
        # Сортировка по возрастанию расстояния
        distances.sort(key=lambda x: x[1])
        
        # Формирование рангов
        rankings = [0] * 5
        scores = [0.0] * 5
        
        for position, (vac_id, dist) in enumerate(distances, 1):
            rankings[vac_id - 1] = position
            scores[vac_id - 1] = 100 / (1 + dist)
        
        return RankingResult(
            cv_id=cv_id,
            rankings=rankings,
            scores=scores,
            method='vm_method'
        )

    
    def rank_okapi_bm25(
        self,
        cv_id: int,
        cv_text: str,
        vacancies: Dict[int, Dict]
    ) -> RankingResult:
        """
        Baseline: OKAPI BM25 алгоритм
        
        Использует библиотеку rank_bm25
        """
        try:
            from rank_bm25 import BM25Okapi
            import nltk
            from nltk.tokenize import word_tokenize
            
            # Токенизация
            tokenized_cv = word_tokenize(cv_text.lower())
            tokenized_vacancies = []
            vac_ids = []
            
            for vac_id, vac_data in vacancies.items():
                tokens = word_tokenize(vac_data['description'].lower())
                tokenized_vacancies.append(tokens)
                vac_ids.append(vac_id)
            
            # BM25
            bm25 = BM25Okapi(tokenized_vacancies)
            scores = bm25.get_scores(tokenized_cv)
            
            # Сортировка по убыванию скоров
            sorted_indices = np.argsort(scores)[::-1]
            
            rankings = [0] * 5
            norm_scores = [0.0] * 5
            
            for position, idx in enumerate(sorted_indices, 1):
                vac_id = vac_ids[idx]
                vacancy_index = vac_id - 1
                rankings[vacancy_index] = position
                norm_scores[vacancy_index] = float(scores[idx])
            
            return RankingResult(
                cv_id=cv_id,
                rankings=rankings,
                scores=norm_scores,
                method='okapi_bm25'
            )
            
        except ImportError:
            print("⚠️ rank_bm25 не установлен. Используется fallback метод.")
            return self._fallback_bm25(cv_id, cv_text, vacancies)
    
    def _fallback_bm25(self, cv_id: int, cv_text: str, vacancies: Dict[int, Dict]) -> RankingResult:
        """Fallback реализация BM25"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        texts = [cv_text] + [v['description'] for v in vacancies.values()]
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        tfidf = vectorizer.fit_transform(texts)
        
        similarities = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
        
        sorted_indices = np.argsort(similarities)[::-1]
        vac_ids = list(vacancies.keys())
        
        rankings = [0] * 5
        scores = [0.0] * 5
        
        for position, idx in enumerate(sorted_indices, 1):
            vac_id = vac_ids[idx]
            vacancy_index = vac_id - 1
            rankings[vacancy_index] = position
            scores[vacancy_index] = float(similarities[idx] * 100)
        
        return RankingResult(
            cv_id=cv_id,
            rankings=rankings,
            scores=scores,
            method='okapi_bm25_fallback'
        )
    
    def rank_bert(self, cv_id: int, cv_text: str, vacancies: Dict[int, Dict]) -> RankingResult:
        """BERT-rank """
        from sentence_transformers import SentenceTransformer
        
        # Кэшируем модель
        if not hasattr(self, '_bert_model'):
            self._bert_model = SentenceTransformer('all-mpnet-base-v2')  # Более точная модель!
        model = self._bert_model
        
        # Берем больше контекста
        cv_emb = model.encode(cv_text[:2000], normalize_embeddings=True)
        
        scores = []
        vac_ids = []
        
        for vac_id, vac_data in vacancies.items():
            vac_text = vac_data['description'][:2000]
            vac_emb = model.encode(vac_text, normalize_embeddings=True)
            
            # Косинусная близость
            similarity = np.dot(cv_emb, vac_emb)
            scores.append(similarity)
            vac_ids.append(vac_id)
        
        # Нормализация скоров
        sorted_indices = np.argsort(scores)[::-1]
        
        rankings = [0] * 5
        norm_scores = [0.0] * 5
        
        for position, idx in enumerate(sorted_indices, 1):
            vac_id = vac_ids[idx]
            rankings[vac_id - 1] = position
            norm_scores[vac_id - 1] = float(scores[idx] * 100)
        
        return RankingResult(
            cv_id=cv_id,
            rankings=rankings,
            scores=norm_scores,
            method='bert_rank'
        )


class EnsembleRanker:
    """
    Ансамблевое ранжирование 
    """
    
    def __init__(self):
        self.ranker = VacancyRanker()
        # СОЗДАЕМ ОДИН векторизатор для ВСЕХ вызовов VM метода
        from sklearn.feature_extraction.text import CountVectorizer
        self.vectorizer = CountVectorizer(
            analyzer='char',
            ngram_range=(1, 3),
            lowercase=True,
            max_features=2000  # ЖЕСТКО ФИКСИРУЕМ!
        )
        self.vectorizer_fitted = False
        self.optimal_weights = None
    
    def _get_vm_scores_fixed(self, cv_text: str, vacancies: Dict[int, Dict]) -> List[float]:
        """
        VM Method с ФИКСИРОВАННОЙ размерностью через общий векторизатор
        Возвращает скоры (чем меньше, тем лучше)
        """
        # 1. СОБИРАЕМ ВСЕ тексты для обучения векторизатора
        if not self.vectorizer_fitted:
            all_texts = [cv_text]
            for vac_data in vacancies.values():
                # Используем тот же метод суммаризации, что и в VM Method
                vac_summary = self.ranker.vm_method.summarize_text(vac_data['description'])
                all_texts.append(vac_summary)
            
            # ОБУЧАЕМ векторизатор на ВСЕХ текстах
            self.vectorizer.fit(all_texts)
            self.vectorizer_fitted = True
            print("✅ Ensemble: VM vectorizer trained")
        
        # 2. Вектор резюме - через ОБУЧЕННЫЙ векторизатор
        cv_vec = self.vectorizer.transform([cv_text]).toarray()[0]
        if np.linalg.norm(cv_vec) > 0:
            cv_vec = cv_vec / np.linalg.norm(cv_vec)
        
        # 3. Векторы вакансий и расстояния
        distances = []
        for vac_id, vac_data in vacancies.items():
            vac_summary = self.ranker.vm_method.summarize_text(vac_data['description'])
            vac_vec = self.vectorizer.transform([vac_summary]).toarray()[0]
            if np.linalg.norm(vac_vec) > 0:
                vac_vec = vac_vec / np.linalg.norm(vac_vec)
            
            # L1 расстояние - ТЕПЕРЬ ВЕКТОРЫ ОДИНАКОВОЙ ДЛИНЫ!
            distance = np.sum(np.abs(cv_vec - vac_vec))
            distances.append((vac_id, distance))
        
        # 4. Конвертируем расстояния в скоры (чем меньше расстояние, тем лучше)
        distances.sort(key=lambda x: x[1])
        scores = [0.0] * 5
        for position, (vac_id, dist) in enumerate(distances, 1):
            scores[vac_id - 1] = 100 / (1 + dist)
        
        return scores
    
    def rank_ensemble(
        self,
        cv_id: int,
        cv_text: str,
        cv_skills: List[str],
        vacancies: Dict[int, Dict],
        cv_experience: Optional[float] = None,
        weights: Dict[str, float] = None
    ) -> RankingResult:
        """
        Ансамбль методов - ИСПРАВЛЕННАЯ РАБОЧАЯ ВЕРСИЯ
        """
        if weights is None:
            if self.optimal_weights:
                weights = self.optimal_weights
            else:    
                weights = {
                    'unified': 0.5,
                    'vm_method': 0.3,
                    'bert_rank': 0.2
                }
   
        try:
            # 1. UNIFIED SCORING
            unified_result = self.ranker.rank_unified(
                cv_id=cv_id,
                cv_text=cv_text,
                cv_skills=cv_skills,
                vacancies=vacancies,
                cv_experience=cv_experience
            )
            
            # 2. VM METHOD - С ФИКСИРОВАННОЙ РАЗМЕРНОСТЬЮ
            vm_scores = self._get_vm_scores_fixed(cv_text, vacancies)
            
            # 3. BERT RANK
            bert_result = self.ranker.rank_bert(cv_id, cv_text, vacancies)
            
        except Exception as e:
            print(f"⚠️ Ensemble error: {e}, using unified only")
            import traceback
            traceback.print_exc()
            return self.ranker.rank_unified(
                cv_id=cv_id,
                cv_text=cv_text,
                cv_skills=cv_skills,
                vacancies=vacancies,
                cv_experience=cv_experience
            )
        
        # НОРМАЛИЗУЕМ скоры в диапазон 0-1
        def normalize_scores(scores):
            min_score = min(scores)
            max_score = max(scores)
            if max_score > min_score:
                return [(s - min_score) / (max_score - min_score) for s in scores]
            return [0.5] * len(scores)
        
        # Получаем скоры из результатов
        unified_scores = unified_result.scores
        vm_scores_norm = normalize_scores(vm_scores)
        bert_scores = bert_result.scores
        
        # ВЗВЕШЕННАЯ СУММА
        final_scores = [0.0] * 5
        for i in range(5):
            final_scores[i] = (
                weights['unified'] * (unified_scores[i] / 100) +
                weights['vm_method'] * vm_scores_norm[i] +
                weights['bert_rank'] * (bert_scores[i] / 100)
            )

        
        # КОНВЕРТИРУЕМ В РАНГИ
        sorted_indices = np.argsort(final_scores)[::-1]
        final_rankings = [0] * 5
        for pos, idx in enumerate(sorted_indices, 1):
            final_rankings[idx] = pos
        
        return RankingResult(
            cv_id=cv_id,
            rankings=final_rankings,
            scores=[s * 100 for s in final_scores],
            method="ensemble"
        )
    
    # РАНЖИРОВАНИЕ КАНДИДАТОВ
    def rank_candidates_for_vacancy(
        self,
        vacancy_id: int,
        vacancy_data: Dict,
        all_cvs: Dict[int, Dict],
        cv_experience: Optional[float] = None
    ) -> List[int]:
        """
        Ранжирование кандидатов для конкретной вакансии
        Возвращает список CV ID, отсортированных от лучшего к худшему
        """
        candidates_scores = []
        
        for cv_id, cv_data in all_cvs.items():
            # Используем Unified Scorer для расчета скора
            result = self.ranker.unified_scorer.calculate_score(
                cv_id=cv_id,
                cv_text=cv_data['text'],
                cv_skills=cv_data['skills'],
                vacancy_id=vacancy_id,
                vacancy_title=vacancy_data['title'],
                vacancy_text=vacancy_data['description'],
                vacancy_skills=vacancy_data['skills'],
                cv_experience=cv_data.get('experience'),
                vacancy_required_years=vacancy_data.get('required_years'),
                corpus=None
            )
            candidates_scores.append((cv_id, result.total_score))
        
        # Сортируем по убыванию скора
        candidates_scores.sort(key=lambda x: x[1], reverse=True)
        return [cv_id for cv_id, _ in candidates_scores]
    
    # МЕТОД - ДВУНАПРАВЛЕННЫЙ АНСАМБЛЬ
    def rank_bidirectional(
        self,
        cv_id: int,
        cv_data: Dict,
        vacancies: Dict[int, Dict],
        all_cvs: Dict[int, Dict],
        cv_experience: Optional[float] = None,
        weights: Dict[str, float] = None
    ) -> RankingResult:
        """
        Двунаправленное ранжирование:
        - 60% прямой скор (вакансия для кандидата)
        - 40% обратный скор (кандидат для вакансии)
        """
        if weights is None:
            weights = {'unified': 0.5, 'vm_method': 0.3, 'bert_rank': 0.2}
        
        # 1. ПРЯМОЕ РАНЖИРОВАНИЕ (обычный ensemble)
        forward_result = self.rank_ensemble(
            cv_id=cv_id,
            cv_text=cv_data['text'],
            cv_skills=cv_data['skills'],
            vacancies=vacancies,
            cv_experience=cv_experience,
            weights=weights
        )
        forward_scores = forward_result.scores
        
        # 2. ОБРАТНОЕ РАНЖИРОВАНИЕ (кандидат для каждой вакансии)
        backward_scores = [0.0] * 5
        
        for vac_id, vac_data in vacancies.items():
            # Ранжируем ВСЕХ кандидатов для этой вакансии
            candidates_rank = self.rank_candidates_for_vacancy(
                vac_id, vac_data, all_cvs, cv_experience
            )
            
            # Находим позицию нашего кандидата
            if cv_id in candidates_rank:
                position = candidates_rank.index(cv_id) + 1
                # Конвертируем позицию в скор (1 место = 100, 30 место = ~3)
                backward_scores[vac_id - 1] = 100 / position
        
        # 3. КОМБИНИРОВАННЫЙ СКОР (60% прямой + 40% обратный)
        final_scores = [0.0] * 5
        for i in range(5):
            final_scores[i] = 0.6 * forward_scores[i] + 0.4 * backward_scores[i]
        
        # 4. КОНВЕРТИРУЕМ В РАНГИ
        sorted_indices = np.argsort(final_scores)[::-1]
        final_rankings = [0] * 5
        for pos, idx in enumerate(sorted_indices, 1):
            final_rankings[idx] = pos
        
        return RankingResult(
            cv_id=cv_id,
            rankings=final_rankings,
            scores=final_scores,
            method="bidirectional_ensemble"
        )
    
    # МЕТОД - КОРРЕКЦИЯ СКОРОВ НА ОСНОВЕ КОНКУРЕНЦИИ 
    def rank_with_competition(
        self,
        cv_id: int,
        cv_data: Dict,
        vacancies: Dict[int, Dict],
        all_cvs: Dict[int, Dict],
        cv_experience: Optional[float] = None
    ) -> RankingResult:
        """
        Корректировка скоров с учетом конкуренции
        """
        # 1. Получаем базовые скоры
        base_result = self.rank_bidirectional(
            cv_id, cv_data, vacancies, all_cvs, cv_experience
        )
        base_scores = base_result.scores
        
        # 2. Корректируем с учетом конкуренции
        adjusted_scores = base_scores.copy()
        
        for vac_id, vac_data in vacancies.items():
            # Ранжируем всех кандидатов для этой вакансии
            candidates_rank = self.rank_candidates_for_vacancy(
                vac_id, vac_data, all_cvs, cv_experience
            )
            
            # Статистика по всем кандидатам
            all_candidates_scores = []
            for cand_id in all_cvs.keys():
                result = self.ranker.unified_scorer.calculate_score(
                    cv_id=cand_id,
                    cv_text=all_cvs[cand_id]['text'],
                    cv_skills=all_cvs[cand_id]['skills'],
                    vacancy_id=vac_id,
                    vacancy_title=vac_data['title'],
                    vacancy_text=vac_data['description'],
                    vacancy_skills=vac_data['skills'],
                    cv_experience=all_cvs[cand_id].get('experience'),
                    vacancy_required_years=vac_data.get('required_years'),
                    corpus=None
                )
                all_candidates_scores.append(result.total_score)
            
            # Вычисляем среднее и стандартное отклонение
            mean_score = np.mean(all_candidates_scores)
            std_score = np.std(all_candidates_scores)
            
            # Штраф за overqualification (слишком высокий скор)
            idx = vac_id - 1
            if base_scores[idx] > mean_score + 1.5 * std_score:
                adjusted_scores[idx] *= 0.85  # -15%
            
            # Бонус за высокий спрос (кандидат в топ-20%)
            if cv_id in candidates_rank[:int(len(all_cvs) * 0.2)]:
                adjusted_scores[idx] *= 1.1  # +10%
        
        # 3. КОНВЕРТИРУЕМ В РАНГИ
        sorted_indices = np.argsort(adjusted_scores)[::-1]
        final_rankings = [0] * 5
        for pos, idx in enumerate(sorted_indices, 1):
            final_rankings[idx] = pos
        
        return RankingResult(
            cv_id=cv_id,
            rankings=final_rankings,
            scores=adjusted_scores,
            method="competition_ensemble"
        )