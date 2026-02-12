"""
Модуль Vector Matching (VM) метода из статьи

Job Vacancy Ranking with Sentence Embeddings, Keywords, and Named Entities
Vanetik & Kogan, Information 2023

Лучшая конфигурация из статьи:
- Resume: Full text
- Vacancy: BERT extractive summary (10 sentences)
- Text representation: Character n-grams (1-3)
- Distance: L1 (Manhattan)
- Krippendorff's Alpha: 0.6287
"""
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import cityblock
import re
from collections import defaultdict

# Опциональные импорты для BERT суммаризации
try:
    from summarizer import Summarizer
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    print("⚠️ BERT summarizer not installed. Using fallback summarization.")


class VMMethod:
    """
    Vector Matching Method из статьи
    
    Полная реализация с поддержкой всех вариантов:
    - Full texts
    - Extractive summaries (BERT)
    - Keyword-enhanced summaries
    - Named-entity-enhanced summaries
    - Word/Character n-grams
    - TF-IDF vectors
    - BERT sentence embeddings
    """
    
    def __init__(self, 
                 resume_text_type: str = 'full',
                 vacancy_text_type: str = 'summary',
                 representation: str = 'char_ngrams',
                 ngram_range: Tuple[int, int] = (1, 3),
                 summary_sentences: int = 10):
        """
        Args:
            resume_text_type: 'full', 'summary', 'kw_summary', 'ne_summary'
            vacancy_text_type: 'full', 'summary', 'kw_summary', 'ne_summary'
            representation: 'char_ngrams', 'word_ngrams', 'tfidf', 'sbert', 'all'
            ngram_range: Диапазон n-грамм
            summary_sentences: Количество предложений в саммари
        """
        self.resume_text_type = resume_text_type
        self.vacancy_text_type = vacancy_text_type
        self.representation = representation
        self.ngram_range = ngram_range
        self.summary_sentences = summary_sentences
        
        # Инициализация суммаризатора
        self.summarizer = None
        if BERT_AVAILABLE:
            try:
                self.summarizer = Summarizer(model='bert-base-uncased')
                print("✅ BERT summarizer loaded")
            except Exception as e:
                print(f"⚠️ Failed to load BERT summarizer: {e}")
        
        # Кэш для векторов
        self.vector_cache = {}
    
    def summarize_text(self, text: str) -> str:
        """
        BERT extractive summarization
        
        Returns:
            Summary with max self.summary_sentences sentences
        """
        if not text or len(text) < 100:
            return text
        
        cache_key = f"summary_{hash(text[:1000])}"
        if cache_key in self.vector_cache:
            return self.vector_cache[cache_key]
        
        if self.summarizer:
            try:
                # Ограничиваем длину для BERT (512 токенов)
                text_for_summary = text[:10000]
                summary = self.summarizer(
                    text_for_summary,
                    num_sentences=self.summary_sentences,
                    min_length=50
                )
                
                if summary and len(summary) > 20:
                    result = summary
                else:
                    result = self._fallback_summary(text)
            except Exception as e:
                print(f"⚠️ Summarization error: {e}")
                result = self._fallback_summary(text)
        else:
            result = self._fallback_summary(text)
        
        self.vector_cache[cache_key] = result
        return result
    
    def _fallback_summary(self, text: str) -> str:
        """Fallback метод суммаризации"""
        sentences = re.split(r'[.!?]+', text)
        
        # Берем первые 10 предложений
        if len(sentences) > self.summary_sentences:
            summary = '. '.join(sentences[:self.summary_sentences]) + '.'
        else:
            summary = text[:1000]  # Первые 1000 символов
        
        return summary
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """
        KeyBERT keyword extraction (упрощенная версия)
        """
        # Простое извлечение частотных терминов
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        freq = defaultdict(int)
        
        for word in words:
            if len(word) > 2 and word not in self._get_stopwords():
                freq[word] += 1
        
        # Топ-N по частоте
        keywords = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        return [kw for kw, _ in keywords]
    
    def _get_stopwords(self) -> set:
        """Стоп-слова для английского"""
        return {
            'the', 'and', 'for', 'with', 'this', 'that', 'from', 'have', 
            'will', 'your', 'you', 'are', 'can', 'was', 'has', 'not', 
            'but', 'all', 'any', 'experience', 'work', 'job', 'skills'
        }
    
    def extract_named_entities(self, text: str) -> List[str]:
        """
        Named Entity extraction (упрощенная версия)
        """
        entities = []
        
        # Извлекаем организации (с заглавной буквы)
        org_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        candidates = re.findall(org_pattern, text)
        
        # Фильтруем
        for cand in candidates[:20]:
            if len(cand) > 2 and cand.lower() not in self._get_stopwords():
                entities.append(cand)
        
        return list(set(entities))
    
    def prepare_text(self, 
                    text: str,
                    text_type: str,
                    keywords: Optional[List[str]] = None,
                    entities: Optional[List[str]] = None) -> str:
        """
        Подготовка текста согласно выбранному типу
        """
        if text_type == 'full':
            return text
        
        elif text_type == 'summary':
            return self.summarize_text(text)
        
        elif text_type == 'kw_summary':
            summary = self.summarize_text(text)
            if keywords is None:
                keywords = self.extract_keywords(text)
            keywords_text = ' '.join(keywords[:10])
            return f"{summary} {keywords_text}"
        
        elif text_type == 'ne_summary':
            summary = self.summarize_text(text)
            if entities is None:
                entities = self.extract_named_entities(text)
            entities_text = ' '.join(entities[:10])
            return f"{summary} {entities_text}"
        
        else:
            return text
    
    def get_vector(self, text: str) -> np.ndarray:
        """
        Получение векторного представления текста с ФИКСИРОВАННОЙ размерностью
        """
        cache_key = f"{self.representation}_{hash(text[:500])}"
        if cache_key in self.vector_cache:
            return self.vector_cache[cache_key]
        
        if self.representation == 'char_ngrams':
            # ✅ ФИКСИРУЕМ количество признаков!
            vectorizer = CountVectorizer(
                analyzer='char',
                ngram_range=self.ngram_range,
                lowercase=True,
                max_features=2000  # Фиксированный размер!
            )
            vec = vectorizer.fit_transform([text]).toarray()[0]
            
            # ✅ Дополняем или обрезаем до 2000
            if len(vec) < 2000:
                vec = np.pad(vec, (0, 2000 - len(vec)), 'constant')
            else:
                vec = vec[:2000]
                
        elif self.representation == 'word_ngrams':
            vectorizer = CountVectorizer(
                analyzer='word',
                ngram_range=self.ngram_range,
                lowercase=True,
                stop_words='english',
                max_features=2000
            )
            vec = vectorizer.fit_transform([text]).toarray()[0]
            if len(vec) < 2000:
                vec = np.pad(vec, (0, 2000 - len(vec)), 'constant')
            else:
                vec = vec[:2000]
                
        elif self.representation == 'tfidf':
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(
                max_features=2000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            vec = vectorizer.fit_transform([text]).toarray()[0]
            if len(vec) < 2000:
                vec = np.pad(vec, (0, 2000 - len(vec)), 'constant')
            else:
                vec = vec[:2000]
                
        elif self.representation == 'sbert':
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer('all-MiniLM-L6-v2')
                vec = model.encode(text[:1000])
                # SBERT уже имеет фиксированную размерность 384
            except:
                vec = np.zeros(384)
                
        elif self.representation == 'all':
            # Комбинированный вектор фиксированного размера
            vec_parts = []
            
            # Char n-grams - 500 признаков
            cv_char = CountVectorizer(analyzer='char', ngram_range=(1, 3), max_features=500)
            char_vec = cv_char.fit_transform([text]).toarray()[0]
            char_vec = np.pad(char_vec, (0, 500 - len(char_vec)), 'constant')[:500]
            vec_parts.append(char_vec)
            
            # Word n-grams - 500 признаков
            cv_word = CountVectorizer(analyzer='word', ngram_range=(1, 2), max_features=500)
            word_vec = cv_word.fit_transform([text]).toarray()[0]
            word_vec = np.pad(word_vec, (0, 500 - len(word_vec)), 'constant')[:500]
            vec_parts.append(word_vec)
            
            # TF-IDF - 500 признаков
            from sklearn.feature_extraction.text import TfidfVectorizer
            tv = TfidfVectorizer(max_features=500)
            tfidf_vec = tv.fit_transform([text]).toarray()[0]
            tfidf_vec = np.pad(tfidf_vec, (0, 500 - len(tfidf_vec)), 'constant')[:500]
            vec_parts.append(tfidf_vec)
            
            # SBERT - 384 признаков (обрезаем до 500?)
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer('all-MiniLM-L6-v2')
                sbert_vec = model.encode(text[:1000])
                sbert_vec = sbert_vec[:500] if len(sbert_vec) > 500 else np.pad(sbert_vec, (0, 500 - len(sbert_vec)), 'constant')
                vec_parts.append(sbert_vec[:500])
            except:
                vec_parts.append(np.zeros(500))
            
            vec = np.concatenate(vec_parts)
        
        else:
            # Default: char n-grams с фиксированным размером
            vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 3), max_features=2000)
            vec = vectorizer.fit_transform([text]).toarray()[0]
            if len(vec) < 2000:
                vec = np.pad(vec, (0, 2000 - len(vec)), 'constant')
            else:
                vec = vec[:2000]
        
        # Нормализация
        if np.linalg.norm(vec) > 0:
            vec = vec / np.linalg.norm(vec)
        
        self.vector_cache[cache_key] = vec
        return vec
    
    def calculate_distance(self, vec1: np.ndarray, vec2: np.ndarray, metric: str = 'l1') -> float:
        """
        Расчет расстояния между векторами
        """
        if metric == 'l1':
            return cityblock(vec1, vec2)
        elif metric == 'l2':
            return np.linalg.norm(vec1 - vec2)
        elif metric == 'cosine':
            if np.linalg.norm(vec1) > 0 and np.linalg.norm(vec2) > 0:
                return 1 - np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            else:
                return 1.0
        else:
            return cityblock(vec1, vec2)
    
    def rank_vacancies(self,
                      resume_text: str,
                      vacancies: Dict[int, Dict],
                      resume_keywords: Optional[List[str]] = None,
                      resume_entities: Optional[List[str]] = None) -> List[Tuple[int, float]]:
        """
        Ранжирование вакансий для одного резюме
        
        Returns:
            List of (vacancy_id, distance) sorted by distance (ascending)
        """
        # Подготовка текста резюме
        resume_processed = self.prepare_text(
            resume_text,
            self.resume_text_type,
            keywords=resume_keywords,
            entities=resume_entities
        )
        
        # Вектор резюме
        resume_vec = self.get_vector(resume_processed)
        
        distances = []
        
        for vac_id, vac_data in vacancies.items():
            # Подготовка текста вакансии
            vac_processed = self.prepare_text(
                vac_data['description'],
                self.vacancy_text_type,
                keywords=vac_data.get('skills'),
                entities=None
            )
            
            # Вектор вакансии
            vac_vec = self.get_vector(vac_processed)
            
            # Расстояние
            distance = self.calculate_distance(resume_vec, vac_vec, metric='l1')
            distances.append((vac_id, distance))
        
        # Сортировка по возрастанию расстояния
        distances.sort(key=lambda x: x[1])
        
        return distances
    
    def get_rankings(self, 
                    resume_text: str,
                    vacancies: Dict[int, Dict],
                    resume_keywords: Optional[List[str]] = None,
                    resume_entities: Optional[List[str]] = None) -> List[int]:
        """
        Получение массива рангов (1-5) для вакансий
        
        Returns:
            [rank_vac1, rank_vac2, rank_vac3, rank_vac4, rank_vac5]
        """
        distances = self.rank_vacancies(
            resume_text, 
            vacancies,
            resume_keywords,
            resume_entities
        )
        
        # Массив рангов
        rankings = [0] * 5
        
        for position, (vac_id, _) in enumerate(distances, 1):
            vacancy_index = vac_id - 1
            rankings[vacancy_index] = position
        
        return rankings

    def get_vector_fixed(self, text: str, max_features: int = 2000) -> np.ndarray:
        """Character n-grams с ФИКСИРОВАННОЙ размерностью 2000"""
        from sklearn.feature_extraction.text import CountVectorizer
        
        # ВСЕГДА используем max_features=2000
        vectorizer = CountVectorizer(
            analyzer='char',
            ngram_range=(1, 3),
            lowercase=True,
            max_features=max_features
        )
        
        try:
            vec = vectorizer.fit_transform([text]).toarray()[0]
            # ВСЕГДА длина 2000
            if len(vec) < max_features:
                vec = np.pad(vec, (0, max_features - len(vec)), 'constant')
            else:
                vec = vec[:max_features]
            
            if np.linalg.norm(vec) > 0:
                vec = vec / np.linalg.norm(vec)
            return vec
        except Exception as e:
            print(f"Vectorization error: {e}")
            return np.zeros(max_features)  
        
    def get_vector_ensemble(self, text: str, vectorizer=None) -> np.ndarray:
        """
        Получение вектора с ИСПОЛЬЗОВАНИЕМ ПЕРЕДАННОГО векторизатора
        Для Ensemble метода - чтобы все векторы были ОДИНАКОВОЙ размерности
        
        Args:
            text: Текст для векторизации
            vectorizer: Предобученный CountVectorizer (если None, вызывает get_vector_fixed)
        
        Returns:
            np.ndarray: Вектор фиксированной размерности
        """
        if vectorizer is None:
            return self.get_vector_fixed(text, max_features=2000)
        
        try:
            vec = vectorizer.transform([text]).toarray()[0]
            if np.linalg.norm(vec) > 0:
                vec = vec / np.linalg.norm(vec)
            return vec
        except Exception as e:
            print(f"Ensemble vectorization error: {e}")
            return np.zeros(vectorizer.max_features)    

class VMOptimizer:
    """
    Оптимизация параметров VM метода
    """
    
    def __init__(self, ground_truth: Dict[int, List[int]]):
        self.ground_truth = ground_truth
        self.best_config = None
        self.best_score = -1
    
    def grid_search(self,
                   resume_texts: Dict[int, str],
                   vacancies: Dict[int, Dict],
                   resume_keywords: Optional[Dict[int, List[str]]] = None):
        """
        Поиск по сетке параметров для воспроизведения результатов статьи
        """
        from ..evaluation.metrics import RankingMetrics
        
        metrics = RankingMetrics()
        
        # Параметры для поиска
        resume_types = ['full', 'summary', 'kw_summary', 'ne_summary']
        vacancy_types = ['full', 'summary', 'kw_summary', 'ne_summary']
        representations = ['char_ngrams', 'word_ngrams', 'tfidf', 'sbert', 'all']
        ngram_options = [(1, 2), (1, 3), (2, 3)]
        
        print("🔍 Grid search for optimal VM configuration...")
        print(f"Resume types: {resume_types}")
        print(f"Vacancy types: {vacancy_types}")
        print(f"Representations: {representations}")
        print(f"Total combinations: {len(resume_types) * len(vacancy_types) * len(representations) * len(ngram_options)}")
        
        results = []
        
        for rt in resume_types:
            for vt in vacancy_types:
                for rep in representations:
                    for ng in ngram_options:
                        print(f"\nTesting: R={rt}, V={vt}, Rep={rep}, Ngram={ng}")
                        
                        vm = VMMethod(
                            resume_text_type=rt,
                            vacancy_text_type=vt,
                            representation=rep,
                            ngram_range=ng
                        )
                        
                        predictions = {}
                        
                        for cv_id, cv_text in resume_texts.items():
                            if cv_id in self.ground_truth:
                                kw = resume_keywords.get(cv_id) if resume_keywords else None
                                rankings = vm.get_rankings(cv_text, vacancies, kw)
                                predictions[cv_id] = rankings
                        
                        # Оценка
                        alpha_values = []
                        for cv_id in predictions:
                            alpha = metrics.krippendorff_alpha([
                                predictions[cv_id],
                                self.ground_truth[cv_id]
                            ])
                            alpha_values.append(alpha)
                        
                        avg_alpha = np.mean(alpha_values)
                        
                        config = {
                            'resume_type': rt,
                            'vacancy_type': vt,
                            'representation': rep,
                            'ngram': ng,
                            'alpha': avg_alpha
                        }
                        
                        results.append(config)
                        
                        if avg_alpha > self.best_score:
                            self.best_score = avg_alpha
                            self.best_config = config
                            print(f"  🆕 New best: α={avg_alpha:.4f}")
                        else:
                            print(f"  α={avg_alpha:.4f}")
        
        print("\n" + "=" * 60)
        print("🏆 BEST CONFIGURATION")
        print("=" * 60)
        print(f"Resume type: {self.best_config['resume_type']}")
        print(f"Vacancy type: {self.best_config['vacancy_type']}")
        print(f"Representation: {self.best_config['representation']}")
        print(f"N-gram range: {self.best_config['ngram']}")
        print(f"Krippendorff's Alpha: {self.best_config['alpha']:.4f}")
        print("=" * 60)
        
        return results