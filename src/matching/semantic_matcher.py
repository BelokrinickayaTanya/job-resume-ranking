"""
Модуль семантического матчинга (5% веса в финальном скоре)
"""
from typing import Dict, List, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re


class SemanticMatcher:
    """
    Семантический матчинг с использованием sentence-transformers
    
    Вес в финальном скоре: 5%
    """
    
    # Доступные модели от легких к тяжелым
    AVAILABLE_MODELS = {
        'mini': 'all-MiniLM-L6-v2',        # Быстрая, 384-dim
        'base': 'all-mpnet-base-v2',        # Точная, 768-dim
        'large': 'all-roberta-large-v1',    # Очень точная, 1024-dim
        'msmarco': 'msmarco-distilbert-base-v4',  # Для IR задач
    }
    
    def __init__(self, model_name: str = 'mini', device: str = 'cpu'):
        """
        Args:
            model_name: Ключ из AVAILABLE_MODELS или путь к модели
            device: 'cpu' или 'cuda'
        """
        self.model_name = self.AVAILABLE_MODELS.get(model_name, model_name)
        self.device = device
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Загрузка модели sentence-transformers"""
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            print(f"✅ Загружена модель: {self.model_name}")
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            print(f"🔄 Пробуем загрузить fallback модель: all-MiniLM-L6-v2")
            self.model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
    
    def encode(self, text: str) -> np.ndarray:
        """Получение эмбеддинга текста"""
        if self.model is None:
            raise ValueError("Модель не загружена")
        
        # Очистка и подготовка текста
        text = self._prepare_text(text)
        
        # Получение эмбеддинга
        embedding = self.model.encode(text, normalize_embeddings=True)
        
        return embedding
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Пакетное получение эмбеддингов"""
        if self.model is None:
            raise ValueError("Модель не загружена")
        
        texts = [self._prepare_text(t) for t in texts]
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        
        return embeddings
    
    def calculate_similarity(
        self, 
        resume_text: str, 
        vacancy_text: str,
        use_chunks: bool = True
    ) -> Dict:
        """
        Расчет семантической схожести между резюме и вакансией
        
        Args:
            resume_text: Текст резюме
            vacancy_text: Текст вакансии
            use_chunks: Разбивать длинные тексты на чанки
        
        Returns:
            {
                'score': 0-100,
                'similarity': float,
                'method': str
            }
        """
        if use_chunks and len(resume_text) > 1000:
            # Для длинных резюме используем метод с чанками
            return self._calculate_similarity_chunked(resume_text, vacancy_text)
        else:
            # Для коротких текстов прямой метод
            return self._calculate_similarity_direct(resume_text, vacancy_text)
    
    def _calculate_similarity_direct(self, resume_text: str, vacancy_text: str) -> Dict:
        """Прямой расчет схожести полных текстов"""
        resume_emb = self.encode(resume_text)
        vacancy_emb = self.encode(vacancy_text)
        
        similarity = self._cosine_similarity(resume_emb, vacancy_emb)
        score = similarity * 100
        
        return {
            'score': round(score, 1),
            'similarity': round(similarity, 4),
            'method': 'direct'
        }
    
    def _calculate_similarity_chunked(self, resume_text: str, vacancy_text: str) -> Dict:
        """
        Расчет схожести с разбиением резюме на смысловые чанки
        
        Метод: максимальная схожесть среди всех чанков
        """
        # Разбиваем резюме на чанки (секции)
        chunks = self._split_into_chunks(resume_text)
        
        if not chunks:
            return self._calculate_similarity_direct(resume_text[:1000], vacancy_text)
        
        # Эмбеддинг вакансии
        vacancy_emb = self.encode(vacancy_text)
        
        # Эмбеддинги чанков
        chunk_embs = self.encode_batch(chunks)
        
        # Схожесть каждого чанка с вакансией
        similarities = []
        for chunk_emb in chunk_embs:
            sim = self._cosine_similarity(chunk_emb, vacancy_emb)
            similarities.append(sim)
        
        # Берем максимальную схожесть (релевантная секция)
        max_similarity = max(similarities) if similarities else 0
        # И среднюю (общий контекст)
        avg_similarity = np.mean(similarities) if similarities else 0
        
        # Комбинируем: 70% макс + 30% среднее
        similarity = max_similarity * 0.7 + avg_similarity * 0.3
        score = similarity * 100
        
        return {
            'score': round(score, 1),
            'similarity': round(similarity, 4),
            'max_similarity': round(max_similarity, 4),
            'avg_similarity': round(avg_similarity, 4),
            'method': 'chunked',
            'num_chunks': len(chunks)
        }
    
    def _prepare_text(self, text: str) -> str:
        """Подготовка текста для модели"""
        if not text:
            return ""
        
        # Ограничиваем длину (модели имеют лимит 512 токенов)
        # Примерно 2000 символов
        if len(text) > 2000:
            # Сохраняем начало и конец (где обычно ключевая информация)
            text = text[:1000] + " " + text[-1000:]
        
        # Удаляем лишние пробелы
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _split_into_chunks(self, text: str) -> List[str]:
        """Разбиение текста на смысловые чанки (по секциям)"""
        chunks = []
        
        # Ищем секции резюме
        section_patterns = [
            r'(?:^|\n)(?:experience|work history|employment)[^\n]*(?:\n)(.*?)(?=\n\s*\n|\n\s*[A-Z]|\Z)',
            r'(?:^|\n)(?:education|academic)[^\n]*(?:\n)(.*?)(?=\n\s*\n|\n\s*[A-Z]|\Z)',
            r'(?:^|\n)(?:skills|technologies|competencies)[^\n]*(?:\n)(.*?)(?=\n\s*\n|\n\s*[A-Z]|\Z)',
            r'(?:^|\n)(?:projects?|portfolio)[^\n]*(?:\n)(.*?)(?=\n\s*\n|\n\s*[A-Z]|\Z)',
            r'(?:^|\n)(?:certifications?|licenses)[^\n]*(?:\n)(.*?)(?=\n\s*\n|\n\s*[A-Z]|\Z)',
            r'(?:^|\n)(?:languages)[^\n]*(?:\n)(.*?)(?=\n\s*\n|\n\s*[A-Z]|\Z)'
        ]
        
        for pattern in section_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match and match.group(1).strip():
                chunk = match.group(1).strip()
                if len(chunk) > 50:  # Игнорируем слишком маленькие секции
                    chunks.append(chunk)
        
        # Если секции не найдены, разбиваем по абзацам
        if not chunks:
            paragraphs = text.split('\n\n')
            chunks = [p.strip() for p in paragraphs if len(p.strip()) > 100]
        
        # Ограничиваем количество чанков
        return chunks[:10]
    
    @staticmethod
    def _cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Косинусное сходство между эмбеддингами"""
        return float(np.dot(emb1, emb2))
    
    def find_most_similar_vacancy(self, resume_text: str, vacancy_texts: List[str]) -> Dict:
        """
        Поиск наиболее подходящей вакансии из списка
        
        Returns:
            {
                'best_index': int,
                'best_score': float,
                'scores': List[float]
            }
        """
        resume_emb = self.encode(resume_text)
        vacancy_embs = self.encode_batch(vacancy_texts)
        
        scores = []
        for vac_emb in vacancy_embs:
            sim = self._cosine_similarity(resume_emb, vac_emb)
            scores.append(sim * 100)
        
        best_idx = np.argmax(scores)
        
        return {
            'best_index': int(best_idx),
            'best_score': round(scores[best_idx], 1),
            'scores': [round(s, 1) for s in scores]
        }