"""
Модуль определения языка текста
"""
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Для воспроизводимости
DetectorFactory.seed = 42


class LanguageDetector:
    """Определение языка текста"""
    
    SUPPORTED_LANGUAGES = ['en', 'ru']
    
    @classmethod
    def detect(cls, text: str) -> str:
        """
        Определение языка текста
        
        Returns:
            'en' - английский
            'ru' - русский
            'unknown' - не удалось определить
        """
        if not text or len(text.strip()) < 50:
            return 'unknown'
        
        try:
            # Берем первые 500 символов для скорости
            sample = text[:500].strip()
            lang = detect(sample)
            return lang if lang in cls.SUPPORTED_LANGUAGES else 'unknown'
        except LangDetectException:
            return 'unknown'
    
    @classmethod
    def is_english(cls, text: str) -> bool:
        """Проверка, является ли текст английским"""
        return cls.detect(text) == 'en'
    
    @classmethod
    def filter_english_cvs(cls, cv_dict: dict) -> dict:
        """Фильтрация только английских резюме"""
        english_cvs = {}
        for cv_id, text in cv_dict.items():
            if cls.is_english(text):
                english_cvs[cv_id] = text
            else:
                print(f"Резюме {cv_id} пропущено (не английский)")
        return english_cvs