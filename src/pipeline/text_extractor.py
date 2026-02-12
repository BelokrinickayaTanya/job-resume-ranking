"""
Модуль очистки и предобработки текста
"""
import re
from typing import List, Optional


class TextExtractor:
    """Очистка и нормализация текста"""
    
    @staticmethod
    def clean_text(text: str, remove_numbers: bool = True) -> str:
        """
        Очистка текста резюме/вакансии - СОХРАНЯЕМ СТРУКТУРУ!
        """
        if not text:
            return ""
        
        # Сохраняем переносы строк 
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Приводим к нижнему регистру
            line = line.lower()
            
            # Очищаем каждую строку отдельно
            if remove_numbers:
                # Сохраняем года
                years = re.findall(r'\b(19|20)\d{2}\b', line)
                for i, year in enumerate(years):
                    line = line.replace(year, f"__YEAR_{i}__")
                
                line = re.sub(r'\b\d+\b', ' ', line)
                
                for i, year in enumerate(years):
                    line = line.replace(f"__YEAR_{i}__", year)
            
            line = re.sub(r'[^\w\s\#\+\@\.\-]', ' ', line)
            line = re.sub(r'\s+', ' ', line).strip()
            cleaned_lines.append(line)
        
        # ВОССТАНАВЛИВАЕМ переносы строк!
        return '\n'.join(cleaned_lines)
    
    @staticmethod
    def extract_experience_periods(text: str) -> List[tuple]:
        """
        Извлечение периодов работы (года)
        Форматы: "2017 - 2020", "2020-now", "2021-present"
        """
        patterns = [
            r'(\b(19|20)\d{2})\s*[-–—]\s*(\b(19|20)\d{2}|now|present)',
            r'(\b(19|20)\d{2})\s*to\s*(\b(19|20)\d{2}|now|present)'
        ]
        
        periods = []
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                start = match[0]
                end = match[2]
                if end in ['now', 'present']:
                    end = '2026'
                periods.append((int(start), int(end)))
        
        return periods
    
    @staticmethod
    def extract_skills_section(text: str) -> str:
        """Извлечение секции с навыками"""
        # Ищем маркеры секции навыков
        skill_markers = [
            r'(?:skills|technologies|competencies|expertise)[:\s]*(.+?)(?:\n\s*\n|\n\s*[a-z]|\Z)',
            r'(?:technical|professional)\s+(?:skills?|experience)[:\s]*(.+?)(?:\n\s*\n|\Z)',
            r'(?:programming languages?|frameworks?|tools?)[:\s]*(.+?)(?:\n\s*\n|\Z)'
        ]
        
        for marker in skill_markers:
            match = re.search(marker, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        
        return ""