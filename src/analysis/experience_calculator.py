"""
Модуль подсчета общего опыта работы
- Поддержка ВСЕХ форматов дат
- Поддержка месяцев
- Поддержка различных паттернов из реальных резюме
"""
import re
from datetime import datetime
from typing import List, Tuple, Optional


class ExperienceCalculator:
    """Калькулятор опыта работы из резюме - РАСШИРЕННАЯ ВЕРСИЯ"""
    
    def __init__(self):
        self.current_year = datetime.now().year
        self.current_month = datetime.now().month
        
        # Маппинг месяцев
        self.months_map = {
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }
    
    def calculate_total_experience(self, resume_text: str, verbose: bool = False) -> float:
        """
        Подсчет опыта - ИЩЕМ ТОЛЬКО В СЕКЦИИ EXPERIENCE!
        """
        # 1. СНАЧАЛА извлекаем ТОЛЬКО секцию опыта
        experience_section = self._extract_experience_section(resume_text)
        
        if not experience_section:
            if verbose:
                print("\n   ⚠️ Секция опыта не найдена")
            return self._extract_years_of_experience(resume_text)
        
        if verbose:
            print(f"\n   📋 Найдена секция опыта ({len(experience_section)} символов)")
        
        # 2. Ищем периоды ТОЛЬКО в секции опыта
    def calculate_total_experience(self, resume_text: str, verbose: bool = False) -> float:
        """
        Подсчет опыта - ИЩЕМ ТОЛЬКО В СЕКЦИИ EXPERIENCE!
        """
        # 1. СНАЧАЛА извлекаем ТОЛЬКО секцию опыта
        experience_section = self._extract_experience_section(resume_text)
        
        if not experience_section:
            if verbose:
                print("\n   ⚠️ Секция опыта не найдена")
            return self._extract_years_of_experience(resume_text)
        
        if verbose:
            print(f"\n   📋 Найдена секция опыта ({len(experience_section)} символов)")
        
    
        # 2. Ищем периоды ТОЛЬКО в секции опыта - РАСШИРЕННЫЕ ПАТТЕРНЫ!
        total_years = 0
        periods = []
        
        # ПАТТЕРН 1: YYYY - YYYY, YYYY - now
        pattern1 = r'(\d{4})\s*[-–—]\s*(\d{4}|now|present|current)'
        for match in re.finditer(pattern1, experience_section, re.IGNORECASE):
            start = int(match.group(1))
            end_str = match.group(2).lower()
            end = self.current_year if end_str in ['now', 'present', 'current'] else int(end_str)
            periods.append((start, end))
        
        # ПАТТЕРН 2: • July 2020 – now (С МАРКЕРОМ СПИСКА!)
        pattern2_bullet = r'(?:^|\n)[\s\•\*\-\d\.]*\s*\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{4})\s*[-–—]\s*(now|present|current)\b'
        for match in re.finditer(pattern2_bullet, experience_section, re.IGNORECASE):
            start = int(match.group(2))
            end = self.current_year
            periods.append((start, end))
            if verbose:
                print(f"      ✅ Найден период с маркером: {match.group(0).strip()}")
        
        # ПАТТЕРН 3: July 2020 - now (без маркера)
        pattern2 = r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{4})\s*[-–—]\s*(now|present|current)'
        for match in re.finditer(pattern2, experience_section, re.IGNORECASE):
            start = int(match.group(2))
            end = self.current_year
            periods.append((start, end))
        
        # ПАТТЕРН 4: Month YYYY - Month YYYY
        pattern3 = r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{4})\s*[-–—]\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{4})'
        for match in re.finditer(pattern3, experience_section, re.IGNORECASE):
            start = int(match.group(2))
            end = int(match.group(4))
            periods.append((start, end))
        
        # ПАТТЕРН 5: YYYY YYYY (без дефиса)
        pattern4 = r'(\d{4})\s+(\d{4})'
        for match in re.finditer(pattern4, experience_section, re.IGNORECASE):
            start = int(match.group(1))
            end = int(match.group(2))
            periods.append((start, end))
        
        # Удаляем дубликаты периодов
        periods = list(set(periods))
        periods.sort()
        
        if verbose:
            print(f"\n   📅 Найденные периоды: {periods}")
        
        # Объединяем периоды
        merged = []
        for start, end in periods:
            if not merged:
                merged.append([start, end])
            else:
                if start <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], end)
                else:
                    merged.append([start, end])
        
        # Суммируем
        for start, end in merged:
            total_years += (end - start)
        
        if verbose:
            print(f"   📅 Объединенные: {merged}")
            print(f"   ✅ Опыт: {total_years} лет")
        
        return float(total_years)        
            

    
    def _extract_experience_section(self, text: str) -> str:
        """
        Извлечение ТОЛЬКО секции опыта - ИЩЕМ ТОЧНЫЕ ЗАГОЛОВКИ!
        """
        lines = text.split('\n')
        experience_lines = []
        in_experience = False
        
        # ТОЧНЫЕ заголовки секции опыта (в нижнем регистре)
        experience_headers = {
            'experience',
            'work experience',
            'professional experience',
            'employment history',
            'work history',
        }
        
        # Заголовки других секций (где нужно остановиться)
        stop_headers = {
            'education',
            'skills',
            'projects',
            'certifications',
            'training',
            'languages',
            'additional',
            'publications',
            'courses',
            'certificates',
        }
        
        for i, line in enumerate(lines):
            line_clean = line.strip().lower()
            
            # Пропускаем пустые строки
            if not line_clean:
                continue
            
            # Поиск начала секции опыта
            if not in_experience:
                # Проверяем, является ли строка ТОЧНЫМ заголовком
                if line_clean in experience_headers:
                    in_experience = True
                    print(f"   ✅ Строка {i}: НАЙДЕН ЗАГОЛОВОК ОПЫТА! '{line.strip()}'")
                    continue
            
            # Поиск конца секции опыта
            if in_experience:
                # Проверяем, является ли строка заголовком другой секции
                if line_clean in stop_headers:
                    print(f"   🔚 Строка {i}: КОНЕЦ СЕКЦИИ (заголовок '{line.strip()}')")
                    break
                
                # Добавляем строку в секцию опыта
                if line_clean:
                    experience_lines.append(line)
        
        return '\n'.join(experience_lines)


    def _merge_periods_absolute(self, periods: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Объединение пересекающихся периодов в абсолютных месяцах"""
        if not periods:
            return []
        
        periods.sort(key=lambda x: x[0])
        merged = []
        
        current_start, current_end = periods[0]
        
        for start, end in periods[1:]:
            if start <= current_end + 1:  # Пересечение или смежный
                current_end = max(current_end, end)
            else:
                merged.append((current_start, current_end))
                current_start, current_end = start, end
        
        merged.append((current_start, current_end))
        return merged
    
    def _month_name(self, month_num: int) -> str:
        """Конвертация номера месяца в название"""
        month_num = max(1, min(12, month_num))
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        return months[month_num - 1]
    
    def _extract_years_of_experience(self, text: str) -> float:
        """Извлечение явно указанного опыта"""
        patterns = [
            r'(\d+)[\+]?\s*years?\s+of\s+experience',
            r'(\d+)[\+]?\s*years?\s+experience',
            r'experience\s+of\s+(\d+)[\+]?\s*years?',
            r'(\d+)[\+]?\s*years?',
            r'(\d+)[\+]?\s*yr?s?\s+exp',
            r'(\d{4})\s*[-–—]\s*(\d{4}|now|present|current)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Проверяем, что это не часть даты
                if not re.search(rf'{match.group(1)}\s*[-–—]', text):
                    return float(match.group(1))
        
        return 0.0