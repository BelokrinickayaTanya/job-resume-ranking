"""
Модуль извлечения именованных сущностей (NER) с помощью SpaCy
Используется ТОЛЬКО для:
- Подсчета опыта работы (даты, периоды)
"""
import spacy
import re
from typing import List, Tuple
from datetime import datetime
from collections import defaultdict


class NamedEntityExtractor:
    """
    Извлечение периодов работы из текста с помощью SpaCy NER
    """
    
    def __init__(self, model_name: str = 'en_core_web_sm'):
        """
        Args:
            model_name: Модель SpaCy (en_core_web_sm)
        """
        try:
            self.nlp = spacy.load(model_name)
            print(f"✅ Загружена модель SpaCy: {model_name}")
        except OSError:
            print(f"⚠️ Модель {model_name} не найдена. Загружаем...")
            spacy.cli.download(model_name)
            self.nlp = spacy.load(model_name)
        
        self.current_year = datetime.now().year
    
    def extract_work_periods_ner(self, text: str) -> List[Tuple[int, int]]:
        """
        Извлечение периодов работы с помощью NER
        Возвращает: [(start_year, end_year)]
        Используется как ДОПОЛНЕНИЕ к regex методу
        """
        periods = []
        seen_periods = set()
        
        # Ограничиваем текст для производительности
        doc = self.nlp(text[:50000])
        
        # Собираем все DATE сущности
        dates = []
        for ent in doc.ents:
            if ent.label_ == 'DATE':
                # Извлекаем годы из DATE сущностей
                years = re.findall(r'\b(19|20)\d{2}\b', ent.text)
                if years:
                    dates.append({
                        'text': ent.text,
                        'year': int(years[0]),
                        'start_char': ent.start_char,
                        'end_char': ent.end_char,
                        'is_present': 'present' in ent.text.lower() or 'now' in ent.text.lower() or 'current' in ent.text.lower()
                    })
        
        # Ищем пары дат, формирующие период
        for i in range(len(dates) - 1):
            date1 = dates[i]
            date2 = dates[i + 1]
            
            # Проверяем, есть ли между ними тире (признак периода)
            between_text = text[date1['end_char']:date2['start_char']].strip()
            
            if between_text in ['-', '–', '—', 'to']:
                start_year = date1['year']
                
                if date2['is_present']:
                    end_year = self.current_year
                else:
                    end_year = date2['year']
                
                period_key = f"{start_year}-{end_year}"
                if period_key not in seen_periods and start_year < end_year:
                    seen_periods.add(period_key)
                    periods.append((start_year, end_year))
        
        # Ищем одиночные периоды с "present" (например: "2020 - present")
        for date in dates:
            if date['is_present']:
                # Ищем предыдущий текст с годом
                context_before = text[max(0, date['start_char']-30):date['start_char']]
                year_match = re.search(r'\b(19|20)\d{2}\b', context_before)
                
                if year_match:
                    start_year = int(year_match.group(0))
                    end_year = self.current_year
                    period_key = f"{start_year}-{end_year}"
                    
                    if period_key not in seen_periods and start_year < end_year:
                        seen_periods.add(period_key)
                        periods.append((start_year, end_year))
        
        return periods
    
    def get_entity_summary(self, text: str) -> dict:
        """
        Минимальная сводка - только количество дат
        """
        doc = self.nlp(text[:20000])
        
        date_count = 0
        for ent in doc.ents:
            if ent.label_ == 'DATE':
                date_count += 1
        
        return {
            'dates_found': date_count
        }