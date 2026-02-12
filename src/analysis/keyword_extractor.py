"""
Модуль извлечения ключевых навыков с помощью KeyBERT
"""
from keybert import KeyBERT
from typing import List, Dict, Set
import re


class KeywordExtractor:
    """Извлечение технологий и навыков из текста"""
    
    # Технические навыки для приоритизации
    TECH_SKILLS = {
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php', 
        'go', 'rust', 'swift', 'kotlin', 'ruby', 'scala', 'perl',
        'react', 'angular', 'vue', 'nodejs', 'django', 'flask', 'spring',
        'sql', 'mysql', 'postgresql', 'mongodb', 'oracle', 'redis',
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins',
        'html', 'css', 'sass', 'less', 'webpack', 'babel',
        'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy',
        'git', 'github', 'gitlab', 'jira', 'confluence',
        'linux', 'unix', 'windows', 'bash', 'powershell',
        'rest', 'graphql', 'soap', 'grpc'
    }
    
    def __init__(self):
        self.kw_model = KeyBERT(model='all-MiniLM-L6-v2')
        self.skill_cache = {}
    
    def extract_keywords(self, text: str, top_n: int = 20) -> List[str]:
        """
        Извлечение ключевых слов с помощью KeyBERT
        """
        if not text or len(text) < 50:
            return []
        
        # Кэширование для повторных вызовов
        text_hash = hash(text[:500])
        if text_hash in self.skill_cache:
            return self.skill_cache[text_hash]
        
        try:
            keywords = self.kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 2),
                stop_words='english',
                top_n=top_n,
                diversity=0.7
            )
            
            # Извлекаем только тексты ключевых фраз
            extracted = [kw[0].lower() for kw in keywords]

            # ВАЖНО: Добавляем отдельные слова из ключевых фраз
            for kw in extracted:
                for word in kw.split():
                    if len(word) > 2 and word not in extracted:
                        extracted.append(word)
            
            # Добавляем прямое извлечение технических навыков
            tech_skills = self._extract_tech_skills(text)
            extracted.extend(tech_skills)
            
            # Удаляем дубликаты
            extracted = list(set(extracted))
            
            self.skill_cache[text_hash] = extracted
            return extracted
            
        except Exception as e:
            print(f"Ошибка KeyBERT: {e}")
            # Fallback на regex извлечение
            return self._extract_tech_skills(text)
        

    def _extract_tech_skills(self, text: str) -> List[str]:
        """Извлечение технических навыков - ТОЛЬКО точные слова!"""
        text_lower = text.lower()
        found_skills = set()
        
        # Разбиваем на слова
        #words = set(re.findall(r'\b[a-z0-9#+]+(?:[+\-.]?[a-z0-9]+)*\b', text_lower))
        
        for skill in self.TECH_SKILLS:
            # Точное совпадение слова
            if re.search(rf'\b{re.escape(skill)}\b', text_lower):
                found_skills.add(skill)
            elif skill in text_lower:
                pattern = rf'(^|\s|[,;])({re.escape(skill)})(\s|[,;]|$)'
                if re.search(pattern, text_lower):
                    found_skills.add(skill)    
            else:
                # Спецобработка для C++ и C#
                if skill == 'c++' and 'c++' in text_lower:
                    found_skills.add('c++')
                elif skill == 'c#' and 'c#' in text_lower:
                    found_skills.add('c#')
        
        # Исключаем ложные срабатывания
        false_positives = {'go', 'r', 'c'}
        for fp in false_positives:
            if fp in found_skills:
                # Проверяем, что это действительно отдельное слово
                if not re.search(rf'\b{fp}\b', text_lower):
                    found_skills.remove(fp)
        
        return list(found_skills)
    
    def calculate_total_experience(self, resume_text: str, verbose: bool = False) -> float:
        """
        Подсчет опыта
        """
        # Ищем все года подряд
        pattern = r'(\d{4})\s*[-–—]\s*(\d{4}|now|present|current)'
        
        total_years = 0
        periods = []
        
        for match in re.finditer(pattern, resume_text, re.IGNORECASE):
            start = int(match.group(1))
            end_str = match.group(2).lower()
            
            if end_str in ['now', 'present', 'current']:
                end = 2026
            else:
                end = int(end_str)
            
            # Исключаем образование
            context = resume_text[max(0, match.start()-50):min(len(resume_text), match.end()+50)].lower()
            if 'education' in context or 'university' in context:
                continue
                
            periods.append((start, end))
        
        # Объединяем периоды
        periods.sort()
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
            print(f"\n   📅 Периоды: {periods}")
            print(f"   📅 Объединенные: {merged}")
            print(f"   ✅ Опыт: {total_years} лет")
        
        return float(total_years)

    def _extract_tech_section(self, text: str) -> str:
        """Извлечение секции Technologies & Frameworks"""
        patterns = [
            r'(?:technologies\s*(?:&\s*)?frameworks?|tech\s*stack|tools\s*summary)[:\s]*(.+?)(?=\n\s*\n|\n\s*[A-Z]|\Z)',
            r'(?:skills|competencies)[:\s]*(.+?)(?=\n\s*\n|\n\s*[A-Z]|\Z)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1)
        return ""
    
    def _extract_by_category(self, text: str, category: str) -> List[str]:
        """Извлечение навыков по категориям """
        text_lower = text.lower()
        
        # Расширенные словари навыков
        category_map = {
            'programming': {
                'java', 'javascript', 'python', 'c++', 'c#', 'php', 'ruby', 
                'swift', 'kotlin', 'r-style', 'typescript', 'go', 'rust', 'scala'
            },
            'framework': {
                'spring', 'spring boot', 'springboot', 'spring mvc', 'spring security',
                'hibernate', 'jpa', 'jdbc', 'junit', 'bootstrap', 'bootstrap3', 
                'jquery', 'react', 'angular', 'vue', 'nodejs'
            },
            'database': {
                'mysql', 'mongodb', 'postgresql', 'oracle', 'redis', 'sql', 
                'nosql', 'mariadb', 'cassandra', 'elasticsearch'
            },
            'tool': {
                'git', 'github', 'maven', 'eclipse', 'intellij', 'postman', 
                'swagger', 'jira', 'confluence', 'jenkins', 'docker'
            }
        }
        
        found = set()
        
        for skill in category_map.get(category, set()):
            # Точное совпадение слова
            if re.search(rf'\b{re.escape(skill)}\b', text_lower):
                found.add(skill)
            # Спецобработка для составных названий
            elif ' ' in skill and skill in text_lower:
                found.add(skill)
            # Спецобработка для springboot
            elif skill == 'springboot' and 'spring boot' in text_lower:
                found.add('springboot')
        
        return list(found)
    
    def extract_skills_from_resume(self, resume_text: str) -> Dict[str, List[str]]:
        """
        Удобный метод для извлечения всех навыков из резюме
        """
        all_skills = self.extract_keywords(resume_text, top_n=40)
        
        return {
            'all_skills': all_skills,
            'programming_languages': self._extract_by_category(resume_text, 'programming'),
            'frameworks': self._extract_by_category(resume_text, 'framework'),
            'databases': self._extract_by_category(resume_text, 'database'),
            'tools': self._extract_by_category(resume_text, 'tool'),
            'cloud': self._extract_by_category(resume_text, 'cloud')
        }