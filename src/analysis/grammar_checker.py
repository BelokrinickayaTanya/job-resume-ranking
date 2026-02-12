"""
Модуль проверки грамматики и орфографии
"""
import re
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

# Опциональный импорт LanguageTool
try:
    import language_tool_python
    LANGTOOL_AVAILABLE = True
except ImportError:
    language_tool_python = None
    LANGTOOL_AVAILABLE = False
    print("⚠️ language_tool_python not installed. Using basic grammar check only.")
    print("   Install with: pip install language-tool-python")

class GrammarChecker:
    """
    Проверка грамматики и орфографии в резюме
    
    Использует LanguageTool для расширенной проверки,
    с fallback на простые правила
    """
    
    def __init__(self, use_languagetool: bool = True):
        """
        Args:
            use_languagetool: Использовать LanguageTool (требует установки)
        """
        self.use_languagetool = use_languagetool and LANGTOOL_AVAILABLE
        self.lt = None
        
        if self.use_languagetool:
            try:
                self.lt = language_tool_python.LanguageTool('en-US')
                print("✅ LanguageTool loaded successfully")
            except Exception as e:
                print(f"⚠️ Failed to load LanguageTool: {e}")
                self.use_languagetool = False
    
    def check(self, text: str) -> Dict:
        """
        Полная проверка текста
        
        Returns:
            {
                'errors': [...],
                'warnings': [...],
                'suggestions': [...],
                'score': 0-100,
                'stats': {...}
            }
        """
        if self.use_languagetool and self.lt:
            return self._check_with_languagetool(text)
        else:
            return self._check_basic(text)
    
    def _check_with_languagetool(self, text: str) -> Dict:
        """Проверка с помощью LanguageTool"""
        matches = self.lt.check(text)
        
        errors = []
        warnings = []
        suggestions = []
        
        for match in matches[:50]:
            error = {
                'message': match.message,
                'replacements': [r.value for r in match.replacements[:3]],
                'offset': match.offset,
                'length': match.errorLength,
                'context': match.context,
                'category': match.category,
                'rule_id': match.ruleId
            }
            
            if match.ruleIssueType == 'error':
                errors.append(error)
            elif match.ruleIssueType == 'warning':
                warnings.append(error)
            else:
                suggestions.append(error)
        
        # Расчет скора (0-100, больше = лучше)
        total_issues = len(errors) + len(warnings) * 0.5 + len(suggestions) * 0.2
        text_length = len(text.split())
        
        if text_length == 0:
            score = 0
        else:
            score = max(0, 100 - (total_issues * 100 / text_length))
            score = min(100, round(score, 1))
        
        return {
            'errors': errors[:20],
            'warnings': warnings[:20],
            'suggestions': suggestions[:20],
            'score': score,
            'total_issues': len(matches),
            'stats': {
                'error_count': len(errors),
                'warning_count': len(warnings),
                'suggestion_count': len(suggestions)
            }
        }
    
    def _check_basic(self, text: str) -> Dict:
        """Базовая проверка без LanguageTool"""
        errors = []
        warnings = []
        suggestions = []
        
        # 1. Проверка на двойные пробелы
        if re.search(r'\s{2,}', text):
            warnings.append({
                'message': 'Multiple spaces detected',
                'category': 'whitespace',
                'severity': 'warning'
            })
        
        # 2. Проверка на повторяющиеся слова
        words = text.lower().split()
        for i in range(len(words) - 1):
            if words[i] == words[i + 1] and len(words[i]) > 1:
                warnings.append({
                    'message': f'Repeated word: "{words[i]}"',
                    'category': 'duplication',
                    'severity': 'warning'
                })
        
        # 3. Проверка на заглавные буквы после точки
        sentences = re.split(r'[.!?]+', text)
        for sent in sentences:
            sent = sent.strip()
            if sent and sent[0].islower():
                errors.append({
                    'message': 'Sentence should start with capital letter',
                    'category': 'capitalization',
                    'severity': 'error'
                })
                break
        
        # 4. Проверка на технические термины
        tech_terms = ['javascript', 'python', 'java', 'c++', 'c#', 'react', 'angular']
        for term in tech_terms:
            if term.lower() in text.lower():
                if term == 'javascript' and 'java script' in text.lower():
                    suggestions.append({
                        'message': 'Use "JavaScript" instead of "java script"',
                        'category': 'terminology',
                        'severity': 'suggestion'
                    })
        
        score = max(0, 100 - len(errors) * 5 - len(warnings) * 2 - len(suggestions))
        
        return {
            'errors': errors,
            'warnings': warnings,
            'suggestions': suggestions,
            'score': score,
            'total_issues': len(errors) + len(warnings) + len(suggestions),
            'stats': {
                'error_count': len(errors),
                'warning_count': len(warnings),
                'suggestion_count': len(suggestions)
            }
        }
    
    def check_resume_quality(self, text: str) -> Dict:
        """
        Специализированная проверка качества резюме
        """
        issues = []
        suggestions = []
        
        # 1. Длина резюме
        words = len(text.split())
        if words < 200:
            issues.append({
                'type': 'length',
                'message': 'Resume is too short (less than 200 words)',
                'severity': 'error'
            })
        elif words > 1500:
            suggestions.append({
                'type': 'length',
                'message': 'Resume is very long (over 1500 words). Consider condensing.',
                'severity': 'suggestion'
            })
        
        # 2. Наличие контактной информации
        has_email = bool(re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text))
        has_phone = bool(re.search(r'[\+]?[\d\s\-\(\)]{10,}', text))
        has_linkedin = 'linkedin' in text.lower()
        
        if not has_email:
            issues.append({
                'type': 'contact',
                'message': 'No email address found',
                'severity': 'error'
            })
        
        if not has_phone:
            suggestions.append({
                'type': 'contact',
                'message': 'No phone number found',
                'severity': 'suggestion'
            })
        
        if not has_linkedin:
            suggestions.append({
                'type': 'contact',
                'message': 'LinkedIn profile not found',
                'severity': 'suggestion'
            })
        
        # 3. Структура резюме
        required_sections = ['experience', 'education', 'skills']
        missing_sections = []
        
        for section in required_sections:
            if not re.search(rf'\b{section}\b', text.lower()):
                missing_sections.append(section)
        
        if missing_sections:
            issues.append({
                'type': 'structure',
                'message': f'Missing sections: {", ".join(missing_sections)}',
                'severity': 'error'
            })
        
        # 4. Достижения (количественные)
        achievements = re.findall(r'\b\d+\s*[%\+\-]', text)
        if len(achievements) < 2:
            suggestions.append({
                'type': 'achievements',
                'message': 'Few quantitative achievements. Try to add metrics.',
                'severity': 'suggestion'
            })
        
        return {
            'issues': issues,
            'suggestions': suggestions,
            'score': max(0, 100 - len(issues) * 10 - len(suggestions) * 3),
            'stats': {
                'word_count': words,
                'has_email': has_email,
                'has_phone': has_phone,
                'has_linkedin': has_linkedin,
                'missing_sections': missing_sections
            }
        }