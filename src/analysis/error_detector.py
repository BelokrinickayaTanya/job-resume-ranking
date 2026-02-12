"""
Модуль обнаружения ошибок и проблем в резюме
"""
import re
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from collections import defaultdict


class ErrorDetector:
    """
    Детектор проблем в резюме:
    - Несоответствие дат
    - Отсутствие обязательных полей
    - Противоречивая информация
    - Форматные ошибки
    """
    
    def __init__(self):
        self.current_year = datetime.now().year
    
    def detect_all(self, cv_text: str, cv_data: Optional[Dict] = None) -> Dict:
        """
        Полное детектирование проблем в резюме
        """
        errors = []
        warnings = []
        suggestions_list = []  # Переименовано 
        
        # 1. Проверка контактной информации
        contact_issues = self._check_contact_info(cv_text)
        errors.extend(contact_issues['errors'])
        warnings.extend(contact_issues['warnings'])
        suggestions_list.extend(contact_issues['suggestions'])
        
        # 2. Проверка дат и периодов работы
        date_issues = self._check_dates(cv_text)
        errors.extend(date_issues['errors'])
        warnings.extend(date_issues['warnings'])
        suggestions_list.extend(date_issues['suggestions'])
        
        # 3. Проверка образования
        edu_issues = self._check_education(cv_text)
        warnings.extend(edu_issues['warnings'])
        suggestions_list.extend(edu_issues['suggestions'])
        
        # 4. Проверка навыков
        skill_issues = self._check_skills(cv_text)
        suggestions_list.extend(skill_issues['suggestions'])
        
        # 5. Проверка на плагиат/шаблонность
        template_issues = self._check_template_usage(cv_text)
        warnings.extend(template_issues['warnings'])
        suggestions_list.extend(template_issues['suggestions'])
        
        # 6. Проверка форматирования
        format_issues = self._check_formatting(cv_text)
        suggestions_list.extend(format_issues['suggestions'])
        
        return {
            'errors': errors,
            'warnings': warnings,
            'suggestions': suggestions_list,  # Исправлено имя ключа
            'total_issues': len(errors) + len(warnings) + len(suggestions_list),
            'severity_counts': {
                'error': len(errors),
                'warning': len(warnings),
                'suggestion': len(suggestions_list)
            }
        }
    
    def _check_contact_info(self, text: str) -> Dict:
        """Проверка контактной информации"""
        errors = []
        warnings = []
        suggestions = []
        
        # Email
        email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
        emails = re.findall(email_pattern, text)
        if not emails:
            errors.append({
                'type': 'missing_email',
                'message': 'No email address found in resume',
                'severity': 'error'
            })
        
        # Phone
        phone_pattern = r'[\+]?[\d\s\-\(\)]{10,}'
        phones = re.findall(phone_pattern, text)
        if not phones:
            warnings.append({
                'type': 'missing_phone',
                'message': 'No phone number found',
                'severity': 'warning'
            })
        
        # LinkedIn
        if 'linkedin' not in text.lower() and 'linked in' not in text.lower():
            warnings.append({
                'type': 'missing_linkedin',
                'message': 'LinkedIn profile not mentioned',
                'severity': 'warning'
            })
        
        # Location
        location_patterns = [
            r'\b(?:located|based|residence|city)[\s:]+\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
            r'\b(?:Herzliya|Tel Aviv|Jerusalem|Haifa|Beer Sheva|Raanana|Petah Tikva)\b'
        ]
        
        location_found = False
        for pattern in location_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                location_found = True
                break
        
        if not location_found:
            suggestions.append({
                'type': 'missing_location',
                'message': 'Location not specified',
                'severity': 'suggestion'
            })
        
        return {
            'errors': errors,
            'warnings': warnings,
            'suggestions': suggestions
        }
    
    def _check_dates(self, text: str) -> Dict:
        """Проверка дат и периодов работы"""
        errors = []
        warnings = []
        suggestions = []
        
        # Ищем все года
        years = re.findall(r'\b(19|20)\d{2}\b', text)
        years = [int(y) for y in years]
        
        if years:
            # Проверка на будущие даты
            future_years = [y for y in years if y > self.current_year]
            if future_years:
                errors.append({
                    'type': 'future_date',
                    'message': f'Future dates found: {future_years}',
                    'severity': 'error'
                })
            
            # Проверка на слишком старые даты (>20 лет назад)
            old_years = [y for y in years if y < self.current_year - 20 and y > 1970]
            if old_years:
                warnings.append({
                    'type': 'old_date',
                    'message': f'Very old dates found: {old_years}',
                    'severity': 'warning'
                })
        
        # Проверка периодов работы
        periods = re.findall(r'(\b\d{4}\b)\s*[-–—]\s*(\b\d{4}\b|\bpresent\b|\bnow\b)', text, re.IGNORECASE)
        
        for start, end in periods:
            if end.lower() not in ['present', 'now']:
                end_year = int(end)
                start_year = int(start)
                
                # Длительность периода
                duration = end_year - start_year
                if duration > 10:
                    warnings.append({
                        'type': 'long_period',
                        'message': f'Very long work period: {start}-{end} ({duration} years)',
                        'severity': 'warning'
                    })
                elif duration < 0:
                    errors.append({
                        'type': 'date_order',
                        'message': f'End date before start date: {start}-{end}',
                        'severity': 'error'
                    })
        
        return {
            'errors': errors,
            'warnings': warnings,
            'suggestions': suggestions
        }
    
    def _check_education(self, text: str) -> Dict:
        """Проверка образования"""
        warnings = []
        suggestions = []
        
        # Проверка наличия образования
        edu_keywords = ['university', 'college', 'institute', 'bachelor', 'master', 'phd', 
                       'bs', 'ms', 'ba', 'ma', 'degree']
        
        has_education = False
        for keyword in edu_keywords:
            if re.search(rf'\b{keyword}\b', text, re.IGNORECASE):
                has_education = True
                break
        
        if not has_education:
            warnings.append({
                'type': 'missing_education',
                'message': 'No education section found',
                'severity': 'warning'
            })
        
        # Проверка на incomplete degree
        if re.search(r'\b(?:bachelor|master|phd|bs|ms|ba|ma)[\'\"]?s?\s+in', text, re.IGNORECASE):
            # Проверяем, есть ли год окончания
            degree_years = re.findall(r'(bachelor|master|phd).*?(\d{4})', text, re.IGNORECASE | re.DOTALL)
            if not degree_years:
                suggestions.append({
                    'type': 'graduation_year',
                    'message': 'Consider adding graduation years for degrees',
                    'severity': 'suggestion'
                })
        
        return {
            'errors': [],
            'warnings': warnings,
            'suggestions': suggestions
        }
    
    def _check_skills(self, text: str) -> Dict:
        """Проверка навыков"""
        suggestions = []
        
        # Проверка на outdated технологии
        outdated_tech = ['flash', 'silverlight', 'vb6', 'cobol', 'fortran', 
                        'jquery 1', 'ie6', 'ie7', 'ie8']
        
        for tech in outdated_tech:
            if tech in text.lower():
                suggestions.append({
                    'type': 'outdated_tech',
                    'message': f'Consider removing outdated technology: {tech}',
                    'severity': 'suggestion'
                })
        
        # Проверка на современные технологии (рекомендации)
        modern_tech = ['react', 'angular', 'vue', 'python', 'aws', 'docker', 
                      'kubernetes', 'typescript', 'nodejs', 'graphql']
        
        has_modern = False
        for tech in modern_tech[:5]:
            if tech in text.lower():
                has_modern = True
                break
        
        if not has_modern:
            suggestions.append({
                'type': 'modern_tech',
                'message': 'Consider adding modern technologies to stay competitive',
                'severity': 'suggestion'
            })
        
        return {
            'errors': [],
            'warnings': [],
            'suggestions': suggestions
        }
    
    def _check_template_usage(self, text: str) -> Dict:
        """Проверка на использование стандартных шаблонов"""
        warnings = []
        suggestions = []
        
        # Признаки шаблонного резюме
        template_phrases = [
            'to whom it may concern',
            'i am writing to apply',
            'i have attached my resume',
            'dear hiring manager',
            'i am confident that my skills',
            'i would welcome the opportunity'
        ]
        
        template_count = 0
        for phrase in template_phrases:
            if phrase in text.lower():
                template_count += 1
        
        if template_count >= 3:
            warnings.append({
                'type': 'template_resume',
                'message': 'Resume appears to be heavily templated. Consider personalizing.',
                'severity': 'warning'
            })
        
        return {
            'errors': [],
            'warnings': warnings,
            'suggestions': suggestions
        }
    
    def _check_formatting(self, text: str) -> Dict:
        """Проверка форматирования"""
        suggestions = []
        
        # Длинные параграфы
        paragraphs = text.split('\n\n')
        for i, para in enumerate(paragraphs):
            if len(para.split()) > 100:
                suggestions.append({
                    'type': 'long_paragraph',
                    'message': f'Paragraph {i+1} is very long. Consider breaking into bullet points.',
                    'severity': 'suggestion'
                })
        
        # Отсутствие буллит-поинтов
        bullet_indicators = ['•', '-', '*', '·']
        has_bullets = any(indicator in text for indicator in bullet_indicators)
        
        if not has_bullets:
            suggestions.append({
                'type': 'no_bullets',
                'message': 'No bullet points found. Use bullet points for better readability.',
                'severity': 'suggestion'
            })
        
        return {
            'errors': [],
            'warnings': [],
            'suggestions': suggestions[:5]
        }
    
    def check_consistency(self, cv_data: Dict) -> Dict:
        """
        Проверка согласованности данных в резюме
        """
        inconsistencies = []
        
        # Проверка: опыт vs навыки
        experience = cv_data.get('experience', 0)
        skills = cv_data.get('skills', [])
        
        if experience > 10 and len([s for s in skills if 'python' in s]) == 0:
            inconsistencies.append({
                'type': 'skill_experience_mismatch',
                'message': f'Senior candidate ({experience} years) missing core skills',
                'severity': 'warning'
            })
        
        return {
            'inconsistencies': inconsistencies,
            'count': len(inconsistencies)
        }