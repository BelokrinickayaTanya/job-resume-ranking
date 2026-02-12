"""
Модуль сопоставления ключевых навыков (80% веса в финальном скоре)
"""
from typing import Dict, List, Set, Tuple, Optional
import re
from difflib import SequenceMatcher
from collections import defaultdict


class KeywordMatcher:
    """
    Матчинг навыков с синонимами и fuzzy matching
    
    Вес в финальном скоре: 80%
    """
    
    # Расширенный словарь синонимов технологий
    SYNONYMS = {
        # Языки программирования
        'python': ['python', 'python3', 'py', 'cpython', 'anaconda'],
        'javascript': ['javascript', 'js', 'ecmascript', 'nodejs', 'node', 'deno'],
        'typescript': ['typescript', 'ts', 'typed javascript'],
        'java': ['java', 'j2ee', 'jdk', 'jre', 'java ee', 'jakarta ee'],
        'c++': ['c++', 'cpp', 'cplusplus', 'c plus plus', 'stl', 'boost'],
        'c#': ['c#', 'csharp', 'c sharp', 'dotnet', '.net', 'net core', 'asp.net'],
        'php': ['php', 'php7', 'php8', 'laravel', 'symfony', 'composer'],
        'go': ['go', 'golang', 'go lang'],
        'rust': ['rust', 'rustlang'],
        'ruby': ['ruby', 'rb', 'ruby on rails', 'ror'],
        'swift': ['swift', 'ios development'],
        'kotlin': ['kotlin', 'android development'],
        'scala': ['scala', 'akka'],
        'perl': ['perl', 'cgi'],
        'r': ['r', 'r language', 'rstudio'],
        'matlab': ['matlab', 'simulink'],
        
        # Фронтенд фреймворки
        'react': ['react', 'reactjs', 'react.js', 'react native', 'next.js', 'gatsby'],
        'angular': ['angular', 'angularjs', 'angular.js', 'angular 2+'],
        'vue': ['vue', 'vuejs', 'vue.js', 'nuxt', 'vite'],
        'svelte': ['svelte', 'sveltekit'],
        'jquery': ['jquery', '$', 'jquery ui'],
        'bootstrap': ['bootstrap', 'bootstrap4', 'bootstrap5', 'twitter bootstrap'],
        'tailwind': ['tailwind', 'tailwindcss', 'tailwind css'],
        'material ui': ['material ui', 'mui', 'material design'],
        
        # Бэкенд фреймворки
        'django': ['django', 'django rest', 'drf'],
        'flask': ['flask', 'flask restful'],
        'fastapi': ['fastapi', 'fast api'],
        'spring': ['spring', 'spring boot', 'spring mvc', 'spring framework'],
        'express': ['express', 'expressjs', 'express.js', 'nodejs express'],
        'asp.net': ['asp.net', 'asp.net core', 'asp', '.net mvc'],
        
        # Базы данных
        'sql': ['sql', 'rdbms', 'relational database'],
        'mysql': ['mysql', 'mariadb', 'percona'],
        'postgresql': ['postgresql', 'postgres', 'pgsql'],
        'mongodb': ['mongodb', 'mongo', 'nosql', 'document database'],
        'redis': ['redis', 'key-value store', 'cache'],
        'cassandra': ['cassandra', 'cql', 'wide column'],
        'elasticsearch': ['elasticsearch', 'es', 'elk', 'elastic stack'],
        'oracle': ['oracle', 'oracle db', 'pl/sql'],
        'sqlite': ['sqlite', 'lite database'],
        'dynamodb': ['dynamodb', 'aws dynamodb'],
        'firebase': ['firebase', 'firestore', 'realtime database'],
        
        # Cloud платформы
        'aws': ['aws', 'amazon web services', 'ec2', 's3', 'lambda', 'rds', 'cloudfront', 
                'route53', 'vpc', 'iam', 'dynamodb', 'ecs', 'eks', 'fargate'],
        'azure': ['azure', 'microsoft azure', 'azure devops', 'aad', 'blob storage', 
                  'azure functions', 'app service', 'sql azure'],
        'gcp': ['gcp', 'google cloud', 'google cloud platform', 'compute engine', 
                'cloud storage', 'bigquery', 'pub/sub', 'kubernetes engine'],
        'heroku': ['heroku', 'heroku platform'],
        'digitalocean': ['digitalocean', 'do', 'droplet'],
        
        # DevOps и контейнеризация
        'docker': ['docker', 'container', 'docker compose', 'dockerfile'],
        'kubernetes': ['kubernetes', 'k8s', 'kube', 'openshift'],
        'jenkins': ['jenkins', 'jenkins ci', 'jenkins pipeline'],
        'gitlab ci': ['gitlab ci', 'gitlab pipeline', 'gitlab runner'],
        'github actions': ['github actions', 'gha', 'actions'],
        'terraform': ['terraform', 'iac', 'infrastructure as code'],
        'ansible': ['ansible', 'ansible playbook'],
        'chef': ['chef', 'chef cookbook'],
        'puppet': ['puppet', 'puppet manifest'],
        'prometheus': ['prometheus', 'prom', 'monitoring'],
        'grafana': ['grafana', 'dashboard', 'visualization'],
        
        # Системы контроля версий
        'git': ['git', 'github', 'gitlab', 'bitbucket', 'version control', 'vcs'],
        'svn': ['svn', 'subversion', 'apache subversion'],
        'mercurial': ['mercurial', 'hg'],
        
        # Операционные системы
        'linux': ['linux', 'unix', 'ubuntu', 'debian', 'centos', 'redhat', 'fedora', 
                  'arch', 'suse', 'opensuse', 'bash', 'shell', 'command line'],
        'windows': ['windows', 'win32', 'winapi', 'mfc', 'dotnet', 'powershell'],
        'macos': ['macos', 'os x', 'mac os x', 'darwin'],
        
        # Тестирование
        'junit': ['junit', 'unit testing', 'test driven', 'tdd'],
        'pytest': ['pytest', 'python testing'],
        'selenium': ['selenium', 'webdriver', 'automation testing'],
        'cypress': ['cypress', 'e2e testing'],
        'jest': ['jest', 'javascript testing', 'react testing'],
        'mocha': ['mocha', 'chai', 'sinon'],
        
        # Очереди и сообщения
        'rabbitmq': ['rabbitmq', 'message queue', 'amqp'],
        'kafka': ['kafka', 'apache kafka', 'pub sub'],
        'activemq': ['activemq', 'jms'],
        'sqs': ['sqs', 'amazon sqs', 'simple queue service'],
        
        # API и интеграция
        'rest': ['rest', 'restful', 'rest api', 'restful api', 'rest webservice'],
        'graphql': ['graphql', 'gql', 'apollo', 'relay'],
        'grpc': ['grpc', 'protocol buffers', 'protobuf'],
        'soap': ['soap', 'soap webservice', 'wsdl'],
        
        # Методологии
        'agile': ['agile', 'scrum', 'kanban', 'sprint', 'standup', 'retrospective'],
        'waterfall': ['waterfall', 'traditional development'],
        
        # Аналитика и ML
        'pandas': ['pandas', 'dataframe', 'data analysis'],
        'numpy': ['numpy', 'numerical python', 'array'],
        'scikit-learn': ['scikit-learn', 'sklearn', 'machine learning'],
        'tensorflow': ['tensorflow', 'tf', 'keras'],
        'pytorch': ['pytorch', 'torch'],
        'jupyter': ['jupyter', 'ipython', 'notebook'],
        'tableau': ['tableau', 'data visualization'],
        'power bi': ['power bi', 'powerbi', 'microsoft bi'],
        
        # Мобильная разработка
        'android': ['android', 'android sdk', 'android studio', 'dalvik'],
        'ios': ['ios', 'iphone', 'ipad', 'apple', 'cocoa touch'],
        'react native': ['react native', 'rn', 'cross platform mobile'],
        'flutter': ['flutter', 'dart'],
        'xamarin': ['xamarin', 'xamarin forms', 'mono'],
        
        # ERP/CRM
        'salesforce': ['salesforce', 'sfdc', 'apex', 'soql'],
        'sap': ['sap', 'abap', 'sap hana'],
        'oracle erp': ['oracle erp', 'e-business suite', 'ebs'],
        
        # Системы трекинга
        'jira': ['jira', 'atlassian', 'issue tracking'],
        'confluence': ['confluence', 'wiki', 'documentation'],
        'trello': ['trello', 'kanban board'],
        'asana': ['asana', 'project management'],
        
        # Коммуникации
        'slack': ['slack', 'chatops'],
        'teams': ['teams', 'microsoft teams'],
        
        # Игры
        'unity': ['unity', 'unity3d', 'game engine'],
        'unreal': ['unreal', 'unreal engine', 'ue4', 'ue5'],
    }
    
    # Список стоп-слов для фильтрации
    STOP_WORDS = {
        'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'until',
        'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
        'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
        'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
        'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
        'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
        'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
        'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should',
        'now', 'experience', 'skill', 'knowledge', 'ability', 'proficient'
    }
    
    def __init__(self, fuzzy_threshold: float = 0.85):
        """
        Args:
            fuzzy_threshold: Порог схожести для fuzzy matching (0.0-1.0)
        """
        self.fuzzy_threshold = fuzzy_threshold
        self.skill_cache = {}
        self.synonym_cache = {}
        
        # Инвертированный индекс синонимов для быстрого поиска
        self._build_synonym_index()
    
    def _extract_soft_skills(self, text: str) -> List[str]:
        """
        Извлечение soft skills из текста вакансии
        """
        soft_skills = {
            'remote', 'work from home', 'wfh', 'distributed team', 
            'async', 'asynchronous', 'communication', 'teamwork', 
            'collaboration', 'leadership', 'problem solving',
            'critical thinking', 'time management', 'agile', 'scrum',
            'self-motivated', 'independent', 'fast learner'
        }
        
        text_lower = text.lower()
        found = set()
        
        for skill in soft_skills:
            if skill in text_lower:
                found.add(skill)
        
        return list(found)

    def _build_synonym_index(self):
        """Построение инвертированного индекса синонимов"""
        self.synonym_to_canonical = {}
        for canonical, synonyms in self.SYNONYMS.items():
            for synonym in synonyms:
                self.synonym_to_canonical[synonym.lower()] = canonical
            # Добавляем сам канонический термин
            self.synonym_to_canonical[canonical.lower()] = canonical
    
    def calculate_match_score(
        self, 
        resume_skills: List[str], 
        vacancy_skills: List[str],
        vacancy_text: Optional[str] = None,
        cv_experience: Optional[float] = None, 
        vacancy_required_years: Optional[int] = None
    ) -> Dict:
        """
        Расчет скора матчинга навыков с учетом весов и важности
        
        Args:
            resume_skills: Список навыков из резюме
            vacancy_skills: Список требуемых навыков из вакансии
            vacancy_text: Полный текст вакансии (для извлечения весов)
            
        Returns:
            {
                'score': 0-100,
                'matched': [...],
                'missing': [...],
                'partial': [...],
                'weights': {...},
                'match_details': {...}
            }
        """
        if not vacancy_skills:
            return {
                'score': 0,
                'matched': [],
                'missing': [],
                'partial': [],
                'weights': {},
                'match_details': {}
            }
        
        # 1. Нормализуем навыки
        resume_norm = self._normalize_skills(resume_skills)
        vacancy_norm = self._normalize_skills(vacancy_skills)
        soft_skills = self._extract_soft_skills(vacancy_text) if vacancy_text else []
        vacancy_norm.extend(soft_skills)
        
        # 2. Определяем веса навыков (если есть текст вакансии)
        skill_weights = self._calculate_skill_weights(vacancy_text, vacancy_norm)
        
        # 3. Точные совпадения
        exact_matches = self._find_exact_matches(resume_norm, vacancy_norm)
        
        # 4. Синонимичные совпадения
        synonym_matches = self._find_synonym_matches(resume_norm, vacancy_norm)
        
        # 5. Fuzzy совпадения
        fuzzy_matches = self._find_fuzzy_matches(resume_norm, vacancy_norm)
        
        # 6. Частичные совпадения (вхождения)
        partial_matches = self._find_partial_matches(resume_skills, vacancy_norm)
        
        # Все совпадения с весами
        all_matches = {}
        match_types = {}
        
        for skill in exact_matches:
            all_matches[skill] = skill_weights.get(skill, 1.0)
            match_types[skill] = 'exact'
            
        for skill in synonym_matches:
            if skill not in all_matches:
                all_matches[skill] = skill_weights.get(skill, 0.9)  # Синоним = 90% веса
                match_types[skill] = 'synonym'
                
        for skill in fuzzy_matches:
            if skill not in all_matches:
                all_matches[skill] = skill_weights.get(skill, 0.8)  # Fuzzy = 80% веса
                match_types[skill] = 'fuzzy'
                
        for skill in partial_matches:
            if skill not in all_matches:
                all_matches[skill] = skill_weights.get(skill, 0.7)  # Частичное = 70% веса
                match_types[skill] = 'partial'
        
        # Отсутствующие навыки
        missing = set(vacancy_norm) - set(all_matches.keys())
        
        # Расчет скора с весами
        total_weight = sum(skill_weights.get(skill, 1.0) for skill in vacancy_norm)
        matched_weight = sum(all_matches.get(skill, 0) for skill in all_matches)
        
        if total_weight > 0:
            score = (matched_weight / total_weight) * 100
        else:
            score = 0
        
        # Бонус за покрытие всех must-have навыков
        must_have_bonus = self._calculate_must_have_bonus(vacancy_text, all_matches)
        score = min(100, score + must_have_bonus)
        
        # Штраф за отсутствие критических навыков
        critical_penalty = self._calculate_critical_penalty(vacancy_text, missing)
        score = max(0, score - critical_penalty)
        
        score = round(score, 1)
        # 1. БУСТ ДЛЯ REMOTE ВАКАНСИЙ
        if vacancy_text and ('remote' in vacancy_text.lower() or 'work from home' in vacancy_text.lower()):
            score += 10  # +10% буст для удаленной работы
            # Дополнительный буст, если у кандидата есть локация (не важно какая)
            score += 5   # +5% буст за наличие локации
            score = min(100, score)  # Не превышаем 100%

        # 2. УЧЕТ ПЕРЕКВАЛИФИКАЦИИ
        if cv_experience is not None and vacancy_required_years is not None:
            if cv_experience > vacancy_required_years * 2:
                score -= 5  # Штраф за overqualification
            elif cv_experience >= vacancy_required_years:
                score += 5  # Бонус за соответствие опыта
            score = max(0, min(100, score))  # Ограничиваем 0-100

        return {
            'score': score,
            'matched': list(all_matches.keys()),
            'missing': list(missing),
            'partial': list(partial_matches),
            'weights': skill_weights,
            'match_details': {
                'exact_matches': list(exact_matches),
                'synonym_matches': list(synonym_matches),
                'fuzzy_matches': list(fuzzy_matches),
                'match_types': match_types,
                'match_scores': all_matches
            }
        }
    
    def _normalize_skills(self, skills: List[str]) -> List[str]:
        """Нормализация списка навыков"""
        normalized = []
        for skill in skills:
            if isinstance(skill, str):
                # Приводим к нижнему регистру
                skill = skill.lower().strip()
                
                # Удаляем стоп-слова
                words = skill.split()
                filtered = [w for w in words if w not in self.STOP_WORDS]
                if filtered:
                    skill = ' '.join(filtered)
                
                normalized.append(skill)
        
        return list(set(normalized))
    
    def _find_exact_matches(self, resume_skills: List[str], vacancy_skills: List[str]) -> Set[str]:
        """Поиск точных совпадений"""
        resume_set = set(resume_skills)
        vacancy_set = set(vacancy_skills)
        return resume_set & vacancy_set
    
    def _find_synonym_matches(self, resume_skills: List[str], vacancy_skills: List[str]) -> Set[str]:
        """Поиск совпадений по синонимам"""
        matches = set()
        
        # Строим множества канонических форм
        resume_canonical = set()
        for skill in resume_skills:
            if skill in self.synonym_to_canonical:
                resume_canonical.add(self.synonym_to_canonical[skill])
            else:
                resume_canonical.add(skill)
        
        vacancy_canonical = set()
        for skill in vacancy_skills:
            if skill in self.synonym_to_canonical:
                vacancy_canonical.add(self.synonym_to_canonical[skill])
            else:
                vacancy_canonical.add(skill)
        
        # Находим пересечения
        for skill in vacancy_canonical & resume_canonical:
            matches.add(skill)
        
        return matches
    
    def _find_fuzzy_matches(self, resume_skills: List[str], vacancy_skills: List[str]) -> Set[str]:
        """Поиск нечетких совпадений"""
        matches = set()
        
        for v_skill in vacancy_skills:
            best_match = None
            best_ratio = 0
            
            for r_skill in resume_skills:
                # Пропускаем если длина сильно отличается
                if abs(len(r_skill) - len(v_skill)) > 5:
                    continue
                    
                ratio = SequenceMatcher(None, r_skill, v_skill).ratio()
                if ratio > self.fuzzy_threshold and ratio > best_ratio:
                    best_ratio = ratio
                    best_match = v_skill
            
            if best_match:
                matches.add(best_match)
        
        return matches
    
    def _find_partial_matches(self, resume_skills: List[str], vacancy_skills: List[str]) -> Set[str]:
        """Поиск частичных совпадений (одно слово из фразы)"""
        matches = set()
        
        for v_skill in vacancy_skills:
            v_words = set(v_skill.split())
            if len(v_words) <= 1:
                continue
                
            for r_skill in resume_skills:
                r_words = set(r_skill.split())
                # Если больше половины слов совпадает
                if len(v_words & r_words) >= len(v_words) / 2:
                    matches.add(v_skill)
                    break
        
        return matches
    
    def _calculate_skill_weights(self, vacancy_text: Optional[str], skills: List[str]) -> Dict[str, float]:
        """
        Расчет весов навыков на основе их важности в вакансии
        
        Факторы:
        - Частота упоминания
        - Позиция в тексте (первые предложения)
        - Маркеры важности (must have, required, essential)
        """
        weights = {}
        
        if not vacancy_text:
            return {skill: 1.0 for skill in skills}
        
        vacancy_lower = vacancy_text.lower()
        
        for skill in skills:
            weight = 1.0
            
            # 1. Частота упоминания
            count = vacancy_lower.count(skill)
            weight += min(count * 0.1, 0.3)  # Макс +0.3
            
            # 2. Поиск в первых 500 символах
            if skill in vacancy_lower[:500]:
                weight += 0.2
            
            # 3. Маркеры важности
            importance_markers = [
                (r'must have.*?' + re.escape(skill), 0.5),
                (r'required.*?' + re.escape(skill), 0.4),
                (r'essential.*?' + re.escape(skill), 0.4),
                (r'need.*?' + re.escape(skill), 0.3),
                (r'prefer.*?' + re.escape(skill), -0.2),
                (r'plus.*?' + re.escape(skill), -0.2),
                (r'nice to have.*?' + re.escape(skill), -0.3)
            ]
            
            for pattern, bonus in importance_markers:
                if re.search(pattern, vacancy_lower, re.IGNORECASE):
                    weight += bonus
            
            # Нормализуем вес (0.5 - 2.0)
            weight = max(0.5, min(2.0, weight))
            weights[skill] = round(weight, 2)
        
        return weights
    
    def _calculate_must_have_bonus(self, vacancy_text: Optional[str], matches: Dict) -> float:
        """Бонус за покрытие must-have требований"""
        if not vacancy_text:
            return 0
        
        bonus = 0
        vacancy_lower = vacancy_text.lower()
        
        # Ищем must-have секцию
        must_have_pattern = r'(?:must have|required|essential|qualifications?)[:\s]+(.*?)(?:\n\s*\n|\.\s+[A-Z]|\Z)'
        must_have_section = re.search(must_have_pattern, vacancy_lower, re.DOTALL | re.IGNORECASE)
        
        if must_have_section:
            must_have_text = must_have_section.group(1)
            matched_skills = set(matches.keys())
            
            for skill in matched_skills:
                if skill in must_have_text:
                    bonus += 2  # +2% за каждый покрытый must-have навык
        
        return min(bonus, 15)  # Максимум 15% бонуса
    
    def _calculate_critical_penalty(self, vacancy_text: Optional[str], missing: Set[str]) -> float:
        """Штраф за отсутствие критических навыков"""
        if not vacancy_text:
            return 0
        
        penalty = 0
        vacancy_lower = vacancy_text.lower()
        
        for skill in missing:
            # Больший штраф если skill упоминается как критический
            critical_patterns = [
                r'must have.*?' + re.escape(skill),
                r'required.*?' + re.escape(skill),
                r'essential.*?' + re.escape(skill)
            ]
            
            for pattern in critical_patterns:
                if re.search(pattern, vacancy_lower, re.IGNORECASE):
                    penalty += 10  # -10% за отсутствие must-have
                    break
            else:
                penalty += 3  # -3% за отсутствие обычного навыка
        
        return min(penalty, 40)  # Максимум 40% штрафа
    
    def extract_vacancy_requirements(self, vacancy_text: str) -> Dict:
        """
        Извлечение структурированных требований из текста вакансии
        """
        requirements = {
            'must_have': [],
            'nice_to_have': [],
            'years_experience': None,
            'education': [],
            'certifications': []
        }
        
        text_lower = vacancy_text.lower()
        
        # 1. Must have / Required
        must_have_pattern = r'(?:must have|required|essential|minimum)[:\s]+(.*?)(?:\n\s*\n|\.\s+[A-Z]|\Z)'
        must_have_section = re.search(must_have_pattern, text_lower, re.DOTALL)
        if must_have_section:
            requirements['must_have'] = self._extract_skills_from_text(must_have_section.group(1))
        
        # 2. Nice to have / Preferred
        nice_pattern = r'(?:nice to have|preferred|plus|desired)[:\s]+(.*?)(?:\n\s*\n|\.\s+[A-Z]|\Z)'
        nice_section = re.search(nice_pattern, text_lower, re.DOTALL)
        if nice_section:
            requirements['nice_to_have'] = self._extract_skills_from_text(nice_section.group(1))
        
        # 3. Опыт работы
        exp_pattern = r'(\d+)[\+]?\s*(?:plus\s*)?years?\s+of\s+experience'
        exp_match = re.search(exp_pattern, text_lower)
        if exp_match:
            requirements['years_experience'] = int(exp_match.group(1))
        
        # 4. Образование
        edu_patterns = [
            r'bachelor(?:["\']?s)?\s+(?:degree\s+)?in\s+([^\.]+)',
            r'master(?:["\']?s)?\s+(?:degree\s+)?in\s+([^\.]+)',
            r'phd\s+in\s+([^\.]+)',
            r'bs\s+in\s+([^\.]+)',
            r'ms\s+in\s+([^\.]+)'
        ]
        
        for pattern in edu_patterns:
            matches = re.findall(pattern, text_lower)
            requirements['education'].extend(matches)
        
        # 5. Сертификации
        cert_pattern = r'(?:certified|certification|certificate)[:\s]+([^\.]+)'
        cert_matches = re.findall(cert_pattern, text_lower)
        requirements['certifications'] = [c.strip() for c in cert_matches]
        
        return requirements
    
    def _extract_skills_from_text(self, text: str) -> List[str]:
        """Извлечение навыков из текстового блока"""
        skills = []
        
        # Разбиваем по запятым, точкам с запятой, буллитам
        parts = re.split(r'[,;•\n]', text)
        
        for part in parts:
            part = part.strip()
            if part and len(part) > 1:
                # Проверяем, есть ли этот навык в словаре синонимов
                for canonical, synonyms in self.SYNONYMS.items():
                    for synonym in synonyms:
                        if synonym in part.lower():
                            skills.append(canonical)
                            break
                    else:
                        continue
                    break
                else:
                    skills.append(part)
        
        return list(set(skills))
    
    def get_match_explanation(self, match_result: Dict) -> str:
        """
        Генерация человеко-читаемого объяснения результатов матчинга
        """
        lines = []
        lines.append("🔍 АНАЛИЗ СООТВЕТСТВИЯ НАВЫКОВ")
        lines.append("=" * 50)
        
        score = match_result['score']
        if score >= 80:
            lines.append(f"✅ ОБЩИЙ СКОР: {score}% - Отличное соответствие!")
        elif score >= 60:
            lines.append(f"👍 ОБЩИЙ СКОР: {score}% - Хорошее соответствие")
        elif score >= 40:
            lines.append(f"⚠️ ОБЩИЙ СКОР: {score}% - Среднее соответствие")
        else:
            lines.append(f"❌ ОБЩИЙ СКОР: {score}% - Слабое соответствие")
        
        lines.append("")
        lines.append("📊 СОВПАДЕНИЯ:")
        
        details = match_result.get('match_details', {})
        exact = details.get('exact_matches', [])
        synonym = details.get('synonym_matches', [])
        fuzzy = details.get('fuzzy_matches', [])
        
        if exact:
            lines.append(f"  ✓ Точные совпадения ({len(exact)}): {', '.join(exact[:5])}")
        if synonym:
            lines.append(f"  ↻ Синонимы ({len(synonym)}): {', '.join(synonym[:3])}")
        if fuzzy:
            lines.append(f"  ~ Близкие совпадения ({len(fuzzy)}): {', '.join(fuzzy[:3])}")
        
        missing = match_result.get('missing', [])
        if missing:
            lines.append("")
            lines.append("❌ ОТСУТСТВУЮТ:")
            for skill in missing[:8]:
                weight = match_result.get('weights', {}).get(skill, 1.0)
                if weight > 1.2:
                    lines.append(f"  ⚠️ {skill} (критический навык!)")
                else:
                    lines.append(f"  ✗ {skill}")
        
        return "\n".join(lines)