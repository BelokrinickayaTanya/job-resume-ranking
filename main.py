
"""
Job Vacancy Ranking System
Главный скрипт для запуска полного пайплайна ранжирования
"""
import re
import os
import sys
import argparse
from pathlib import Path
import pickle
import json
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import pandas as pd
import numpy as np
from tqdm import tqdm

# Добавляем путь к src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Импорт модулей проекта
from src.pipeline.document_loader import DocumentLoader
from src.pipeline.text_extractor import TextExtractor
from src.pipeline.data_saver import DataSaver, DataLoader

from src.analysis.language_detector import LanguageDetector
from src.analysis.keyword_extractor import KeywordExtractor
from src.analysis.named_entity import NamedEntityExtractor
from src.analysis.experience_calculator import ExperienceCalculator
from src.analysis.grammar_checker import GrammarChecker
from src.analysis.error_detector import ErrorDetector

from src.matching.keyword_matcher import KeywordMatcher
from src.matching.tfidf_matcher import TfidfMatcher, AdaptiveTfidfMatcher
from src.matching.semantic_matcher import SemanticMatcher
from src.matching.unified_scorer import UnifiedScorer, MatchResult

from src.ranking.ranker import VacancyRanker, EnsembleRanker, RankingResult
from src.ranking.vm_method import VMMethod, VMOptimizer

from src.evaluation.metrics import RankingMetrics, Evaluator

# ============================================
# ИМПОРТ GROUND TRUTH - ВСЕ ПЕРЕМЕННЫЕ СУЩЕСТВУЮТ!
# ============================================
try:
    from data.annotations.ground_truth import (
        ANNOTATOR_1_RANKINGS,        # ✅ список рангов аннотатора 1
        ANNOTATOR_2_RANKINGS,        # ✅ список рангов аннотатора 2
        GROUND_TRUTH_AVERAGE,        # ✅ усредненные ранги (ВЫЧИСЛЕН!)
        GROUND_TRUTH_DICT,           # ✅ словарь {cv_id: [rank1,...,rank5]}
        validate_rankings,           # ✅ функция валидации
        get_annotator_agreement     # ✅ функция согласованности
    )
    
    GROUND_TRUTH_AVAILABLE = True
    print("✅ Ground truth annotations loaded successfully")
    print(f"   CVs with ground truth: {len(GROUND_TRUTH_DICT)}")
    print(f"   Ground truth average computed: {len(GROUND_TRUTH_AVERAGE)} consensus rankings")
    
    # Проверяем согласованность аннотаторов
    agreement = get_annotator_agreement()
    print(f"   Annotator agreement: {agreement['mean_spearman']:.3f} (Spearman)")
    
    # Валидируем ранжирования
    valid1 = validate_rankings(ANNOTATOR_1_RANKINGS)
    valid2 = validate_rankings(ANNOTATOR_2_RANKINGS)
    print(f"   Annotator 1 rankings valid: {valid1}")
    print(f"   Annotator 2 rankings valid: {valid2}")
    
except ImportError as e:
    print(f"⚠️ Ground truth not available: {e}")
    print("   Please ensure data/annotations/ground_truth.py exists")
    GROUND_TRUTH_AVAILABLE = False
    
    # Заглушки для данных
    ANNOTATOR_1_RANKINGS = []
    ANNOTATOR_2_RANKINGS = []
    GROUND_TRUTH_AVERAGE = []
    GROUND_TRUTH_DICT = {}
    
    # Заглушки для функций
    def validate_rankings(x): return False
    def compute_consensus_rankings(x, y): return []
    def get_ground_truth(cv_id): return [1, 2, 3, 4, 5]
    def get_all_ground_truth(): return {}
    def get_annotator_agreement(): 
        return {'mean_spearman': 0.0, 'min_spearman': 0.0, 'max_spearman': 0.0, 'std_spearman': 0.0}

class JobRankingSystem:
    """
    Основной класс системы ранжирования резюме-вакансий
    """
    
    def __init__(self, 
                 cv_dir: str = "data/CV", 
                 vacancies_file: str = "data/vacancies/5_vacancies.csv",
                 output_dir: str = "results"):
        """
        Args:
            cv_dir: Папка с резюме в формате DOCX
            vacancies_file: CSV файл с 5 вакансиями
            output_dir: Директория для сохранения результатов
        """
        self.cv_dir = cv_dir
        self.vacancies_file = vacancies_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Инициализация компонентов
        self.loader = DocumentLoader(cv_dir, vacancies_file)
        self.text_extractor = TextExtractor()
        self.data_saver = DataSaver()
        
        # Аналитические модули
        self.keyword_extractor = KeywordExtractor()
        self.ner_extractor = NamedEntityExtractor()
        self.experience_calculator = ExperienceCalculator()
        self.grammar_checker = GrammarChecker(use_languagetool=False)
        self.error_detector = ErrorDetector()
        
        # Матчинг и ранжирование
        self.unified_scorer = UnifiedScorer(
            use_adaptive_tfidf=True,
            semantic_model='mini',
            device='cpu'
        )
        self.ranker = VacancyRanker()
        self.ensemble_ranker = EnsembleRanker()
        
        # VM метод из статьи
        self.vm_method = VMMethod(
            resume_text_type='full',
            vacancy_text_type='summary',
            representation='char_ngrams',
            ngram_range=(1, 3),
            summary_sentences=10
        )
        
        # Данные
        self.cv_data = {}      # {cv_id: {'text': str, 'skills': List, 'experience': float, ...}}
        self.vacancies = {}    # {vac_id: {'title': str, 'description': str, 'skills': List, ...}}
        
        # Результаты
        self.ranking_results = {}
        self.evaluation_results = []
        
    def load_data(self, limit_cvs: Optional[int] = None, english_only: bool = True, verbose: bool = False) -> Tuple[Dict, Dict]:
        """
        Загрузка всех данных
        
        Args:
            limit_cvs: Ограничить количество загружаемых резюме
            english_only: Только английские резюме
        """
        print("\n" + "=" * 70)
        print("📂 ЗАГРУЗКА ДАННЫХ".center(70))
        print("=" * 70)
        
        # 1. Загрузка резюме
        print("\n🔄 Загрузка резюме из DOCX файлов...")
        cv_texts = self.loader.load_all_cvs()
        
        if limit_cvs:
            # Берем первые N резюме
            cv_ids = sorted(cv_texts.keys(), key=lambda x: int(x))
            cv_texts = {k: cv_texts[k] for k in cv_ids[:limit_cvs]}
            print(f"   Загружены резюме: {list(cv_texts.keys())}")
        
        print(f"   Загружено резюме: {len(cv_texts)}")
        
        # 2. Фильтрация по языку
        if english_only:
            print("\n🔄 Фильтрация английских резюме...")
            english_cvs = {}
            for cv_id, text in tqdm(cv_texts.items(), desc="   Detecting language"):
                if LanguageDetector.is_english(text):
                    english_cvs[cv_id] = text
                else:
                    print(f"   ⚠️ Резюме {cv_id} пропущено (не английский)")
            cv_texts = english_cvs
            print(f"   Английских резюме: {len(cv_texts)}")
        
        # 3. Анализ каждого резюме
        print("\n🔬 Анализ резюме...")
        for cv_id, text in tqdm(cv_texts.items(), desc="   Processing"):
            # Очистка текста
            cleaned_text = self.text_extractor.clean_text(text)
            
            # используем методы KeywordExtractor
            all_skills = self.keyword_extractor.extract_keywords(cleaned_text, top_n=40)
            
            # Извлекаем навыки по категориям
            programming = self.keyword_extractor._extract_by_category(cleaned_text, 'programming')
            frameworks = self.keyword_extractor._extract_by_category(cleaned_text, 'framework')
            databases = self.keyword_extractor._extract_by_category(cleaned_text, 'database')
            tools = self.keyword_extractor._extract_by_category(cleaned_text, 'tool')
            
            skills_data = {
                'all_skills': all_skills,
                'programming_languages': programming,
                'frameworks': frameworks,
                'databases': databases,
                'tools': tools
            }
            
            if verbose:
                print(f"\n   📍 Резюме {cv_id}:")
                print(f"      Навыки ({len(all_skills)}): {', '.join(all_skills[:15])}")
                print(f"      Языки: {programming}")
                print(f"      Фреймворки: {frameworks}")
                print(f"      Базы данных: {databases}")
                print(f"      Инструменты: {tools}")

            # === NER: ТОЛЬКО ДЛЯ ПОДСЧЕТА ОПЫТА ===
            # Извлекаем периоды работы через NER (как дополнение к regex)
            ner_periods = self.ner_extractor.extract_work_periods_ner(cleaned_text)
            
            # Получаем минимальную сводку (только количество дат)
            entity_summary = self.ner_extractor.get_entity_summary(cleaned_text)
            
            # === ПОДСЧЕТ ОПЫТА (REGEX + NER) ===
            # Основной подсчет через regex
            experience_regex = self.experience_calculator.calculate_total_experience(
                cleaned_text, 
                verbose=verbose
            )
            
            # Если regex не нашел периоды, используем NER
            if experience_regex == 0.0 and ner_periods:
                total_years = 0
                for start, end in ner_periods:
                    total_years += (end - start)
                experience = float(total_years)
                experience_method = 'ner'
                if verbose:
                    print(f"   🤖 NER: найдены периоды {ner_periods}, опыт: {experience} лет")
            else:
                experience = experience_regex
                experience_method = 'regex'
            
            # Проверка грамматики
            grammar_result = self.grammar_checker.check(cleaned_text[:5000])
            
            # Детекция ошибок
            errors = self.error_detector.detect_all(cleaned_text)
            
            # Сохраняем все данные
            self.cv_data[cv_id] = {
                'id': cv_id,
                'text': cleaned_text,
                'raw_text': text,
                'skills': skills_data['all_skills'],
                'skills_by_category': skills_data,
                'experience': experience,
                'experience_method': experience_method,  # откуда взяли опыт
                'ner_periods': ner_periods,              # периоды от NER
                'ner_dates_count': entity_summary['dates_found'],  # количество дат
                'entities': entity_summary,
                'grammar_score': grammar_result['score'],
                'grammar_issues': grammar_result.get('total_issues', 0),
                'error_count': errors['total_issues'],
                'word_count': len(cleaned_text.split()),
                'has_email': 'email' in str(errors.get('stats', {})),
                'has_phone': 'phone' in str(errors.get('stats', {}))
            }
        
        print(f"\n✅ Проанализировано резюме: {len(self.cv_data)}")
        
        # 4. Загрузка вакансий
        print("\n🔄 Загрузка вакансий...")
        df_vacancies = self.loader.load_vacancies()
        
        for _, row in df_vacancies.iterrows():
            vac_id = row['vacancy_id']
            description = row['job_description']
            
            # Очистка
            cleaned_desc = self.text_extractor.clean_text(description)
            
            # Извлечение навыков из вакансии
            vac_skills = self.keyword_extractor.extract_keywords(cleaned_desc, top_n=30)
            
            required_years = None
            exp_match = re.search(r'(\d+)[\+]?\s*(?:plus\s*)?years?\s+of\s+experience', description, re.IGNORECASE)
            if exp_match:
                required_years = int(exp_match.group(1))
            
            
            self.vacancies[vac_id] = {
                'id': vac_id,
                'title': row['job_title'],
                'description': cleaned_desc,
                'raw_description': description,
                'skills': vac_skills,
                'required_years': required_years, 
                'uid': row['uid']
            }
        
        print(f"✅ Загружено вакансий: {len(self.vacancies)}")
        
        # 5. Сохраняем обработанные данные
        self._save_processed_data()
        
        return self.cv_data, self.vacancies
    
    def _save_processed_data(self):
        """Сохранение обработанных данных"""
        # Сохраняем резюме
        for cv_id, cv_data in self.cv_data.items():
            self.data_saver.save_processed_resume(cv_id, cv_data)
        
        # Сохраняем вакансии
        for vac_id, vac_data in self.vacancies.items():
            self.data_saver.save_processed_vacancy(vac_id, vac_data)
        
        # Сохраняем метадату
        metadata = {
            'num_cvs': len(self.cv_data),
            'num_vacancies': len(self.vacancies),
            'cv_ids': list(self.cv_data.keys()),
            'vacancy_ids': list(self.vacancies.keys()),
            'timestamp': datetime.now().isoformat()
        }
        
        self.data_saver.save_json(metadata, 'metadata', subdir='')
    
    def analyze_experience_distribution(self) -> Dict:
        """Анализ распределения опыта"""
        experiences = [data['experience'] for data in self.cv_data.values()]
        
        if not experiences:
            return {}
        
        stats = {
            'mean': float(np.mean(experiences)),
            'median': float(np.median(experiences)),
            'min': float(np.min(experiences)),
            'max': float(np.max(experiences)),
            'std': float(np.std(experiences)),
            'count': len(experiences)
        }
        
        # Распределение по уровням
        levels = {
            'Junior (<3)': sum(1 for e in experiences if e < 3),
            'Middle (3-5)': sum(1 for e in experiences if 3 <= e < 5),
            'Senior (5-8)': sum(1 for e in experiences if 5 <= e < 8),
            'Lead (8+)': sum(1 for e in experiences if e >= 8)
        }
        
        stats['levels'] = levels
        stats['level_percentages'] = {
            k: round(v / len(experiences) * 100, 1) 
            for k, v in levels.items()
        }
        
        print("\n" + "=" * 70)
        print("📊 АНАЛИЗ ОПЫТА КАНДИДАТОВ".center(70))
        print("=" * 70)
        print(f"   Средний опыт: {stats['mean']:.1f} лет")
        print(f"   Медианный опыт: {stats['median']:.1f} лет")
        print(f"   Мин/Макс: {stats['min']:.1f} - {stats['max']:.1f} лет")
        print(f"\n   📈 Распределение по уровням:")
        for level, count in levels.items():
            print(f"      {level}: {count} ({stats['level_percentages'][level]}%)")
        
        return stats
    
    def run_ranking(self, 
                    method: str = 'unified', 
                    save_results: bool = True,
                    verbose: bool = True) -> Dict[int, List[int]]:
        """
        Запуск ранжирования выбранным методом
        
        Args:
            method: 'unified', 'vm_method', 'okapi_bm25', 'bert_rank', 'ensemble'
            save_results: Сохранять результаты в файл
            verbose: Детальный вывод
        """
        print("\n" + "=" * 70)
        print(f"🎯 РАНЖИРОВАНИЕ МЕТОДОМ: {method.upper()}".center(70))
        print("=" * 70)
        
        if not self.cv_data or not self.vacancies:
            print("❌ Данные не загружены. Сначала выполните load_data()")
            return {}
        
        predictions = {}
        
        # Выбор метода
        for cv_id, cv_data in tqdm(self.cv_data.items(), 
                                  desc=f"   Ранжирование {method}",
                                  disable=not verbose):
            
            if method == 'unified':
                result = self.ranker.rank_unified(
                    cv_id=cv_id,
                    cv_text=cv_data['text'],
                    cv_skills=cv_data['skills'],
                    cv_experience=cv_data['experience'], 
                    vacancies=self.vacancies
                )

            elif method == 'bidirectional':
                result = self.ensemble_ranker.rank_bidirectional(
                    cv_id=cv_id,
                    cv_data=cv_data,
                    vacancies=self.vacancies,
                    all_cvs=self.cv_data,  # Передаем ВСЕ резюме для обратного ранжирования
                    cv_experience=cv_data['experience']
                )
            elif method == 'competition':
                result = self.ensemble_ranker.rank_with_competition(
                    cv_id=cv_id,
                    cv_data=cv_data,
                    vacancies=self.vacancies,
                    all_cvs=self.cv_data,
                    cv_experience=cv_data['experience']
                )    
            elif method == 'vm_method':
                # Используем VM метод из статьи
                rankings = self.vm_method.get_rankings(
                    resume_text=cv_data['text'],
                    vacancies=self.vacancies,
                    resume_keywords=cv_data['skills']
                )
                from src.ranking.ranker import RankingResult
                result = RankingResult(
                    cv_id=cv_id,
                    rankings=rankings,
                    scores=[0]*5,
                    method='vm_method'
                )
            elif method == 'okapi_bm25':
                result = self.ranker.rank_okapi_bm25(
                    cv_id=cv_id,
                    cv_text=cv_data['text'],
                    vacancies=self.vacancies
                )
            elif method == 'bert_rank':
                result = self.ranker.rank_bert(
                    cv_id=cv_id,
                    cv_text=cv_data['text'],
                    vacancies=self.vacancies
                )
            elif method == 'ensemble':
                result = self.ensemble_ranker.rank_ensemble(
                    cv_id=cv_id,
                    cv_text=cv_data['text'],
                    cv_skills=cv_data['skills'],
                    vacancies=self.vacancies,
                    cv_experience=cv_data['experience']
                )
            else:
                raise ValueError(f"Неизвестный метод: {method}")
            
            predictions[cv_id] = result.rankings
        
        if verbose:
            print(f"\n   📊 Ранжирование для CV {cv_id}:")
            print(f"      Вакансия 1: {result.rankings[0]} ранг")
            print(f"      Вакансия 2: {result.rankings[1]} ранг")
            print(f"      Вакансия 3: {result.rankings[2]} ранг")
            print(f"      Вакансия 4: {result.rankings[3]} ранг")
            print(f"      Вакансия 5: {result.rankings[4]} ранг")
            
            # Если есть ground truth, показать сравнение
            if cv_id in GROUND_TRUTH_DICT:
                gt = GROUND_TRUTH_DICT[cv_id]
                print(f"      📌 Ground truth: {gt}")
                
                # Подсчет точности top-1
                top1_pred = result.rankings.index(1) + 1  # Какая вакансия на 1 месте
                top1_gt = gt.index(1) + 1
                if top1_pred == top1_gt:
                    print(f"      ✅ Top-1 совпадает: Вакансия {top1_pred}")
                else:
                    print(f"      ❌ Top-1 не совпадает: предсказано {top1_pred}, должно быть {top1_gt}")
                self.ranking_results[method] = predictions
                
                # Сохранение
                if save_results:
                    self._save_ranking_results(method, predictions)
                
                # Краткая статистика
                if verbose and predictions:
                    cv_with_gt = [cv for cv in predictions if cv in GROUND_TRUTH_DICT]
                    print(f"\n   ✅ Ранжировано резюме: {len(predictions)}")
                    print(f"   📊 С ground truth: {len(cv_with_gt)}")
        
        return predictions
    
    def _save_ranking_results(self, method: str, predictions: Dict[int, List[int]]):
        """Сохранение результатов ранжирования"""
        # CSV формат
        results_list = []
        for cv_id, rankings in predictions.items():
            results_list.append({
                'cv_id': cv_id,
                'rankings': str(rankings),
                'rank_vac1': rankings[0],
                'rank_vac2': rankings[1],
                'rank_vac3': rankings[2],
                'rank_vac4': rankings[3],
                'rank_vac5': rankings[4]
            })
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"rankings_{method}_{timestamp}.csv"
        
        df = pd.DataFrame(results_list)
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"\n   💾 Результаты сохранены: {filename}")
        
        # Также сохраняем в pickle для быстрой загрузки
        pickle_file = self.output_dir / f"rankings_{method}_{timestamp}.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump(predictions, f)
    
    def evaluate_all_methods(self, methods: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Оценка всех методов ранжирования
        """
        if not GROUND_TRUTH_AVAILABLE:
            print("\n❌ Ground truth не доступен. Оценка невозможна.")
            return pd.DataFrame()
        
        print("\n" + "=" * 70)
        print("📏 ОЦЕНКА КАЧЕСТВА РАНЖИРОВАНИЯ".center(70))
        print("=" * 70)
        
        if methods is None:
            # ✅ ДОБАВЛЯЕМ BIDIRECTIONAL В СПИСОК!
            methods = ['unified', 'vm_method', 'okapi_bm25', 'bert_rank', 'ensemble', 'bidirectional']
        
        # Запускаем методы, если еще не выполнены
        for method in methods:
            if method not in self.ranking_results:
                print(f"\n🔄 Метод {method} не выполнен. Запускаем...")
                self.run_ranking(method, save_results=True, verbose=False)
        
        # Инициализация evaluator
        evaluator = Evaluator(GROUND_TRUTH_DICT)
        
        # Оценка каждого метода

        for method in methods:
            if method in self.ranking_results:
                predictions = self.ranking_results[method]
                
                eval_predictions = {
                    cv_id: pred for cv_id, pred in predictions.items()
                    if cv_id in GROUND_TRUTH_DICT
                }
                
                if eval_predictions:
                    # ✅ Для bidirectional используем специальную оценку
                    if method == 'bidirectional':
                        results = evaluator.evaluate_bidirectional(eval_predictions, method.upper())
                        # Показываем комбинированную метрику
                        print(f"\n📊 {method.upper()}:")
                        print(f"   Krippendorff's Alpha: {results['combined_krippendorff_alpha']:.4f}")
                        print(f"   Spearman's Rho: {results['combined_spearman_rho']:.4f}")
                        print(f"   Accuracy@1: {results['combined_accuracy_at_1']:.4f}")
                    else:
                        results = evaluator.evaluate(eval_predictions, method.upper())
                        print(f"\n📊 {method.upper()}:")
                        print(f"   Krippendorff's Alpha: {results['krippendorff_alpha']:.4f}")
                        print(f"   Spearman's Rho: {results['spearman_rho']:.4f}")
                        print(f"   Accuracy@1: {results['accuracy_at_1']:.4f}")
                        print(f"   NDCG@5: {results['ndcg@5']:.4f}")
                    
                    self.evaluation_results.append(results)
                    
                    print(f"\n📊 {method.upper()}:")
                    print(f"   Krippendorff's Alpha: {results.get('krippendorff_alpha', results.get('combined_krippendorff_alpha', 0)):.4f}")
                    print(f"   Spearman's Rho: {results.get('spearman_rho', results.get('combined_spearman_rho', 0)):.4f}")
                    print(f"   Accuracy@1: {results.get('accuracy_at_1', results.get('combined_accuracy_at_1', 0)):.4f}")
                    print(f"   NDCG@5: {results.get('ndcg@5', 0):.4f}")
        
        # Финальное сравнение
        print("\n" + "=" * 70)
        print("🏆 ИТОГОВОЕ СРАВНЕНИЕ".center(70))
        print("=" * 70)
        
        evaluator.print_comparison()
        
        # Сохраняем сравнение
        df_comparison = evaluator.compare_methods()
        comparison_file = self.output_dir / "method_comparison.csv"
        df_comparison.to_csv(comparison_file, index=False)
        print(f"\n💾 Сравнение сохранено: {comparison_file}")
        
        return df_comparison
    
    def generate_report(self) -> str:
        """Генерация полного отчета"""
        print("\n" + "=" * 70)
        print("📝 ГЕНЕРАЦИЯ ОТЧЕТА".center(70))
        print("=" * 70)
        
        report_lines = []
        report_lines.append("# Job Vacancy Ranking System Report")
        report_lines.append("")
        report_lines.append(f"**Дата генерации:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # 1. Информация о данных
        report_lines.append("## 📊 Dataset Statistics")
        report_lines.append("")
        report_lines.append(f"- **Резюме загружено:** {len(self.cv_data)}")
        report_lines.append(f"- **Вакансий:** {len(self.vacancies)}")
        
        if self.cv_data:
            experiences = [d['experience'] for d in self.cv_data.values()]
            report_lines.append(f"- **Средний опыт:** {np.mean(experiences):.1f} лет")
            report_lines.append(f"- **Медианный опыт:** {np.median(experiences):.1f} лет")
        
        # 2. Ground truth информация
        report_lines.append("")
        report_lines.append("## 🎯 Ground Truth")
        report_lines.append("")
        if GROUND_TRUTH_AVAILABLE:
            report_lines.append(f"- **Резюме с разметкой:** {len(GROUND_TRUTH_DICT)}")
            agreement = get_annotator_agreement()
            report_lines.append(f"- **Согласованность аннотаторов:** {agreement['mean_spearman']:.3f} (Spearman)")
        else:
            report_lines.append("- **Ground truth не загружен**")
        
        # 3. Результаты методов
        if self.evaluation_results:
            report_lines.append("")
            report_lines.append("## 📈 Methods Comparison")
            report_lines.append("")
            report_lines.append("| Method | Krippendorff's α | Spearman ρ | Acc@1 | NDCG@5 | MRR |")
            report_lines.append("|--------|-----------------|------------|-------|--------|-----|")
            
            for result in sorted(self.evaluation_results, 
                               key=lambda x: x.get('krippendorff_alpha', 0), 
                               reverse=True):
                report_lines.append(
                    f"| {result['method']} | "
                    f"{result.get('krippendorff_alpha', 0):.4f} | "
                    f"{result.get('spearman_rho', 0):.4f} | "
                    f"{result.get('accuracy_at_1', 0):.4f} | "
                    f"{result.get('ndcg@5', 0):.4f} | "
                    f"{result.get('mrr', 0):.4f} |"
                )
        
        # 4. Лучшая конфигурация
        report_lines.append("")
        report_lines.append("## 🏆 Best Configuration")
        report_lines.append("")
        report_lines.append("Based on the original paper:")
        report_lines.append("")
        report_lines.append("- **Resume:** Full text")
        report_lines.append("- **Vacancy:** BERT extractive summary (10 sentences)")
        report_lines.append("- **Text representation:** Character n-grams (1-3)")
        report_lines.append("- **Distance:** L1 (Manhattan)")
        report_lines.append("- **Krippendorff's Alpha:** 0.6287")
        
        # 5. Наши улучшения
        report_lines.append("")
        report_lines.append("## 🚀 Our Improvements")
        report_lines.append("")
        report_lines.append("- **Unified Scoring:** Keyword 80% + TF-IDF 15% + Semantic 5%")
        report_lines.append("- **Fuzzy matching:** Synonyms and partial matches")
        report_lines.append("- **Adaptive TF-IDF:** Tech boost for IT terms")
        report_lines.append("- **Chunked semantic:** Section-based similarity")
        report_lines.append("- **Ensemble:** Combined predictions from multiple methods")
        report_lines.append("- **Full pipeline:** From DOCX to evaluation report")
        
        # Сохраняем отчет
        report_file = self.output_dir / "ranking_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"\n✅ Отчет сохранен: {report_file}")
        
        return str(report_file)

    def optimize_weights_ensemble(self):
        """
        Оптимизация весов 
        """
        print("\n" + "=" * 70)
        print("⚙️ ОПТИМИЗАЦИЯ ВЕСОВ".center(70))
        print("=" * 70)
        
        # Предобучаем векторизатор
        first_cv_id = list(self.cv_data.keys())[0]
        first_cv_text = self.cv_data[first_cv_id]['text']
        self.ensemble_ranker._get_vm_scores_fixed(first_cv_text, self.vacancies)
        
        best_weights = None
        best_score = -1
        
        # Только простые веса, ТОЛЬКО rank_ensemble 
        for unified_w in [0.4, 0.5, 0.6]:
            for vm_w in [0.2, 0.3, 0.4]:
                for bert_w in [0.1, 0.2, 0.3]:
                    weights = {
                        'unified': unified_w,
                        'vm_method': vm_w,
                        'bert_rank': bert_w
                    }
                    
                    # Нормализуем
                    total = sum(weights.values())
                    weights = {k: v/total for k, v in weights.items()}
                    
                    print(f"\n🔄 Тестируем веса: {weights}")
                    
                    # ✅ ТОЛЬКО rank_ensemble, БЕЗ bidirectional!
                    scores = []
                    for cv_id, cv_data in self.cv_data.items():
                        if cv_id <= 30:
                            try:
                                result = self.ensemble_ranker.rank_ensemble(  # ← НЕ rank_bidirectional!
                                    cv_id=cv_id,
                                    cv_text=cv_data['text'],
                                    cv_skills=cv_data['skills'],
                                    vacancies=self.vacancies,
                                    cv_experience=cv_data['experience'],
                                    weights=weights
                                )
                                if cv_id in GROUND_TRUTH_DICT:
                                    gt = GROUND_TRUTH_DICT[cv_id]
                                    if result.rankings.index(1) == gt.index(1):
                                        scores.append(1)
                                    else:
                                        scores.append(0)
                            except Exception as e:
                                continue
                    
                    if scores:
                        acc1 = np.mean(scores)
                        print(f"   Accuracy@1: {acc1:.3f}")
                        if acc1 > best_score:
                            best_score = acc1
                            best_weights = weights
                            print(f"   🆕 НОВЫЙ ЛУЧШИЙ!")
        
        print("\n" + "=" * 70)
        print(f"🏆 ОПТИМАЛЬНЫЕ ВЕСА: {best_weights}")
        print(f"   Accuracy@1: {best_score:.3f}")
        print("=" * 70)
        
        return best_weights

def main():
    """Точка входа"""
    parser = argparse.ArgumentParser(
        description='Job Vacancy Ranking System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python main.py --limit 30 --method unified --evaluate
  python main.py --method all --evaluate --report
  python main.py --method vm_method --limit 10
  python main.py --method ensemble --evaluate --report
        """
    )
    
    parser.add_argument('--cv_dir', type=str, default='data/CV',
                       help='Папка с резюме в формате DOCX')
    parser.add_argument('--vacancies', type=str, 
                       default='data/vacancies/5_vacancies.csv',
                       help='CSV файл с 5 вакансиями')
    parser.add_argument('--output', type=str, default='results',
                       help='Директория для сохранения результатов')
    parser.add_argument('--limit', type=int, default=None,
                       help='Количество резюме для загрузки')
    parser.add_argument('--method', type=str, default='unified',
                       choices=['unified', 'vm_method', 'okapi_bm25', 
                               'bert_rank', 'ensemble', 'all', 'bidirectional', 'competition'],
                       help='Метод ранжирования')
    parser.add_argument('--no_english_only', action='store_true',
                       help='Не фильтровать только английские резюме')
    parser.add_argument('--evaluate', action='store_true',
                       help='Оценить качество (требуется ground truth)')
    parser.add_argument('--report', action='store_true',
                       help='Сгенерировать отчет')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Детальный вывод')
    parser.add_argument('--optimize', action='store_true',
                       help='Оптимизировать веса bidirectional')
    parser.add_argument('--fast', action='store_true',
                   help='Быстрый режим (использовать веса по умолчанию, без оптимизации)')    
    args = parser.parse_args()
    
    # Создаем экземпляр системы
    system = JobRankingSystem(
        cv_dir=args.cv_dir,
        vacancies_file=args.vacancies,
        output_dir=args.output
    )
    
    # Загрузка данных
    system.load_data(
        limit_cvs=args.limit,
        english_only=not args.no_english_only,
        verbose=args.verbose
    )
    
    # Анализ опыта
    system.analyze_experience_distribution()
    
    # Оптимизация весов bidirectional
    if args.optimize and not args.fast:
        print("\n" + "=" * 70)
        print("⚙️ ЗАПУСК ОПТИМИЗАЦИИ ВЕСОВ".center(70))
        print("=" * 70)
        
        optimal_weights = system.optimize_weights_ensemble()
        
        # Сохраняем оптимальные веса для использования
        system.ensemble_ranker.optimal_weights = optimal_weights
    else:
        print("\n⚡ Быстрый режим: используются веса по умолчанию")
    
        
        # Также обновляем веса в UnifiedScorer если нужно
        if hasattr(system.unified_scorer, 'WEIGHTS'):
            print("\n📊 Текущие веса Unified Scorer:")
            print(f"   Keyword: {system.unified_scorer.WEIGHTS['keyword']*100}%")
            print(f"   TF-IDF: {system.unified_scorer.WEIGHTS['tfidf']*100}%")
            print(f"   Semantic: {system.unified_scorer.WEIGHTS['semantic']*100}%")    

    # Ранжирование
    if args.method == 'all':
        methods = ['unified', 'vm_method', 'okapi_bm25', 'bert_rank', 'ensemble', 'bidirectional', 'competition']
        for method in methods:
            system.run_ranking(method, save_results=True, verbose=args.verbose)
    else:
        system.run_ranking(args.method, save_results=True, verbose=args.verbose)
    
    # Оценка
    if args.evaluate:
        if GROUND_TRUTH_AVAILABLE:
            system.evaluate_all_methods()
        else:
            print("\n❌ Невозможно выполнить оценку: ground truth не загружен")
            print("   Убедитесь, что файл data/annotations/ground_truth.py существует")
    
    # Отчет
    if args.report:
        system.generate_report()
    
    print("\n" + "=" * 70)
    print("✅ ГОТОВО!".center(70))
    print("=" * 70)


if __name__ == "__main__":
    main()