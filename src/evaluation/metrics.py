"""
Модуль метрик оценки качества ранжирования
Поддерживает метрики из статьи:
- Krippendorff's Alpha
- Spearman's Rank Correlation
- Cohen's Kappa
- Accuracy (для бинарной классификации)
"""
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from scipy.stats import spearmanr, kendalltau
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import cohen_kappa_score, accuracy_score
import pandas as pd
from collections import Counter


class RankingMetrics:
    """
    Класс для вычисления метрик качества ранжирования
    """
    
    @staticmethod
    def krippendorff_alpha(
        data: Union[List[List[int]], np.ndarray],
        level: str = 'ordinal'
    ) -> float:
        """
        Вычисление Krippendorff's Alpha - ИСПРАВЛЕННАЯ ВЕРСИЯ
        """
        data = np.array(data)
        
        # Проверка размерности
        if data.ndim != 2:
            raise ValueError("Данные должны быть 2-мерным массивом")
        
        # Транспонируем если нужно (ожидаем raters × items)
        if data.shape[0] > data.shape[1]:
            data = data.T
        
        n_raters, n_items = data.shape
        
        # Значения и частоты
        values = np.unique(data[~np.isnan(data)])
        n_values = len(values)
        
        if n_values == 0:
            return 1.0
        
        # Матрица наблюдаемых совпадений
        o = np.zeros((n_values, n_values))
        
        for i in range(n_items):
            column = data[:, i]
            column = column[~np.isnan(column)]
            for val1 in column:
                for val2 in column:
                    idx1 = np.where(values == val1)[0][0]
                    idx2 = np.where(values == val2)[0][0]
                    o[idx1, idx2] += 1
        
        # Нормализация
        o = o / np.sum(o) if np.sum(o) > 0 else o
        
        # Матрица ожидаемых совпадений
        col_sum = np.sum(o, axis=0)
        row_sum = np.sum(o, axis=1)
        e = np.outer(row_sum, col_sum)
        
        # Матрица расстояний для ordinal данных
        d = np.zeros((n_values, n_values))
        if level == 'ordinal':
            for i in range(n_values):
                for j in range(n_values):
                    d[i, j] = (i - j) ** 2  # Квадрат разницы рангов
        
        # Расчет alpha
        observed_agreement = np.sum(o * d)
        expected_agreement = np.sum(e * d)
        
        if expected_agreement == 0:
            return 1.0
        
        alpha = 1 - (observed_agreement / expected_agreement)
        
        return round(alpha, 4)

    
    @staticmethod
    def spearman_correlation(
        ranking1: List[int],
        ranking2: List[int]
    ) -> Dict[str, float]:
        """
        Spearman's rank correlation coefficient
        
        Args:
            ranking1: Первый массив рангов (1-5)
            ranking2: Второй массив рангов (1-5)
        
        Returns:
            {'rho': float, 'p_value': float}
        """
        rho, p_value = spearmanr(ranking1, ranking2)
        
        return {
            'rho': round(rho, 4),
            'p_value': round(p_value, 4)
        }
    
    @staticmethod
    def cohen_kappa(
        ranking1: List[int],
        ranking2: List[int],
        weights: Optional[str] = 'quadratic'
    ) -> float:
        """
        Cohen's kappa coefficient
        
        Args:
            ranking1: Первый массив рангов
            ranking2: Второй массив рангов
            weights: 'linear', 'quadratic', None
        """
        kappa = cohen_kappa_score(ranking1, ranking2, weights=weights)
        return round(kappa, 4)
    
    @staticmethod
    def accuracy_at_k(
        predicted_rankings: List[List[int]],
        true_rankings: List[List[int]],
        k: int = 1
    ) -> float:
        """
        Accuracy@k - точность предсказания top-k вакансий
        
        Args:
            predicted_rankings: Предсказанные ранжирования
            true_rankings: Истинные ранжирования
            k: Рассматривать top-k вакансий
        """
        correct = 0
        total = len(predicted_rankings)
        
        for pred, true in zip(predicted_rankings, true_rankings):
            # Индексы top-k вакансий
            pred_topk = set(np.argsort(pred)[:k])
            true_topk = set(np.argsort(true)[:k])
            
            if pred_topk == true_topk:
                correct += 1
        
        return round(correct / total, 4)
    
    @staticmethod
    def ndcg_score(
        predicted_rankings: List[int],
        true_rankings: List[int],
        k: int = 5
    ) -> float:
        """
        Normalized Discounted Cumulative Gain @k
        """
        # Конвертируем ранги в релевантность (1-5, 5 - лучшая)
        true_relevance = [6 - r for r in true_rankings]
        pred_relevance = [6 - r for r in predicted_rankings]
        
        # Сортируем по предсказанию
        sorted_indices = np.argsort(pred_relevance)[::-1]
        
        dcg = 0
        idcg = 0
        
        for i, idx in enumerate(sorted_indices[:k]):
            dcg += (2 ** true_relevance[idx] - 1) / np.log2(i + 2)
        
        # Идеальный DCG
        ideal_relevance = sorted(true_relevance, reverse=True)
        for i in range(min(k, len(ideal_relevance))):
            idcg += (2 ** ideal_relevance[i] - 1) / np.log2(i + 2)
        
        if idcg == 0:
            return 0.0
        
        return round(dcg / idcg, 4)
    
    @staticmethod
    def mean_reciprocal_rank(
        predicted_rankings: List[List[int]],
        true_rankings: List[List[int]]
    ) -> float:
        """
        Mean Reciprocal Rank - средний обратный ранг первой правильной вакансии
        """
        mrrs = []
        
        for pred, true in zip(predicted_rankings, true_rankings):
            # Индекс лучшей вакансии по ground truth
            best_vacancy = np.argmin(true)
            
            # Ранг этой вакансии в предсказании
            rank = pred[best_vacancy]
            
            mrrs.append(1.0 / rank)
        
        return round(np.mean(mrrs), 4)
    
    @staticmethod
    def pairwise_accuracy(
        predicted_rankings: List[int],
        true_rankings: List[int]
    ) -> float:
        """
        Pairwise accuracy - доля правильно упорядоченных пар
        """
        n = len(predicted_rankings)
        correct_pairs = 0
        total_pairs = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                total_pairs += 1
                
                pred_order = predicted_rankings[i] < predicted_rankings[j]
                true_order = true_rankings[i] < true_rankings[j]
                
                if pred_order == true_order:
                    correct_pairs += 1
        
        return round(correct_pairs / total_pairs, 4)


class Evaluator:
    """
    Комплексная оценка методов ранжирования
    """
    
    def __init__(self, ground_truth: Dict[int, List[int]]):
        """
        Args:
            ground_truth: Словарь {cv_id: [rank_vac1, ..., rank_vac5]}
        """
        self.ground_truth = ground_truth
        self.metrics = RankingMetrics()
        self.results_history = []
    
    def evaluate(
        self,
        predictions: Dict[int, List[int]],
        method_name: str
    ) -> Dict[str, float]:
        """
        Оценка предсказаний для всех резюме 
        """
        cv_ids = list(self.ground_truth.keys())
        
        pred_list = []
        true_list = []
        
        for cv_id in cv_ids:
            if cv_id in predictions:
                pred_list.append(predictions[cv_id])
                true_list.append(self.ground_truth[cv_id])
        
        if not pred_list:
            return {}
        
        # 1. Krippendorff's Alpha - для каждой пары отдельно
        alpha_values = []
        for pred, true in zip(pred_list, true_list):
            # Создаем матрицу 2 raters × 5 items
            alpha_data = [pred, true]
            alpha = self.metrics.krippendorff_alpha(alpha_data, level='ordinal')
            alpha_values.append(alpha)
        
        avg_alpha = np.mean(alpha_values) if alpha_values else 0.0
        
        # 2. Spearman correlation
        spearman_values = []
        for pred, true in zip(pred_list, true_list):
            rho, _ = spearmanr(pred, true)
            if not np.isnan(rho):
                spearman_values.append(rho)
        avg_spearman = np.mean(spearman_values) if spearman_values else 0.0
        
        # 3. Cohen's Kappa
        kappa_values = []
        for pred, true in zip(pred_list, true_list):
            kappa = cohen_kappa_score(pred, true, weights='quadratic')
            if not np.isnan(kappa):
                kappa_values.append(kappa)
        avg_kappa = np.mean(kappa_values) if kappa_values else 0.0
        
        # 4. Accuracy@1
        acc_at_1 = self.metrics.accuracy_at_k(pred_list, true_list, k=1)
        
        # 5. NDCG@5
        ndcg_values = []
        for pred, true in zip(pred_list, true_list):
            ndcg = self.metrics.ndcg_score(pred, true, k=5)
            if not np.isnan(ndcg):
                ndcg_values.append(ndcg)
        avg_ndcg = np.mean(ndcg_values) if ndcg_values else 0.0
        
        # 6. MRR
        mrr = self.metrics.mean_reciprocal_rank(pred_list, true_list)
        
        results = {
            'method': method_name,
            'krippendorff_alpha': round(avg_alpha, 4),
            'spearman_rho': round(avg_spearman, 4),
            'cohen_kappa': round(avg_kappa, 4),
            'accuracy_at_1': round(acc_at_1, 4),
            'accuracy_at_3': round(self.metrics.accuracy_at_k(pred_list, true_list, k=3), 4),
            'ndcg@5': round(avg_ndcg, 4),
            'mrr': round(mrr, 4),
            'num_samples': len(pred_list)
        }
        
        self.results_history.append(results)
        return results
    
    def evaluate_bidirectional(
        self,
        predictions: Dict[int, List[int]],
        method_name: str
    ) -> Dict[str, float]:
        """
        Двунаправленная оценка качества ранжирования
        
        Args:
            predictions: Словарь {cv_id: [rank_vac1, ..., rank_vac5]}
            method_name: Название метода
        
        Returns:
            Dict с метриками для обоих направлений и комбинированной оценкой
        """
        # 1. Прямое ранжирование (вакансии для каждого CV) - как сейчас
        cv_scores = self.evaluate(predictions, f"{method_name}_cv")
        
        # 2. Обратное ранжирование (CV для каждой вакансии)
        # Транспонируем матрицу предсказаний
        vac_predictions = {}
        vac_ground_truth = {}
        
        # ID вакансий 1-5
        for vac_id in range(1, 6):
            vac_predictions[vac_id] = []
            vac_ground_truth[vac_id] = []
            
            # Все CV с 1 по 30
            for cv_id in range(1, 31):
                if cv_id in predictions and cv_id in self.ground_truth:
                    # Ранг этой вакансии в предсказании для CV
                    pred_rank = predictions[cv_id][vac_id - 1]
                    vac_predictions[vac_id].append(pred_rank)
                    
                    # Ранг этой вакансии в ground truth для CV
                    true_rank = self.ground_truth[cv_id][vac_id - 1]
                    vac_ground_truth[vac_id].append(true_rank)
        
        # 3. Оцениваем качество ранжирования кандидатов для каждой вакансии
        vac_alphas = []
        vac_spearmans = []
        vac_acc1s = []
        
        for vac_id in range(1, 6):
            if vac_id in vac_predictions and vac_id in vac_ground_truth:
                # Создаем временный словарь для evaluate
                temp_pred = {vac_id: vac_predictions[vac_id]}
                temp_gt = {vac_id: vac_ground_truth[vac_id]}
                
                # Создаем временный evaluator для этой вакансии
                temp_evaluator = Evaluator(temp_gt)
                vac_scores = temp_evaluator.evaluate(temp_pred, f"{method_name}_vac{vac_id}")
                
                vac_alphas.append(vac_scores['krippendorff_alpha'])
                vac_spearmans.append(vac_scores['spearman_rho'])
                vac_acc1s.append(vac_scores['accuracy_at_1'])
        
        # 4. Усредненные метрики для обратного ранжирования
        vac_scores_avg = {
            'krippendorff_alpha': np.mean(vac_alphas) if vac_alphas else 0.0,
            'spearman_rho': np.mean(vac_spearmans) if vac_spearmans else 0.0,
            'accuracy_at_1': np.mean(vac_acc1s) if vac_acc1s else 0.0
        }
        
        # 5. Комбинированные метрики
        combined_alpha = (cv_scores['krippendorff_alpha'] + vac_scores_avg['krippendorff_alpha']) / 2
        combined_spearman = (cv_scores['spearman_rho'] + vac_scores_avg['spearman_rho']) / 2
        combined_acc1 = (cv_scores['accuracy_at_1'] + vac_scores_avg['accuracy_at_1']) / 2
        
        # 6. Итоговый результат
        results = {
            'method': f"{method_name}_bidirectional",
            'krippendorff_alpha': round(cv_scores['krippendorff_alpha'], 4),
            'spearman_rho': round(cv_scores['spearman_rho'], 4),
            'accuracy_at_1': round(cv_scores['accuracy_at_1'], 4),
            'ndcg@5': round(cv_scores.get('ndcg@5', 0), 4),
            'mrr': round(cv_scores.get('mrr', 0), 4),
            'cv_krippendorff_alpha': round(cv_scores['krippendorff_alpha'], 4),
            'cv_spearman_rho': round(cv_scores['spearman_rho'], 4),
            'cv_accuracy_at_1': round(cv_scores['accuracy_at_1'], 4),
            'cv_ndcg@5': round(cv_scores.get('ndcg@5', 0), 4),
            'cv_mrr': round(cv_scores.get('mrr', 0), 4),
            'vac_krippendorff_alpha': round(vac_scores_avg['krippendorff_alpha'], 4),
            'vac_spearman_rho': round(vac_scores_avg['spearman_rho'], 4),
            'vac_accuracy_at_1': round(vac_scores_avg['accuracy_at_1'], 4),
            'combined_krippendorff_alpha': round(combined_alpha, 4),
            'combined_spearman_rho': round(combined_spearman, 4),
            'combined_accuracy_at_1': round(combined_acc1, 4)
        }
        
        self.results_history.append(results)
        return results

    def compare_methods(self) -> pd.DataFrame:
        """
        Сравнение всех оцененных методов
        """
        if not self.results_history:
            print("⚠️ Нет результатов для сравнения")
            return pd.DataFrame()
        
        # Нормализуем результаты: у bidirectional другие ключи
        normalized_results = []
        for res in self.results_history:
            if 'combined_krippendorff_alpha' in res:
                # Это bidirectional результат
                normalized = {
                    'method': res['method'],
                    'krippendorff_alpha': res['combined_krippendorff_alpha'],
                    'spearman_rho': res['combined_spearman_rho'],
                    'accuracy_at_1': res['combined_accuracy_at_1'],
                    'ndcg@5': res.get('ndcg@5', 0),
                    'mrr': res.get('mrr', 0)
                }
                normalized_results.append(normalized)
            else:
                normalized_results.append(res)
        
        df = pd.DataFrame(normalized_results)
        
        if 'krippendorff_alpha' in df.columns:
            df = df.sort_values('krippendorff_alpha', ascending=False)
        
        return df
    
    def print_comparison(self):
        """
        Вывод сравнения методов в консоль
        """
        df = self.compare_methods()
        
        print("\n" + "=" * 80)
        print("СРАВНЕНИЕ МЕТОДОВ РАНЖИРОВАНИЯ".center(80))
        print("=" * 80)
        
        # Форматированный вывод
        columns = ['method', 'krippendorff_alpha', 'spearman_rho', 
                  'accuracy_at_1', 'ndcg@5', 'mrr']
        
        print(f"\n{'Метод':<30} {'Alpha':<10} {'Spearman':<10} {'Acc@1':<10} {'NDCG@5':<10} {'MRR':<10}")
        print("-" * 80)
        
        for _, row in df.iterrows():
            print(f"{row['method'][:30]:<30} "
                  f"{row['krippendorff_alpha']:<10.4f} "
                  f"{row['spearman_rho']:<10.4f} "
                  f"{row['accuracy_at_1']:<10.4f} "
                  f"{row['ndcg@5']:<10.4f} "
                  f"{row['mrr']:<10.4f}")
        
        print("=" * 80)
        
        # Лучший метод
        best = df.iloc[0]
        print(f"\n🏆 Лучший метод: {best['method']}")
        print(f"   Krippendorff's Alpha: {best['krippendorff_alpha']:.4f}")
        print(f"   Spearman's Rho: {best['spearman_rho']:.4f}")
        print(f"   Accuracy@1: {best['accuracy_at_1']:.4f}")