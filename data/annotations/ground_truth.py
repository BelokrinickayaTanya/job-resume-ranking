"""
Ground truth аннотации от двух HR-специалистов.
Ранжирование для первых 30 резюме (1-30) по 5 вакансиям.
1 = лучшая вакансия, 5 = худшая вакансия.

Источник: Job Vacancy Ranking with Sentence Embeddings, Keywords, and Named Entities
Vanetik & Kogan, Information 2023
"""

import numpy as np
from typing import List, Dict, Tuple

# ============================================
# ANNOTATOR 1 RANKINGS (из статьи)
# ============================================
ANNOTATOR_1_RANKINGS = [
    [2, 1, 4, 3, 5],  # CV 1
    [1, 2, 3, 4, 5],  # CV 2
    [1, 2, 3, 4, 5],  # CV 3
    [3, 1, 2, 4, 5],  # CV 4
    [1, 5, 4, 2, 3],  # CV 5
    [3, 2, 1, 4, 5],  # CV 6
    [3, 2, 1, 5, 4],  # CV 7
    [2, 4, 3, 1, 5],  # CV 8
    [1, 5, 2, 3, 4],  # CV 9  
    [3, 2, 1, 4, 5],  # CV 10
    [1, 2, 3, 4, 5],  # CV 11
    [1, 2, 3, 4, 5],  # CV 12
    [1, 3, 2, 4, 5],  # CV 13
    [1, 2, 3, 4, 5],  # CV 14
    [3, 1, 2, 4, 5],  # CV 15
    [3, 1, 2, 4, 5],  # CV 16
    [3, 1, 2, 4, 5],  # CV 17
    [1, 2, 5, 3, 4],  # CV 18
    [3, 2, 1, 4, 5],  # CV 19
    [3, 2, 1, 4, 5],  # CV 20
    [2, 3, 1, 4, 5],  # CV 21
    [1, 2, 3, 5, 4],  # CV 22
    [2, 1, 3, 5, 4],  # CV 23
    [1, 2, 3, 5, 4],  # CV 24
    [1, 2, 3, 4, 5],  # CV 25
    [2, 1, 3, 4, 5],  # CV 26
    [2, 3, 4, 5, 1],  # CV 27
    [2, 4, 3, 1, 5],  # CV 28 
    [5, 1, 2, 4, 3],  # CV 29
    [2, 1, 4, 3, 5]   # CV 30
]

# ============================================
# ANNOTATOR 2 RANKINGS (из статьи)
# ============================================
ANNOTATOR_2_RANKINGS = [
    [4, 3, 1, 5, 2],  # CV 1
    [2, 4, 3, 1, 5],  # CV 2
    [5, 4, 2, 3, 1],  # CV 3
    [1, 3, 2, 4, 5],  # CV 4
    [5, 1, 2, 4, 3],  # CV 5
    [1, 3, 2, 4, 5],  # CV 6
    [4, 2, 3, 1, 5],  # CV 7
    [2, 4, 3, 1, 5],  # CV 8
    [3, 4, 2, 1, 5],  # CV 9
    [4, 1, 2, 5, 3],  # CV 10
    [2, 4, 3, 5, 1],  # CV 11
    [4, 3, 2, 1, 5],  # CV 12
    [4, 2, 3, 1, 5],  # CV 13
    [3, 4, 2, 1, 5],  # CV 14
    [2, 4, 3, 1, 5],  # CV 15
    [3, 2, 4, 1, 5],  # CV 16
    [4, 2, 3, 1, 5],  # CV 17
    [4, 2, 5, 3, 1],  # CV 18
    [4, 2, 3, 1, 5],  # CV 19
    [1, 5, 2, 4, 3],  # CV 20
    [1, 3, 4, 5, 2],  # CV 21
    [4, 1, 3, 2, 5],  # CV 22
    [1, 3, 4, 2, 5],  # CV 23
    [1, 4, 3, 5, 2],  # CV 24
    [1, 4, 2, 5, 3],  # CV 25
    [1, 5, 2, 4, 3],  # CV 26
    [4, 3, 1, 2, 5],  # CV 27
    [1, 4, 2, 3, 5],  # CV 28
    [5, 1, 2, 4, 3],  # CV 29
    [1, 2, 3, 4, 5]   # CV 30
]

# ============================================
# КОРРЕКТИРОВКА: исправление явных опечаток
# ============================================
def validate_rankings(rankings: List[List[int]]) -> bool:
    """Проверка корректности ранжирования (все числа 1-5, уникальны)"""
    for i, ranking in enumerate(rankings):
        if sorted(ranking) != [1, 2, 3, 4, 5]:
            print(f"Предупреждение: ранжирование {i+1} содержит ошибку: {ranking}")
            return False
    return True

# ============================================
# УСРЕДНЕННЫЙ GROUND TRUTH (для оценки)
# ============================================
def compute_consensus_rankings(
    rankings1: List[List[int]], 
    rankings2: List[List[int]]
) -> List[List[int]]:
    """
    Вычисление консенсусного ранжирования по методу средних рангов.
    Используется для сравнения с автоматическими методами.
    """
    consensus = []
    
    for i in range(len(rankings1)):
        # Преобразуем ранги в скоры (меньший ранг = лучше)
        # Для усреднения конвертируем в "очки": 5 = лучшая, 1 = худшая
        scores1 = [6 - r for r in rankings1[i]]
        scores2 = [6 - r for r in rankings2[i]]
        
        # Средние скоры
        avg_scores = np.mean([scores1, scores2], axis=0)
        
        # Конвертируем обратно в ранги
        # Сортируем вакансии по убыванию скора
        sorted_indices = np.argsort(avg_scores)[::-1]
        ranks = np.zeros(5, dtype=int)
        for pos, idx in enumerate(sorted_indices):
            ranks[idx] = pos + 1
            
        consensus.append(ranks.tolist())
    
    return consensus

# ============================================
# ГОТОВЫЙ GROUND TRUTH ДЛЯ ИСПОЛЬЗОВАНИЯ
# ============================================
GROUND_TRUTH_AVERAGE = compute_consensus_rankings(
    ANNOTATOR_1_RANKINGS, 
    ANNOTATOR_2_RANKINGS
)

# Словарь для быстрого доступа по ID резюме
GROUND_TRUTH_DICT = {
    i+1: GROUND_TRUTH_AVERAGE[i] 
    for i in range(len(GROUND_TRUTH_AVERAGE))
}

# ============================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================
def get_ground_truth(cv_id: int) -> List[int]:
    """Получить ground truth для конкретного резюме (1-30)"""
    return GROUND_TRUTH_DICT.get(cv_id, [1, 2, 3, 4, 5])

def get_all_ground_truth() -> Dict[int, List[int]]:
    """Получить все ground truth"""
    return GROUND_TRUTH_DICT.copy()

def get_annotator_agreement() -> Dict[str, float]:
    """
    Расчет согласованности между аннотаторами
    (Krippendorff's alpha будет вычислен в модуле evaluation)
    """
    from scipy.stats import spearmanr
    
    agreements = []
    for i in range(30):
        rho, _ = spearmanr(ANNOTATOR_1_RANKINGS[i], ANNOTATOR_2_RANKINGS[i])
        agreements.append(rho)
    
    return {
        'mean_spearman': np.mean(agreements),
        'min_spearman': np.min(agreements),
        'max_spearman': np.max(agreements),
        'std_spearman': np.std(agreements)
    }

# ============================================
# ДЕМОНСТРАЦИЯ
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("GROUND TRUTH VALIDATION")
    print("=" * 60)
    
    valid1 = validate_rankings(ANNOTATOR_1_RANKINGS)
    valid2 = validate_rankings(ANNOTATOR_2_RANKINGS)
    print(f"Annotator 1 rankings valid: {valid1}")
    print(f"Annotator 2 rankings valid: {valid2}")
    
    print("\n" + "=" * 60)
    print("ANNOTATOR AGREEMENT")
    print("=" * 60)
    agreement = get_annotator_agreement()
    print(f"Mean Spearman correlation: {agreement['mean_spearman']:.3f}")
    print(f"Min Spearman correlation: {agreement['min_spearman']:.3f}")
    print(f"Max Spearman correlation: {agreement['max_spearman']:.3f}")
    
    print("\n" + "=" * 60)
    print("SAMPLE CONSENSUS RANKINGS (первые 5 CV)")
    print("=" * 60)
    for i in range(5):
        print(f"CV {i+1}: {GROUND_TRUTH_AVERAGE[i]}")
        print(f"  Ann1: {ANNOTATOR_1_RANKINGS[i]}")
        print(f"  Ann2: {ANNOTATOR_2_RANKINGS[i]}")
        print()