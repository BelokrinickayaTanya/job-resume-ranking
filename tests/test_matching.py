"""
Тесты для matching модулей
"""
import unittest
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.matching.keyword_matcher import KeywordMatcher
from src.matching.tfidf_matcher import TfidfMatcher, AdaptiveTfidfMatcher
from src.matching.semantic_matcher import SemanticMatcher
from src.matching.unified_scorer import UnifiedScorer, MatchResult


class TestKeywordMatcher(unittest.TestCase):
    """Тесты keyword matcher"""
    
    def setUp(self):
        self.matcher = KeywordMatcher()
    
    def test_exact_match(self):
        """Тест точных совпадений"""
        resume_skills = ['python', 'java', 'sql']
        vacancy_skills = ['python', 'c++', 'sql']
        
        result = self.matcher.calculate_match_score(resume_skills, vacancy_skills)
        
        self.assertIn('score', result)
        self.assertIn('matched', result)
        self.assertIn('missing', result)
        
        self.assertIn('python', result['matched'])
        self.assertIn('sql', result['matched'])
        self.assertIn('c++', result['missing'])
    
    def test_synonym_match(self):
        """Тест синонимов"""
        resume_skills = ['reactjs', 'cpp']
        vacancy_skills = ['react', 'c++']
        
        result = self.matcher.calculate_match_score(resume_skills, vacancy_skills)
        
        self.assertIn('react', result['matched'])
        self.assertIn('c++', result['matched'])


class TestUnifiedScorer(unittest.TestCase):
    """Тесты unified scorer"""
    
    def setUp(self):
        self.scorer = UnifiedScorer()
    
    def test_score_calculation(self):
        """Тест расчета скора"""
        result = self.scorer.calculate_score(
            cv_id=1,
            cv_text="Python developer with 5 years experience",
            cv_skills=['python', 'django', 'sql'],
            vacancy_id=1,
            vacancy_title="Python Developer",
            vacancy_text="Need Python developer with Django knowledge",
            vacancy_skills=['python', 'django', 'aws']
        )
        
        self.assertIsInstance(result, MatchResult)
        self.assertEqual(result.cv_id, 1)
        self.assertEqual(result.vacancy_id, 1)
        self.assertTrue(0 <= result.total_score <= 100)
        self.assertTrue(0 <= result.keyword_score <= 100)
        self.assertTrue(0 <= result.tfidf_score <= 100)
        self.assertTrue(0 <= result.semantic_score <= 100)
    
    def test_weights(self):
        """Тест весов компонентов"""
        self.assertAlmostEqual(self.scorer.WEIGHTS['keyword'], 0.60)
        self.assertAlmostEqual(self.scorer.WEIGHTS['tfidf'], 0.25)
        self.assertAlmostEqual(self.scorer.WEIGHTS['semantic'], 0.15)


if __name__ == '__main__':
    unittest.main()