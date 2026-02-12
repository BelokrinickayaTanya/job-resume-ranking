"""
Тесты для pipeline модулей
"""
import unittest
import os
import sys
from pathlib import Path

# Добавляем путь к src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.document_loader import DocumentLoader
from src.pipeline.text_extractor import TextExtractor
from src.pipeline.data_saver import DataSaver, DataLoader


class TestDocumentLoader(unittest.TestCase):
    """Тесты загрузчика документов"""
    
    def setUp(self):
        self.loader = DocumentLoader(
            cv_dir="data/CV",
            vacancies_file="data/vacancies/5_vacancies.csv"
        )
    
    def test_load_cv_docx(self):
        """Тест загрузки DOCX файла"""
        # Проверяем существование тестового файла
        test_file = Path("data/CV/10.docx")
        if test_file.exists():
            text = self.loader.load_cv_docx(test_file)
            self.assertIsNotNone(text)
            self.assertIsInstance(text, str)
            self.assertTrue(len(text) > 0)
    
    def test_load_vacancies(self):
        """Тест загрузки вакансий"""
        df = self.loader.load_vacancies()
        self.assertIsNotNone(df)
        self.assertTrue(len(df) > 0)
        self.assertIn('vacancy_id', df.columns)


class TestTextExtractor(unittest.TestCase):
    """Тесты экстрактора текста"""
    
    def setUp(self):
        self.extractor = TextExtractor()
    
    def test_clean_text(self):
        """Тест очистки текста"""
        text = "Hello, World! 123 2023-now"
        cleaned = self.extractor.clean_text(text)
        self.assertNotIn("Hello", cleaned)  # lowercase
        self.assertNotIn("123", cleaned)    # numbers removed
        self.assertIn("2023", cleaned)      # years preserved
    
    def test_extract_experience_periods(self):
        """Тест извлечения периодов опыта"""
        text = "Worked at Company (2017-2020) and then at Startup (2020-now)"
        periods = self.extractor.extract_experience_periods(text)
        self.assertEqual(len(periods), 2)
        self.assertEqual(periods[0][0], 2017)
        self.assertEqual(periods[0][1], 2020)


if __name__ == '__main__':
    unittest.main()