"""
Модуль загрузки документов: резюме (DOCX) и вакансии (CSV)
"""
import os
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import docx
from tqdm import tqdm


class DocumentLoader:
    """Загрузчик документов для резюме и вакансий"""
    
    def __init__(self, cv_dir: str = "data/CV", vacancies_file: str = "data/vacancies/5_vacancies.csv"):
        self.cv_dir = Path(cv_dir)
        self.vacancies_file = Path(vacancies_file)
        self.cv_documents = {}
        self.vacancies_df = None
        
    def load_cv_docx(self, file_path: Path) -> Optional[str]:
        """Извлечение текста из DOCX файла с сохранением структуры"""
        try:
            doc = docx.Document(file_path)
            full_text = []
            for para in doc.paragraphs:
                if para.text.strip():
                    full_text.append(para.text)  
            return '\n'.join(full_text) 
        except Exception as e:
            print(f"Ошибка загрузки {file_path}: {e}")
            return None
    
    def load_all_cvs(self, limit: int = None) -> Dict[int, str]:
        """Загрузка всех резюме из папки CV с ЧИСЛОВОЙ сортировкой"""
        cv_files = sorted(self.cv_dir.glob("*.docx"), key=lambda x: int(x.stem))
        cv_dict = {}
        
        print(f"Загрузка резюме из {self.cv_dir}...")
        for cv_file in tqdm(cv_files[:limit] if limit else cv_files):
            cv_id = int(cv_file.stem)  # Преобразуем в число!
            text = self.load_cv_docx(cv_file)
            if text:
                cv_dict[cv_id] = text
        
        print(f"Загружено {len(cv_dict)} резюме")
        print(f"   ID резюме: {sorted(cv_dict.keys())}")
        return cv_dict
    
    def load_vacancies(self) -> pd.DataFrame:
        """Загрузка вакансий из CSV"""
        self.vacancies_df = pd.read_csv(self.vacancies_file)
        # Перенумеровываем ID вакансий с 1 до 5
        self.vacancies_df['vacancy_id'] = range(1, len(self.vacancies_df) + 1)
        print(f"Загружено {len(self.vacancies_df)} вакансий")
        return self.vacancies_df
    
    def get_vacancy_by_id(self, vacancy_id: int) -> Dict:
        """Получение вакансии по ID (1-5)"""
        if self.vacancies_df is None:
            self.load_vacancies()
        vacancy = self.vacancies_df[self.vacancies_df['vacancy_id'] == vacancy_id].iloc[0]
        return {
            'id': vacancy_id,
            'title': vacancy['job_title'],
            'description': vacancy['job_description'],
            'uid': vacancy['uid']
        }