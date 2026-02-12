"""
Модуль сохранения обработанных данных
"""
import json
import pickle
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import csv


class DataSaver:
    """Сохранение обработанных данных в различных форматах"""
    
    def __init__(self, base_dir: str = "data/processed"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
    def save_text(self, content: str, filename: str, subdir: str = "") -> Path:
        """Сохранение текстового файла"""
        save_dir = self.base_dir / subdir
        save_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = save_dir / f"{filename}.txt"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return filepath
    
    def save_json(self, data: Dict, filename: str, subdir: str = "") -> Path:
        """Сохранение JSON файла"""
        save_dir = self.base_dir / subdir
        save_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = save_dir / f"{filename}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def save_pickle(self, data: Any, filename: str, subdir: str = "") -> Path:
        """Сохранение pickle файла"""
        save_dir = self.base_dir / subdir
        save_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = save_dir / f"{filename}.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        return filepath
    
    def save_csv(self, data: List[Dict], filename: str, subdir: str = "") -> Path:
        """Сохранение CSV файла"""
        save_dir = self.base_dir / subdir
        save_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = save_dir / f"{filename}.csv"
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False, encoding='utf-8')
        
        return filepath
    
    def save_processed_resume(self, cv_id: int, cv_data: Dict) -> Dict[str, Path]:
        """Сохранение обработанного резюме"""
        files = {}
        
        # Текстовый файл с очищенным текстом
        files['text'] = self.save_text(
            cv_data['text'],
            f"cv_{cv_id}_cleaned",
            subdir="resumes"
        )
        
        # JSON с полными данными
        files['json'] = self.save_json(
            cv_data,
            f"cv_{cv_id}_full",
            subdir="resumes"
        )
        
        return files
    
    def save_processed_vacancy(self, vac_id: int, vac_data: Dict) -> Dict[str, Path]:
        """Сохранение обработанной вакансии"""
        files = {}
        
        files['text'] = self.save_text(
            vac_data['description'],
            f"vacancy_{vac_id}_cleaned",
            subdir="vacancies"
        )
        
        files['json'] = self.save_json(
            vac_data,
            f"vacancy_{vac_id}_full",
            subdir="vacancies"
        )
        
        return files
    
    def save_ranking_results(self, results: List[Dict], method: str) -> Path:
        """Сохранение результатов ранжирования"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.save_csv(
            results,
            f"rankings_{method}_{timestamp}",
            subdir="rankings"
        )


class DataLoader:
    """Загрузка сохраненных данных"""
    
    def __init__(self, base_dir: str = "data/processed"):
        self.base_dir = Path(base_dir)
    
    def load_json(self, filepath: Path) -> Dict:
        """Загрузка JSON файла"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_pickle(self, filepath: Path) -> Any:
        """Загрузка pickle файла"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def load_text(self, filepath: Path) -> str:
        """Загрузка текстового файла"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    
    def list_processed_files(self, subdir: str = "") -> List[Path]:
        """Список обработанных файлов"""
        dir_path = self.base_dir / subdir
        if dir_path.exists():
            return list(dir_path.glob("*"))
        return []