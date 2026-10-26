import pandas as pd

from pathlib import Path
path = "data/salary_prediction_data.csv"
project_root = str(Path(__file__).parent.parent / path)
new_project_root = str(Path(__file__).parent.parent / "data" / "salary_prediction_data_redacted")
print(project_root)
print(new_project_root)
df = pd.read_csv(project_root)

print("Размер до очистки")
print(df.shape)
print("/////")
print("Первые пять строк")
print(df.head(5))
print("/////")
print("Типы данных")
print(df.dtypes)
print("/////")
print("Пропуски")
print(df.isnull().sum())
print("/////")
print("Дубликаты")
print(df.duplicated().sum())

df = df.drop_duplicates() # удаление дубликатов

df = df.dropna(subset=["Salary"]) # Удаление пустых строк в целевом столбце

df = df.dropna(axis=1, how='all') # Удаление пустых столбцов

for col in df.select_dtypes(include=["object", "string"]).columns:
    df[col] = df[col].apply(
        lambda x: x.strip() if isinstance(x, str) else x
    )

df.to_csv(new_project_root , index=False)
print(df.shape)