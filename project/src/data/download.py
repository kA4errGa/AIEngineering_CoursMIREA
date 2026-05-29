import kagglehub
import os
import shutil


def download_data():

    # Скачиваем датасет
    path = kagglehub.dataset_download(
        "nikhil1e9/loan-default"
    )

    print("Исходный путь:", path)

    target_dir = "./data/RAW/"
    os.makedirs(target_dir, exist_ok=True)

    csv_files = [
        f for f in os.listdir(path)
        if f.endswith(".csv")
    ]

    if not csv_files:
        raise FileNotFoundError(
            "CSV file not found in downloaded dataset"
        )

    csv_file = csv_files[0]

    source_path = os.path.join(path, csv_file)

    target_path = os.path.join(
        target_dir,
        csv_file
    )

    shutil.copy(source_path, target_path)

    print(f"Файл скопирован в {target_path}")

