import os
import random
from PIL import Image
from data_manager import embed_lsb_partial  # Используем твой уже написанный инжектор

# ПУТИ (проверь, чтобы папка с PGM называлась именно так или замени путь)
SOURCE_DIR = r'D:\StegoAnalyzer\pgm_data'
DATASET_DIR = 'dataset'


def prepare():
    # Получаем список всех файлов
    all_files = [f for f in os.listdir(SOURCE_DIR) if f.endswith('.pgm')]
    random.shuffle(all_files)  # Перемешиваем для честности
    all_files = all_files[:10000]

    # Делим на 80% (train) и 20% (val)
    split_idx = int(len(all_files) * 0.8)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]

    sets = [('train', train_files), ('val', val_files)]

    for set_name, files in sets:
        print(f" Обработка набора: {set_name}...")
        for i, filename in enumerate(files):
            source_path = os.path.join(SOURCE_DIR, filename)

            # 1. Формируем пути сохранения
            name_only = os.path.splitext(filename)[0]
            cover_path = os.path.join(DATASET_DIR, set_name, 'cover', f"{name_only}.png")
            stego_path = os.path.join(DATASET_DIR, set_name, 'stego', f"{name_only}.png")

            # 2. Конвертируем PGM в PNG и сохраняем как Cover
            with Image.open(source_path) as img:
                # Мы конвертируем в RGB, чтобы наш data_manager работал корректно
                img.convert('RGB').save(cover_path)

            # 3. Создаем Stego из только что сохраненного Cover
            embed_lsb_partial(cover_path, stego_path)

            if i % 100 == 0:
                print(f"Готово: {i}/{len(files)}")


if __name__ == "__main__":
    # Перед запуском убедись, что SOURCE_DIR указан верно!
    # Если файлы лежат в папке рядом, например 'BOSSBase_256', укажи её.
    if not os.path.exists(SOURCE_DIR):
        print(f"Ошибка: Папка {SOURCE_DIR} не найдена!")
    else:
        prepare()
        print("\n[+++] Датасет успешно подготовлен!")