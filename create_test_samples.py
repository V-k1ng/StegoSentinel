from data_manager import embed_lsb_partial, embed_lsb
import os


def prepare_test():
    # Названия твоих исходных файлов (замени на свои, если они другие)
    original_images = ["my_photo1.png", "my_photo2.png"]

    for img_path in original_images:
        if not os.path.exists(img_path):
            print(f"⚠️ Файл {img_path} не найден, пропускаю.")
            continue

        name = os.path.splitext(img_path)[0]

        # 1. Создаем вариант с 15% шума (на чем учили Fine-tuning)
        stego_15 = f"{name}_stego_15.png"
        embed_lsb_partial(img_path, stego_15, density=0.15)
        print(f"✅ Создан файл: {stego_15} (15% шума)")

        # 2. Создаем вариант со 100% шумом (максимальная нагрузка)
        stego_100 = f"{name}_stego_100.png"
        embed_lsb_partial(img_path, stego_100, density=1.0)
        print(f"✅ Создан файл: {stego_100} (100% шума)")

        # 3. Создаем вариант с реальным текстом (через наш классический метод)
        stego_text = f"{name}_stego_text.png"
        embed_lsb(img_path, stego_text, "Secret message for university project 2026")
        print(f"✅ Создан файл: {stego_text} (внедрен реальный текст)")


if __name__ == "__main__":
    prepare_test()