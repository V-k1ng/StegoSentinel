import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from model import StegoNet
from safetensors.torch import load_model
import os
import numpy as np

# --- НАСТРОЙКИ ---
MODEL_PATH = 'stego_model_best.safetensors'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATCH_SIZE = 256  # Размер окна, на котором училась сеть


def sliding_window_predict(image_path):
    if not os.path.exists(MODEL_PATH):
        print("❌ Ошибка: Веса модели не найдены!")
        return

    # 1. Загрузка модели
    model = StegoNet().to(DEVICE)
    load_model(model, MODEL_PATH)
    model.eval()

    # 2. Загрузка изображения
    img = Image.open(image_path).convert('RGB')
    w, h = img.size
    print(f"📷 Оригинальный размер: {w}x{h}")

    # Если картинка меньше 256x256, мы её просто дополним или скажем об ошибке
    if w < PATCH_SIZE or h < PATCH_SIZE:
        print("⚠️ Картинка слишком мала для глубокого анализа. Использую Resize.")
        img = img.resize((PATCH_SIZE, PATCH_SIZE))
        w, h = PATCH_SIZE, PATCH_SIZE

    # 3. Подготовка трансформации (только ToTensor, никакого ресайза!)
    transform = transforms.ToTensor()

    # 4. Логика скользящего окна
    predictions = []
    # Шаг (stride) можно сделать меньше 256, если хочешь более плотный анализ (но будет дольше)
    stride = 256

    print(f"🚀 Запуск сканирования окном {PATCH_SIZE}x{PATCH_SIZE}...")

    # Считаем количество плиток по горизонтали и вертикали
    n_x = w // stride
    n_y = h // stride

    total_patches = n_x * n_y
    detected_patches = 0

    with torch.no_grad():
        for y in range(0, n_y * stride, stride):
            row_probs = []
            for x in range(0, n_x * stride, stride):
                # Вырезаем патч (crop) пиксель-в-пиксель
                patch = img.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE))
                patch_tensor = transform(patch).unsqueeze(0).to(DEVICE)

                # Анализируем
                output = model(patch_tensor)
                probs = F.softmax(output, dim=1)
                stego_prob = probs[0][1].item() * 100

                predictions.append(stego_prob)
                if stego_prob > 50:
                    detected_patches += 1

    # 5. Агрегация результатов
    avg_stego_prob = sum(predictions) / len(predictions)
    max_stego_prob = max(predictions)

    print("\n" + "=" * 45)
    print(f"📊 ОТЧЕТ СКАНИРОВАНИЯ: {os.path.basename(image_path)}")
    print("=" * 45)
    print(f"Всего проанализировано блоков : {total_patches}")
    print(f"Подозрительных блоков         : {detected_patches}")
    print(f"Средняя вероятность Stego     : {avg_stego_prob:.2f}%")
    print(f"Пиковая вероятность в блоке   : {max_stego_prob:.2f}%")
    print("-" * 45)

    # Вердикт на основе пиковой или средней вероятности
    # В стеганографии даже один "красный" блок — это повод для тревоги
    if max_stego_prob > 70 or detected_patches > (total_patches * 0.1):
        print("🚨 ВЕРДИКТ: ОБНАРУЖЕНО СКРЫТОЕ ВНЕДРЕНИЕ!")
        print("💡 Совет: Проверьте подозрительные области экстрактором.")
    else:
        print("✅ ВЕРДИКТ: Аномалий не обнаружено.")
    print("=" * 45 + "\n")


if __name__ == "__main__":
    while True:
        path = input("Введите путь к картинке (или 'q'): ").strip().replace('"', '')
        if path.lower() == 'q': break
        try:
            sliding_window_predict(path)
        except Exception as e:
            print(f"❌ Ошибка: {e}")