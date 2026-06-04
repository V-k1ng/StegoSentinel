import numpy as np
from PIL import Image
import os

STOP_SIGNAL = "#####"
STOP_BYTES = STOP_SIGNAL.encode('utf-8')


def text_to_bits(text):
    """Надежный перевод текста в битовую строку через байты"""
    # Сначала переводим текст в байты UTF-8, затем каждый байт в 8 бит
    return "".join(f"{b:08b}" for b in text.encode('utf-8'))


def bits_to_text(bits):
    """Безопасное преобразование бит в текст"""
    try:
        # Превращаем биты в целое число
        n = int(bits, 2)
        # Считаем количество байт
        bytes_count = (n.bit_length() + 7) // 8
        # Пытаемся декодировать. errors='ignore' пропустит битые символы
        return n.to_bytes(bytes_count, 'big').decode('utf-8', errors='ignore')
    except Exception:
        return ""


def embed_lsb(image_path, output_path, message, target_density=0.0):
    """Внедрение текста с опциональным дозаполнением случайным шумом"""
    img = Image.open(image_path).convert('RGB')
    img_arr = np.array(img, dtype=np.uint8)

    # 1. Готовим текст и стоп-сигнал
    bit_string = text_to_bits(message + STOP_SIGNAL)
    flat_arr = img_arr.flatten()
    total_pixels = len(flat_arr)

    # 2. Считаем, сколько бит нам нужно для достижения target_density (от 0.0 до 1.0)
    target_bits = int(total_pixels * target_density)

    # Если текст больше, чем целевая плотность — расширяем лимит
    if len(bit_string) > target_bits:
        target_bits = len(bit_string)

    if target_bits > total_pixels:
        raise ValueError("Сообщение слишком большое для этой картинки!")

    # 3. Добиваем остаток случайным шумом (чтобы обмануть/протестировать нейросеть)
    noise_size = target_bits - len(bit_string)
    if noise_size > 0:
        # Генерируем массив строк '0' и '1'
        noise_bits = np.random.randint(0, 2, noise_size, dtype=np.uint8).astype(str)
        full_payload = bit_string + "".join(noise_bits)
    else:
        full_payload = bit_string

    # 4. Внедряем всё подряд
    for i in range(len(full_payload)):
        flat_arr[i] = (flat_arr[i] & 254) | int(full_payload[i])

    stego_arr = flat_arr.reshape(img_arr.shape)
    Image.fromarray(stego_arr, 'RGB').save(output_path, format="PNG")


def embed_lsb_partial(image_path, output_path, density=0.15):
    """Внедрение случайного шума заданной плотности (для Fine-tuning)"""
    img = Image.open(image_path).convert('RGB')
    img_arr = np.array(img, dtype=np.uint8)
    mask = np.random.random(img_arr.shape) < density
    random_bits = np.random.randint(0, 2, img_arr.shape, dtype=np.uint8)

    stego_arr = img_arr.copy()
    stego_arr[mask] = (img_arr[mask] & 254) | random_bits[mask]
    Image.fromarray(stego_arr, 'RGB').save(output_path, format="PNG")


def extract_lsb(image_path):
    """Универсальный экстрактор (поддерживает Кириллицу)"""
    if not os.path.exists(image_path):
        return "Ошибка: Файл не найден"

    try:
        img = Image.open(image_path).convert('RGB')
        img_arr = np.array(img)
        flat_arr = img_arr.flatten()

        all_bytes = bytearray()
        current_bits = ""

        # Ограничение для скорости (1 млн пикселей достаточно для любого текста)
        limit = min(len(flat_arr), 1000000)

        for i in range(limit):
            # Собираем биты
            current_bits += str(flat_arr[i] & 1)

            # Как только накопили 8 бит — превращаем в байт
            if len(current_bits) == 8:
                byte_val = int(current_bits, 2)
                all_bytes.append(byte_val)
                current_bits = ""

                # Проверяем, не закончились ли данные стоп-сигналом
                # Сравниваем последние байты с байтовым представлением '#####'
                if all_bytes.endswith(STOP_BYTES):
                    # Отрезаем стоп-сигнал и декодируем всё целиком
                    return all_bytes[:-len(STOP_BYTES)].decode('utf-8', errors='replace')

        return "⚠️ Сигнатура не найдена. Скрытых данных нет."

    except Exception as e:
        return f"❌ Ошибка при анализе: {str(e)}"


# Блок теста - проверь, чтобы он был на самом краю (без отступов слева)
if __name__ == "__main__":
    test_img = "test_cover.png"
    if not os.path.exists(test_img):
        Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)).save(test_img)

    print("--- Тест внедрения текста ---")
    embed_lsb(test_img, "test_stego_text.png", "Привет, курсовая!")
    print(f"Извлечено: {extract_lsb('test_stego_text.png')}")