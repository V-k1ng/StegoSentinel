import torch
from model import StegoNet
from safetensors.torch import save_model


def convert():
    # 1. Создаем архитектуру
    model = StegoNet()

    # 2. Загружаем старые веса .pth
    pth_path = 'stego_model_best.pth'

    if not torch.os.path.exists(pth_path):
        print(f"Ошибка: Файл {pth_path} не найден!")
        return

    print(f"Загрузка старых весов из {pth_path}...")
    # Используем стандартный torch.load для старого формата
    state_dict = torch.load(pth_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)

    # 3. Сохраняем в новом формате .safetensors
    save_path = 'stego_model_best.safetensors'
    save_model(model, save_path)
    print(f"✅ Успешно конвертировано в {save_path}!")


if __name__ == "__main__":
    convert()