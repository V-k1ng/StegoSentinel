import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import StegoNet
from safetensors.torch import save_model, load_model  # Импортируем новый формат
import os

# 1. НАСТРОЙКИ (Глобальные параметры)
BATCH_SIZE = 64
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = 'stego_model_best.safetensors'


def main():
    # 0. Очистка кэша перед стартом
    torch.cuda.empty_cache()
    # 2. ПОДГОТОВКА ДАННЫХ
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_data = datasets.ImageFolder(root='dataset/train', transform=transform)
    val_data = datasets.ImageFolder(root='dataset/val', transform=transform)

    # Используем num_workers=2 и pin_memory для RTX 3070
    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers = True # Чтобы не пересоздавать процессы
    )
    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers = True # Чтобы не пересоздавать процессы
    )

    # 3. ИНИЦИАЛИЗАЦИЯ
    model = StegoNet().to(DEVICE)

    # Загрузка весов из SAFETENSORS (если файл существует)
    if os.path.exists(MODEL_NAME):
        load_model(model, MODEL_NAME)
        print(f"✅ Базовые веса загружены из {MODEL_NAME} для Fine-tuning")
    else:
        print("⚠️ Файл весов не найден. Обучение начнется с нуля!")

    criterion = nn.CrossEntropyLoss()
    # Маленький LR для тонкой настройки (Fine-tuning) на 15% шума
    optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-5)

    # Планировщик для снижения LR при плато точности
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    # 4. ЦИКЛ ОБУЧЕНИЯ
    print(f"🚀 Запуск обучения на устройстве: {DEVICE}")
    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # ВАЛИДАЦИЯ (Проверка)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total

        # Обновляем планировщик и получаем текущий LR
        scheduler.step(accuracy)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Эпоха [{epoch + 1}/{EPOCHS}] | Loss: {running_loss / len(train_loader):.4f} | "
              f"Accuracy: {accuracy:.2f}% | LR: {current_lr:.6f}")

        # СОХРАНЕНИЕ ЛУЧШЕЙ МОДЕЛИ В SAFETENSORS
        if accuracy > best_acc:
            best_acc = accuracy
            save_model(model, MODEL_NAME)
            print(f"💾 Сохранена новая лучшая модель с точностью {accuracy:.2f}%!")

        if accuracy > 98.0:
            print("🎉 Цель достигнута!")
            break


# 5. ЗАЩИТА ТОЧКИ ВХОДА (Обязательно для Windows и num_workers)
if __name__ == "__main__":
    main()