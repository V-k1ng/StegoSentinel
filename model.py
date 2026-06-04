import torch
import torch.nn as nn
import numpy as np


class StegoNet(nn.Module):
    def __init__(self):
        super(StegoNet, self).__init__()

        self.preprocessing = nn.Conv2d(3, 30, kernel_size=5, padding=2, bias=False)
        self.load_srm_weights()

        # ОСНОВНЫЕ СЛОИ
        self.layer1 = self._make_layer(30, 32)  # Выход: 128x128
        self.layer2 = self._make_layer(32, 64)  # Выход: 64x64
        self.layer3 = self._make_layer(64, 128)  # Выход: 32x32
        self.layer4 = self._make_layer(128, 256)  # Выход: 16x16

        self.gap = nn.AdaptiveAvgPool2d(1)

        # КЛАССИФИКАТОР
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def _make_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def load_srm_weights(self):
        k1 = torch.tensor([[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]],
                          dtype=torch.float32)
        k2 = torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                          dtype=torch.float32)
        k3 = torch.tensor(
            [[-1, -1, -1, -1, -1], [-1, 2, 2, 2, -1], [-1, 2, 8, 2, -1], [-1, 2, 2, 2, -1], [-1, -1, -1, -1, -1]],
            dtype=torch.float32)

        weights = torch.zeros(30, 3, 5, 5)
        kernels = [k1, k2, k3]

        for i in range(30):
            base_k = kernels[i % 3]
            for j in range(3):
                weights[i, j] = base_k + torch.randn(5, 5) * 0.1

        with torch.no_grad():
            self.preprocessing.weight.copy_(weights)

        for param in self.preprocessing.parameters():
            param.requires_grad = True

    def forward(self, x):
        # Отменяем эффект ToTensor для фильтров, возвращая пикселям их реальные значения
        x = x * 255.0

        # 2. Пропускаем через SRM фильтры
        x = self.preprocessing(x)

        # 3. КРИТИЧЕСКИЙ ФИКС: Берем модуль (Absolute value)
        x = torch.abs(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Проверка создания модели
if __name__ == "__main__":
    model = StegoNet()
    print(model)
    print("\n[+] Архитектура модели успешно создана!")