import customtkinter as ctk
import random
from tkinter import filedialog, messagebox
from PIL import Image, ImageDraw
import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from model import StegoNet
from safetensors.torch import load_model
from data_manager import embed_lsb, extract_lsb
import tempfile

# --- НАСТРОЙКИ ТЕМЫ ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class StegoApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("StegoSentinel | LSB Forensics Suite")
        self.geometry("1100x750")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # Переменные ИИ
        self.model_path = 'stego_model_best.safetensors'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

        self.setup_sidebar()
        self.setup_frames()
        self.load_neural_network()

        # Открываем первую вкладку по умолчанию
        self.select_frame_by_name("scan")

    def load_neural_network(self):
        if os.path.exists(self.model_path):
            self.model = StegoNet().to(self.device)
            load_model(self.model, self.model_path)
            self.model.eval()
        else:
            messagebox.showwarning("Внимание", "Файл весов не найден. Нейросеть отключена.")

    # ==================== НАВИГАЦИЯ ====================
    def setup_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(4, weight=1)

        logo_label = ctk.CTkLabel(self.sidebar, text="🛡️ StegoSentinel", font=ctk.CTkFont(size=20, weight="bold"))
        logo_label.grid(row=0, column=0, padx=20, pady=(20, 30))

        self.btn_scan = ctk.CTkButton(self.sidebar, text="🔍 Нейро-Анализатор", anchor="w",
                                      command=lambda: self.select_frame_by_name("scan"))
        self.btn_scan.grid(row=1, column=0, padx=20, pady=10)

        self.btn_inject = ctk.CTkButton(self.sidebar, text="💉 Инжектор & Тест", anchor="w",
                                        command=lambda: self.select_frame_by_name("inject"))
        self.btn_inject.grid(row=2, column=0, padx=20, pady=10)

        self.btn_extract = ctk.CTkButton(self.sidebar, text="🔓 Экстрактор", anchor="w",
                                         command=lambda: self.select_frame_by_name("extract"))
        self.btn_extract.grid(row=3, column=0, padx=20, pady=10)

    def select_frame_by_name(self, name):
        self.btn_scan.configure(fg_color=("gray75", "gray25") if name == "scan" else "transparent")
        self.btn_inject.configure(fg_color=("gray75", "gray25") if name == "inject" else "transparent")
        self.btn_extract.configure(fg_color=("gray75", "gray25") if name == "extract" else "transparent")

        if name == "scan":
            self.frame_scan.grid(row=0, column=1, sticky="nsew")
        else:
            self.frame_scan.grid_forget()
        if name == "inject":
            self.frame_inject.grid(row=0, column=1, sticky="nsew")
        else:
            self.frame_inject.grid_forget()
        if name == "extract":
            self.frame_extract.grid(row=0, column=1, sticky="nsew")
        else:
            self.frame_extract.grid_forget()

    def setup_frames(self):
        self.frame_scan = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.frame_inject = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.frame_extract = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")

        self.build_scan_ui()
        self.build_inject_ui()
        self.build_extract_ui()

    # ==================== ВКЛАДКА 1: АНАЛИЗАТОР ====================
    def build_scan_ui(self):
        self.scan_filepath = None

        title = ctk.CTkLabel(self.frame_scan, text="Нейросетевой Детектор LSB",
                             font=ctk.CTkFont(size=24, weight="bold"))
        title.pack(pady=(20, 10))

        # Карточка управления
        control_card = ctk.CTkFrame(self.frame_scan)
        control_card.pack(pady=10, padx=20, fill="x")

        ctk.CTkButton(control_card, text="📁 Выбрать файл", command=self.select_scan_image).pack(side="left", padx=20,
                                                                                                pady=20)
        self.lbl_scan_file = ctk.CTkLabel(control_card, text="Файл не выбран", text_color="gray")
        self.lbl_scan_file.pack(side="left", padx=10)

        self.btn_run_scan = ctk.CTkButton(control_card, text="🚀 Запустить скан", fg_color="#C85A17",
                                          hover_color="#9F400B", command=self.run_scan)
        self.btn_run_scan.pack(side="right", padx=20)

        # Область результатов (две колонки)
        res_container = ctk.CTkFrame(self.frame_scan, fg_color="transparent")
        res_container.pack(fill="both", expand=True, padx=20, pady=10)
        res_container.grid_columnconfigure(0, weight=1)
        res_container.grid_columnconfigure(1, weight=1)

        # Левая: Изображение
        img_card = ctk.CTkFrame(res_container)
        img_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.scan_img_label = ctk.CTkLabel(img_card, text="Превью")
        self.scan_img_label.pack(expand=True, pady=10)

        # Правая: Логи и прогресс
        log_card = ctk.CTkFrame(res_container)
        log_card.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        self.scan_log = ctk.CTkTextbox(log_card, font=ctk.CTkFont(family="Consolas", size=13))
        self.scan_log.pack(fill="both", expand=True, padx=10, pady=10)

    def select_scan_image(self):
        self.scan_filepath = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp")])
        if self.scan_filepath:
            self.lbl_scan_file.configure(text=os.path.basename(self.scan_filepath))
            self.show_preview(self.scan_filepath, self.scan_img_label, 350)

    def run_scan(self):
        if not self.scan_filepath or not self.model: return
        self.scan_log.delete("1.0", "end")
        self.scan_log.insert("end", "[*] Сканирование начато...\n")
        self.update()

        img = Image.open(self.scan_filepath).convert('RGB')
        w, h = img.size
        transform = transforms.ToTensor()
        stride = 256
        n_x, n_y = max(1, w // stride), max(1, h // stride)

        predictions = []
        draw_img = img.copy()
        draw = ImageDraw.Draw(draw_img)
        detected = 0

        with torch.no_grad():
            for y in range(0, n_y * stride, stride):
                for x in range(0, n_x * stride, stride):
                    patch = img.crop((x, y, x + 256, y + 256))
                    if patch.size != (256, 256): patch = patch.resize((256, 256))
                    patch_tensor = transform(patch).unsqueeze(0).to(self.device)

                    stego_prob = F.softmax(self.model(patch_tensor), dim=1)[0][1].item() * 100
                    predictions.append(stego_prob)

                    if stego_prob > 50:
                        detected += 1
                        draw.rectangle([x, y, x + 256, y + 256], outline="red", width=3)

        max_prob = max(predictions) if predictions else 0
        self.scan_log.insert("end", f"[-] Всего блоков: {len(predictions)}\n")
        self.scan_log.insert("end", f"[-] Подозрительных: {detected}\n")
        self.scan_log.insert("end", f"[-] Макс. вероятность: {max_prob:.2f}%\n\n")

        if max_prob > 70 or detected > 0:
            self.scan_log.insert("end", "🚨 ВЕРДИКТ: НАЙДЕНА СТЕГАНОГРАФИЯ", "warning")
            res_path = "temp_scan_map.png"
            draw_img.save(res_path)
            self.show_preview(res_path, self.scan_img_label, 350)
        else:
            self.scan_log.insert("end", "✅ ВЕРДИКТ: ЧИСТО")

    # ==================== ВКЛАДКА 2: ИНЖЕКТОР ====================
    def build_inject_ui(self):
        self.hide_filepath = None

        title = ctk.CTkLabel(self.frame_inject, text="Маскировка & Тест Скрытности",
                             font=ctk.CTkFont(size=24, weight="bold"))
        title.pack(pady=(20, 10))

        # Сетка карточек
        grid = ctk.CTkFrame(self.frame_inject, fg_color="transparent")
        grid.pack(fill="both", expand=True, padx=20)
        grid.grid_columnconfigure(0, weight=1)
        grid.grid_columnconfigure(1, weight=1)

        # Левая колонка: Настройки
        left_card = ctk.CTkFrame(grid)
        left_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        ctk.CTkButton(left_card, text="📁 Выбрать Оригинал", command=self.select_hide_image).pack(pady=15)
        self.lbl_hide_file = ctk.CTkLabel(left_card, text="Файл не выбран", text_color="gray")
        self.lbl_hide_file.pack()

        ctk.CTkLabel(left_card, text="Секретное сообщение:", font=ctk.CTkFont(weight="bold")).pack(pady=(15, 5))
        self.text_input = ctk.CTkTextbox(left_card, height=100)
        self.text_input.pack(fill="x", padx=15)
        self.text_input.bind("<KeyRelease>", self.update_density_stats)

        ctk.CTkLabel(left_card, text="Дозаполнение шумом (Плотность):", font=ctk.CTkFont(weight="bold")).pack(
            pady=(20, 5))
        self.slider_density = ctk.CTkSlider(left_card, from_=0, to=100, command=self.slider_event)
        self.slider_density.pack(fill="x", padx=20)
        self.slider_density.set(0)

        self.lbl_density_stat = ctk.CTkLabel(left_card, text="Текст: 0.00% | Шум: 0% | Итого: 0%", text_color="#00FF00")
        self.lbl_density_stat.pack(pady=5)

        ctk.CTkLabel(left_card, text="Превью генерируемого шума (LSB):", font=ctk.CTkFont(weight="bold", size=12)).pack(
            pady=(10, 0))
        self.noise_preview = ctk.CTkTextbox(left_card, height=60, font=ctk.CTkFont(family="Consolas", size=11),
                                            text_color="gray")
        self.noise_preview.pack(fill="x", padx=15, pady=(5, 15))
        self.noise_preview.insert("1.0", "Измените плотность, чтобы увидеть биты шума...")
        self.noise_preview.configure(state="disabled")  # Делаем окно только для чтения

        ctk.CTkButton(left_card, text="💾 Зашифровать", fg_color="green", hover_color="darkgreen",
                      command=self.hide_data).pack(pady=20)

        # Правая колонка: Нейро-сканер (Тест)
        right_card = ctk.CTkFrame(grid)
        right_card.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

        ctk.CTkLabel(right_card, text="Симуляция Детектора (StegoSentinel)", font=ctk.CTkFont(weight="bold")).pack(
            pady=15)
        ctk.CTkLabel(right_card, text="Нажмите, чтобы проверить, найдет ли ИИ ваш файл\nс выбранными настройками шума.",
                     text_color="gray").pack()

        self.btn_simulate = ctk.CTkButton(right_card, text="🎯 Проверить скрытность", command=self.simulate_detection)
        self.btn_simulate.pack(pady=20)

        self.lbl_sim_result = ctk.CTkLabel(right_card, text="Вероятность обнаружения: --%", font=ctk.CTkFont(size=18))
        self.lbl_sim_result.pack(pady=10)

        self.prob_bar = ctk.CTkProgressBar(right_card, width=300)
        self.prob_bar.pack(pady=10)
        self.prob_bar.set(0)

    def select_hide_image(self):
        self.hide_filepath = filedialog.askopenfilename(filetypes=[("PNG Images", "*.png")])
        if self.hide_filepath:
            self.lbl_hide_file.configure(text=os.path.basename(self.hide_filepath))
            self.update_density_stats(None)

    def slider_event(self, value):
        self.update_density_stats(None)

    def update_density_stats(self, event):
        if not self.hide_filepath: return
        import random

        text = self.text_input.get("1.0", "end-1c")
        text_bits = (len(text.encode('utf-8')) + 5) * 8

        img = Image.open(self.hide_filepath)
        total_capacity = img.size[0] * img.size[1] * 3

        text_density = (text_bits / total_capacity) * 100
        slider_val = self.slider_density.get()

        final_density = max(text_density, slider_val)

        color = "#FF4500" if final_density > 15 else "#00FF00"
        self.lbl_density_stat.configure(
            text=f"Текст: {text_density:.3f}% | Задано: {slider_val:.1f}% | Итого: {final_density:.1f}%",
            text_color=color
        )

        # === ЛОГИКА ОТОБРАЖЕНИЯ ШУМА ===
        self.noise_preview.configure(state="normal")  # Разрешаем редактирование для обновления текста
        self.noise_preview.delete("1.0", "end")

        # Считаем, сколько именно битов шума нам придется сгенерировать
        target_bits = int(total_capacity * (final_density / 100.0))
        noise_bits_count = max(0, target_bits - text_bits)

        if noise_bits_count > 0:
            # Чтобы не зависла программа, генерируем для превью только первые 300 бит
            sample_size = min(noise_bits_count, 300)
            noise_sample = "".join(str(random.randint(0, 1)) for _ in range(sample_size))

            # Дописываем, сколько еще бит останется за кадром
            if noise_bits_count > 300:
                noise_sample += f" ...\n[И еще {noise_bits_count - 300:,} случайных бит]"

            self.noise_preview.insert("1.0", noise_sample)
            self.noise_preview.configure(text_color="#00FF00")  # Делаем цвет матричным/хакерским
        else:
            self.noise_preview.insert("1.0", "Шум не требуется. Ваш текст или файл занимает весь заданный объем.")
            self.noise_preview.configure(text_color="gray")

        self.noise_preview.configure(state="disabled")  # Снова блокируем

    def hide_data(self):
        if not self.hide_filepath: return
        text = self.text_input.get("1.0", "end-1c")
        target_density = self.slider_density.get() / 100.0

        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if save_path:
            embed_lsb(self.hide_filepath, save_path, text, target_density)
            messagebox.showinfo("Готово", "Сообщение и шум успешно внедрены!")

    def simulate_detection(self):
        """Создает временный файл с текущими настройками и сканирует его"""
        if not self.hide_filepath or not self.model: return

        text = self.text_input.get("1.0", "end-1c")
        target_density = self.slider_density.get() / 100.0

        # Создаем временный файл
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            temp_path = tmp.name

        embed_lsb(self.hide_filepath, temp_path, text, target_density)

        # Быстрый скан
        img = Image.open(temp_path).convert('RGB')
        transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
        tensor = transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            prob = F.softmax(self.model(tensor), dim=1)[0][1].item() * 100

        self.lbl_sim_result.configure(text=f"Вероятность обнаружения: {prob:.1f}%")
        self.prob_bar.set(prob / 100.0)
        self.prob_bar.configure(progress_color="red" if prob > 50 else "green")

        os.remove(temp_path)

    # ==================== ВКЛАДКА 3: ЭКСТРАКТОР ====================
    def build_extract_ui(self):
        self.ext_filepath = None

        title = ctk.CTkLabel(self.frame_extract, text="Криминалистический Экстрактор",
                             font=ctk.CTkFont(size=24, weight="bold"))
        title.pack(pady=(20, 10))

        card = ctk.CTkFrame(self.frame_extract)
        card.pack(fill="both", expand=True, padx=40, pady=20)

        ctk.CTkButton(card, text="📁 Загрузить улику (Стего-фото)", command=self.select_ext_image).pack(pady=20)
        self.lbl_ext_file = ctk.CTkLabel(card, text="Файл не выбран", text_color="gray")
        self.lbl_ext_file.pack()

        ctk.CTkButton(card, text="🔓 Извлечь скрытый текст", fg_color="#E49B0F", hover_color="#C5850A",
                      text_color="black", command=self.extract_data).pack(pady=20)

        ctk.CTkLabel(card, text="Извлеченные данные:", font=ctk.CTkFont(weight="bold")).pack(pady=(10, 0))
        self.ext_result = ctk.CTkTextbox(card, font=ctk.CTkFont(size=14))
        self.ext_result.pack(fill="both", expand=True, padx=20, pady=20)

    def select_ext_image(self):
        self.ext_filepath = filedialog.askopenfilename(filetypes=[("PNG Images", "*.png")])
        if self.ext_filepath:
            self.lbl_ext_file.configure(text=os.path.basename(self.ext_filepath))

    def extract_data(self):
        if not self.ext_filepath: return
        self.ext_result.delete("1.0", "end")
        result = extract_lsb(self.ext_filepath)
        self.ext_result.insert("end", result)

    # ==================== УТИЛИТЫ ====================
    def show_preview(self, path, label_widget, size):
        img = Image.open(path)
        img.thumbnail((size, size))
        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
        label_widget.configure(image=ctk_img, text="")
        label_widget.image = ctk_img


if __name__ == "__main__":
    app = StegoApp()
    app.mainloop()