import torch
import torch.nn as nn
from Classifier import evaluate_model, load_dataset, SimpleCNN, predict_inference_single
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk


class CancerIdentifier(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.tk_image = None
        self.model = SimpleCNN()
        self.model_init()
        self.geometry("400x400")  # Adjusted window size to accommodate the image

        # Create a button to load an image
        self.load_button = ctk.CTkButton(self, text="Завантажити зображення", command=self.load_image)
        self.load_button.pack(pady=10)

        # Create a button to start calculation
        self.calculate_button = ctk.CTkButton(self, text="Визначити тип раку", command=self.calculate)
        self.calculate_button.pack(pady=10)

        # Create a button to close the window
        self.close_button = ctk.CTkButton(self, text="Закрити вікно", command=self.close_window)
        self.close_button.pack(pady=10)

        # Create a label to display the image
        self.image_label = ctk.CTkLabel(self)
        self.image_label.pack(pady=20)
        self.image_label.configure(text="")

        # Variable to store the image path
        self.image_path = None
        self.image = None

    def model_init(self):
        torch.manual_seed(42)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Using ", device)
        criterion = nn.CrossEntropyLoss()
        base_dir = 'PatientData'
        image_size = (512, 512)
        split_percentage = 0.8
        train_loader, val_loader, inference_loader = load_dataset(base_dir, image_size, split_percentage)
        self.model.load_state_dict(torch.load('Model5.pth'))
        val_loss, val_accuracy = evaluate_model(self.model, val_loader, criterion, device)
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

    def load_image(self):
        # Open a file dialog to select an image
        self.image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.tif;*.tiff")])
        if self.image_path:
            self.display_image()
            self.image = Image.open(self.image_path)

    def display_image(self):
        # Load the image using PIL
        pil_image = Image.open(self.image_path)
        pil_image = pil_image.resize((512, 512), Image.Resampling.LANCZOS)  # Resize image to 512x512
        self.tk_image = ImageTk.PhotoImage(pil_image)

        # Display the image in the label
        self.image_label.configure(image=self.tk_image)
        self.image_label.image = self.tk_image

    def calculate(self):
        if not self.image_path:
            messagebox.showerror("Помилка", "Відсутнє зображення")
            return
        cancers = {
            "0": "Низькодиференційована аденокарцинома",
            "1": "Бронхогенна диференційована плоскоклітинна карцинома",
            "2": "Інвазивна аденокарцинома змішаної будови-муцинозної та колоїдної",
            "3": "Плоскоклітинна карцинома без ороговіння",
            "4": "Первинна плоско клітинна карцинома",
            "5": "Плеоморфна веретено клітинна та гігантоклітинна карцинома",
            "6": "Бронхогенна плоско клітинна карцинома з ороговінням",
            "7": "Інвазивна аденокарцинома змішаної гістологічної структури",
            "8": "Карциноїд",
            "9": "Протокова карцинома слинної залози (аденокарцинома)",
            "10": "Низькодиференційована плеоморфна веретено клітинна карцинома",
            "11": "Аденоїд-кістозна карцинома",
            "12": "Аденокарцинома колоїдного типу",
            "13": "Плоскоклітинна карцинома із ороговінням"
        }

        predicted = predict_inference_single(self.model, self.image)
        # Placeholder for calculation logic
        result = cancers.get(str(predicted))

        # Show the result in a popup window
        messagebox.showinfo("Тип раку\n", result)

    def close_window(self):
        # Function to close the application
        self.destroy()


if __name__ == '__main__':
    app = CancerIdentifier()
    app.mainloop()
