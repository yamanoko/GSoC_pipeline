import os
import glob
from PIL import Image

import tkinter as tk
import customtkinter

from backend import inference

FONT = ("Arial", 12)

class App(customtkinter.CTk):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.fonts = FONT
		self.input_image_list = []
		self.count = 0
		self.setup_form()
	
	def setup_form(self):
		customtkinter.set_appearance_mode("light")
		customtkinter.set_default_color_theme("blue")

		self.geometry("800x600")
		self.title("Directory Selection")

		self.grid_rowconfigure(3, weight=1)
		self.grid_columnconfigure((0, 1), weight=1)

		self.input_selection_frame = DirectorySelectionFrame(self, header_name="Select Input Directory")
		self.input_selection_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew", columnspan=2)

		self.output_selection_frame = DirectorySelectionFrame(self, header_name="Select Output Directory")
		self.output_selection_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew", columnspan=2)

		self.load_button = customtkinter.CTkButton(self, text="Load Images", command=self.load_images, font=self.fonts)
		self.load_button.grid(row=2, column=1, padx=10, pady=10, sticky="ew")

		self.process_button = customtkinter.CTkButton(self, text="Process Images", command=self.process_images, font=self.fonts)
		self.process_button.grid(row=2, column=0, padx=10, pady=10, sticky="ew")

		self.displayed_image = None
		self.image_label = customtkinter.CTkLabel(self, text="An image is not loaded", image=self.displayed_image, font=self.fonts)
		self.image_label.grid(row=3, column=0, padx=10, pady=10, sticky="nsew")

		self.text_output_frame = customtkinter.CTkTextbox(self, width=200, height=300, font=self.fonts)
		self.text_output_frame.grid(row=3, column=1, padx=10, pady=10, sticky="nsew")

		self.page_switch_slider = customtkinter.CTkSlider(self, from_=1, to=10, orientation="horizontal", number_of_steps=10, command=self.switch_image)
		self.page_switch_slider.set(1)
		self.page_switch_slider.grid(row=4, column=0, padx=10, pady=10, sticky="ew", columnspan=2)

	def _get_input_image_list(self):
		directory = self.input_selection_frame.folder_path
		if directory is None or not os.path.exists(directory):
			return
		self.input_image_list = glob.glob(os.path.join(directory, '*.png')) + glob.glob(os.path.join(directory, '*.jpg'))
	
	def _display_image(self):
		if not self.input_image_list:
			return
		image = Image.open(self.input_image_list[self.count])
		self.displayed_image = customtkinter.CTkImage(light_image=image, dark_image=image, size=(400, 400))
		self.image_label.configure(text="", image=self.displayed_image)
	
	def _display_text(self):
		output_folder = self.output_selection_frame.folder_path
		if output_folder is None:
			return
		txt_file_path = os.path.join(output_folder, os.path.basename(self.input_image_list[self.count]).split(".")[0] + ".txt")
		if os.path.isfile(txt_file_path):
			with open(txt_file_path, "r", encoding="utf-8") as f:
				self.text_output_frame.delete("0.0", "end")
				self.text_output_frame.insert("0.0", f.read())

	def load_images(self):
		self._get_input_image_list()
		self.page_switch_slider.configure(from_=1, to=len(self.input_image_list), number_of_steps=len(self.input_image_list)-1)
		self.page_switch_slider.set(1)
		self.count = 0
		self._display_image()
	
	def process_images(self):
		self.load_images()
		inference(input_files=self.input_image_list, output_dir=self.output_selection_frame.folder_path)
		self._display_text()

	def switch_image(self, value):
		self.count = int(value) - 1
		self._display_image()
		self._display_text()

class DirectorySelectionFrame(customtkinter.CTkFrame):
	def __init__(self, *args, header_name="Select Directory", **kwargs):
		super().__init__(*args, **kwargs)
		self.fonts = FONT
		self.header_name = header_name
		self.folder_path = None
		self.setup_form()
	
	def setup_form(self):
		self.grid_rowconfigure(0, weight=1)
		self.grid_columnconfigure(0, weight=1)
		self.label = customtkinter.CTkLabel(self, text=self.header_name, font=self.fonts)
		self.label.grid(row=0, column=0, padx=20, sticky="w")
		self.textbox = customtkinter.CTkEntry(self, placeholder_text="Directory Path", width=120, font=self.fonts)
		self.textbox.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="ew")
		self.browse_button = customtkinter.CTkButton(self, text="Browse", command=self.browse_directory, font=self.fonts)
		self.browse_button.grid(row=1, column=1, padx=10, pady=(0, 10))
	
	def browse_directory(self):
		selection = tk.filedialog.askdirectory()
		if selection is None:
			return
		self.folder_path = selection
		self.textbox.delete(0, tk.END)
		self.textbox.insert(0, self.folder_path)

if __name__ == "__main__":
	app = App()
	app.mainloop()