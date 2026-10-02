#!/usr/bin/env python3

import os
import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image, ImageDraw, ImageTk


class CropTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Outil de Crop - Images")
        self.root.geometry("1400x900")

        self.image_path = None
        self.original_image = None
        self.rotated_image = None
        self.display_image = None
        self.photo_image = None
        self.canvas_image = None

        self.start_x = None
        self.start_y = None
        self.current_rect = None
        self.crop_box = None

        self.scale_factor = 1.0
        self.zoom_level = 1.0
        self.image_width = 0
        self.image_height = 0
        self.rotation_angle = 0.0

        self.folder_path = None
        self.image_list = []
        self.current_index = 0

        self.setup_ui()
        self.display_image_on_canvas()

    def setup_ui(self):
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_frame = tk.Frame(main_frame, width=300, bg='#f0f0f0')
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_frame.pack_propagate(False)

        title = tk.Label(left_frame, text="Outil de Crop", font=('Arial', 16, 'bold'), bg='#f0f0f0')
        title.pack(pady=10)

        btn_load_folder = tk.Button(left_frame, text="Charger Dossier", command=self.load_folder,
                                   font=('Arial', 12, 'bold'), bg='#4CAF50', fg='white', padx=20, pady=10)
        btn_load_folder.pack(pady=10, padx=10, fill=tk.X)

        btn_load_image = tk.Button(left_frame, text="Charger Image", command=self.load_image,
                                  font=('Arial', 12, 'bold'), bg='#2196F3', fg='white', padx=20, pady=10)
        btn_load_image.pack(pady=10, padx=10, fill=tk.X)

        self.info_label = tk.Label(left_frame, text="", font=('Arial', 9), bg='#f0f0f0', fg='#666', wraplength=280)
        self.info_label.pack(pady=10, padx=10, fill=tk.X)

        tk.Frame(left_frame, height=2, bg='gray').pack(fill=tk.X, pady=15, padx=10)

        instructions = tk.Label(left_frame, text="Instructions:\n\n1. Sélectionnez un dossier\n2. Dessinez un rectangle sur l'image\n3. Cliquez 'Cropper'\n4. Naviguez vers l'image suivante",
                              font=('Arial', 9), bg='#f0f0f0', justify=tk.LEFT)
        instructions.pack(pady=10, padx=10, fill=tk.X)

        tk.Frame(left_frame, height=2, bg='gray').pack(fill=tk.X, pady=15, padx=10)

        rotation_label = tk.Label(left_frame, text="Rotation:", font=('Arial', 11, 'bold'), bg='#f0f0f0')
        rotation_label.pack(pady=(10, 5))

        rotation_btn_frame = tk.Frame(left_frame, bg='#f0f0f0')
        rotation_btn_frame.pack(padx=10, fill=tk.X, pady=5)

        btn_rot_left = tk.Button(rotation_btn_frame, text="↺ -90°", command=self.rotate_left,
                                font=('Arial', 9), bg='#FF6B6B', fg='white', padx=3, pady=3)
        btn_rot_left.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        btn_rot_right = tk.Button(rotation_btn_frame, text="↻ +90°", command=self.rotate_right,
                                 font=('Arial', 9), bg='#FF6B6B', fg='white', padx=3, pady=3)
        btn_rot_right.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        btn_rot_reset = tk.Button(rotation_btn_frame, text="↻ Reset", command=self.rotation_reset,
                                 font=('Arial', 9), bg='#FF6B6B', fg='white', padx=3, pady=3)
        btn_rot_reset.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        fine_rotation_frame = tk.Frame(left_frame, bg='#f0f0f0')
        fine_rotation_frame.pack(padx=10, fill=tk.X, pady=3)

        btn_rot_fine_left = tk.Button(fine_rotation_frame, text="◄ -1°", command=self.rotate_fine_left_1,
                                     font=('Arial', 8), bg='#FFA726', fg='white', padx=2, pady=2)
        btn_rot_fine_left.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)

        btn_rot_fine_left_small = tk.Button(fine_rotation_frame, text="◄◄ -0.5°", command=self.rotate_fine_left_05,
                                           font=('Arial', 8), bg='#FFCA28', fg='black', padx=2, pady=2)
        btn_rot_fine_left_small.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)

        btn_rot_fine_right_small = tk.Button(fine_rotation_frame, text="+0.5° ►►", command=self.rotate_fine_right_05,
                                            font=('Arial', 8), bg='#FFCA28', fg='black', padx=2, pady=2)
        btn_rot_fine_right_small.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)

        btn_rot_fine_right = tk.Button(fine_rotation_frame, text="+1° ►", command=self.rotate_fine_right_1,
                                      font=('Arial', 8), bg='#FFA726', fg='white', padx=2, pady=2)
        btn_rot_fine_right.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)

        angle_frame = tk.Frame(left_frame, bg='#f0f0f0')
        angle_frame.pack(padx=10, fill=tk.X, pady=5)

        self.angle_label = tk.Label(angle_frame, text="0.0°", font=('Arial', 10), bg='#f0f0f0', width=6)
        self.angle_label.pack(side=tk.LEFT, padx=5)

        self.angle_slider = tk.Scale(angle_frame, from_=-180, to=180, orient=tk.HORIZONTAL,
                                    command=self.on_angle_slider_change, bg='#f0f0f0', fg='#333',
                                    resolution=0.1)
        self.angle_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        angle_input_frame = tk.Frame(left_frame, bg='#f0f0f0')
        angle_input_frame.pack(padx=10, fill=tk.X, pady=5)

        tk.Label(angle_input_frame, text="Angle:", font=('Arial', 9), bg='#f0f0f0').pack(side=tk.LEFT, padx=5)
        
        self.angle_entry = tk.Entry(angle_input_frame, width=8, font=('Arial', 10))
        self.angle_entry.pack(side=tk.LEFT, padx=2)
        self.angle_entry.insert(0, "0.0")
        self.angle_entry.bind("<Return>", self.on_angle_entry_change)

        btn_apply_angle = tk.Button(angle_input_frame, text="Appliquer", command=self.apply_angle_from_entry,
                                   font=('Arial', 9), bg='#2196F3', fg='white', padx=5, pady=2)
        btn_apply_angle.pack(side=tk.LEFT, padx=2)

        btn_apply_rotation = tk.Button(left_frame, text="✓ Appliquer Rotation", command=self.apply_rotation_permanent,
                                      font=('Arial', 10), bg='#4CAF50', fg='white', padx=10, pady=5)
        btn_apply_rotation.pack(pady=5, padx=10, fill=tk.X)

        tk.Frame(left_frame, height=2, bg='gray').pack(fill=tk.X, pady=15, padx=10)

        coords_label = tk.Label(left_frame, text="Coordonnées:", font=('Arial', 11, 'bold'), bg='#f0f0f0')
        coords_label.pack(pady=(10, 5))

        self.coords_label = tk.Label(left_frame, text="Aucune sélection", font=('Arial', 9), bg='#f0f0f0', fg='#333')
        self.coords_label.pack(pady=5, padx=10, fill=tk.X)

        btn_crop = tk.Button(left_frame, text="CROPPER", command=self.crop_image,
                            font=('Arial', 13, 'bold'), bg='#FF9800', fg='white',
                            padx=20, pady=12, relief=tk.RAISED, borderwidth=3)
        btn_crop.pack(pady=15, padx=10, fill=tk.X)

        btn_clear = tk.Button(left_frame, text="Annuler Sélection", command=self.clear_selection,
                             font=('Arial', 10), bg='#f44336', fg='white', padx=10, pady=5)
        btn_clear.pack(pady=5, padx=10, fill=tk.X)

        tk.Frame(left_frame, height=2, bg='gray').pack(fill=tk.X, pady=15, padx=10)

        nav_label = tk.Label(left_frame, text="Navigation:", font=('Arial', 11, 'bold'), bg='#f0f0f0')
        nav_label.pack(pady=(10, 5))

        nav_frame = tk.Frame(left_frame, bg='#f0f0f0')
        nav_frame.pack(padx=10, fill=tk.X, pady=5)

        btn_prev = tk.Button(nav_frame, text="◀ Précédent", command=self.prev_image,
                            font=('Arial', 10), bg='#9E9E9E', fg='white', padx=5, pady=5)
        btn_prev.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        btn_next = tk.Button(nav_frame, text="Suivant ▶", command=self.next_image,
                            font=('Arial', 10), bg='#9E9E9E', fg='white', padx=5, pady=5)
        btn_next.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        self.counter_label = tk.Label(left_frame, text="", font=('Arial', 9), bg='#f0f0f0', fg='#666')
        self.counter_label.pack(pady=5, padx=10, fill=tk.X)

        tk.Frame(left_frame, height=2, bg='gray').pack(fill=tk.X, pady=15, padx=10)

        zoom_label = tk.Label(left_frame, text="Zoom:", font=('Arial', 11, 'bold'), bg='#f0f0f0')
        zoom_label.pack(pady=(10, 5))

        zoom_frame = tk.Frame(left_frame, bg='#f0f0f0')
        zoom_frame.pack(padx=10, fill=tk.X)

        btn_zoom_out = tk.Button(zoom_frame, text="-", command=self.zoom_out,
                                font=('Arial', 14, 'bold'), bg='#9E9E9E', fg='white', width=3)
        btn_zoom_out.pack(side=tk.LEFT, padx=2)

        self.zoom_label_display = tk.Label(zoom_frame, text="100%", font=('Arial', 11), bg='#f0f0f0', width=8)
        self.zoom_label_display.pack(side=tk.LEFT, padx=5)

        btn_zoom_in = tk.Button(zoom_frame, text="+", command=self.zoom_in,
                               font=('Arial', 14, 'bold'), bg='#9E9E9E', fg='white', width=3)
        btn_zoom_in.pack(side=tk.LEFT, padx=2)

        btn_zoom_reset = tk.Button(left_frame, text="Réinitialiser Zoom", command=self.zoom_reset, font=('Arial', 9),
                                   bg='#9E9E9E', fg='white', padx=5, pady=3)
        btn_zoom_reset.pack(pady=5, padx=10, fill=tk.X)

        canvas_frame = tk.Frame(main_frame, bg='white')
        canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_frame, bg='gray', cursor="crosshair", width=800, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Motion>", self.on_mouse_move)

        self.root.bind("<Control-s>", lambda e: self.crop_image())
        self.root.bind("<Right>", lambda e: self.next_image())
        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("<r>", lambda e: self.rotate_right())
        self.root.bind("<l>", lambda e: self.rotate_left())
        self.root.bind("<0>", lambda e: self.rotation_reset())

    def load_folder(self):
        folder = filedialog.askdirectory(title="Sélectionnez un dossier contenant des images")
        if not folder:
            return
        self.load_folder_directly(folder)

    def load_folder_directly(self, folder):
        self.folder_path = folder
        self.image_list = []
        self.current_index = 0

        extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        for root, dirs, files in os.walk(folder):
            for file in files:
                file_lower = file.lower()
                if any(file_lower.endswith(ext) for ext in extensions):
                    file_path = os.path.join(root, file)
                    self.image_list.append(file_path)

        self.image_list = sorted(self.image_list)

        if not self.image_list:
            messagebox.showerror("Erreur", "Aucune image trouvée dans ce dossier")
            self.info_label.config(text="Aucune image trouvée", fg='red')
            return

        self.info_label.config(text=f"{len(self.image_list)} images chargées", fg='green')
        self.load_image_by_index(0)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Sélectionnez une image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("Tous", "*.*")]
        )
        if not file_path:
            return

        self.image_path = file_path
        self.load_and_display_image()

    def load_image_by_index(self, index):
        if 0 <= index < len(self.image_list):
            self.current_index = index
            self.image_path = str(self.image_list[index])
            self.load_and_display_image()

    def load_and_display_image(self):
        try:
            self.original_image = Image.open(self.image_path)
            self.image_width, self.image_height = self.original_image.size
            
            canvas_width = 800
            canvas_height = 600
            
            scale_x = canvas_width / self.image_width
            scale_y = canvas_height / self.image_height
            self.zoom_level = min(scale_x, scale_y, 1.0)
            
            self.zoom_label_display.config(text=f"{int(self.zoom_level * 100)}%")
            self.crop_box = None
            self.rotation_angle = 0.0
            self.angle_slider.set(0)
            self.rotated_image = None
            self.clear_selection()
            self.display_image_on_canvas()
            self.update_info()
            
            self.root.after(100, self.optimize_zoom)
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de charger l'image: {e}")

    def optimize_zoom(self):
        """Recalcule le zoom optimal basé sur les vraies dimensions du canvas"""
        if self.original_image:
            safe_zoom = self.get_safe_zoom_level()
            if safe_zoom < self.zoom_level:
                self.zoom_level = safe_zoom
                self.zoom_label_display.config(text=f"{int(self.zoom_level * 100)}%")
                self.display_image_on_canvas()

    def rotate_left(self):
        """Rotation de -90 degrés"""
        self.rotation_angle = (self.rotation_angle - 90) % 360
        self.angle_slider.set(self.rotation_angle)

    def rotate_right(self):
        """Rotation de +90 degrés"""
        self.rotation_angle = (self.rotation_angle + 90) % 360
        self.angle_slider.set(self.rotation_angle)

    def rotation_reset(self):
        """Réinitialise la rotation"""
        self.rotation_angle = 0.0
        self.angle_slider.set(0)
        self.angle_entry.delete(0, tk.END)
        self.angle_entry.insert(0, "0.0")
        self.display_image_on_canvas()

    def rotate_fine_left_1(self):
        """Rotation fine de -1 degré"""
        self.rotation_angle = max(-180, self.rotation_angle - 1.0)
        self.angle_slider.set(self.rotation_angle)

    def rotate_fine_left_05(self):
        """Rotation fine de -0.5 degrés"""
        self.rotation_angle = max(-180, self.rotation_angle - 0.5)
        self.angle_slider.set(self.rotation_angle)

    def rotate_fine_right_05(self):
        """Rotation fine de +0.5 degrés"""
        self.rotation_angle = min(180, self.rotation_angle + 0.5)
        self.angle_slider.set(self.rotation_angle)

    def rotate_fine_right_1(self):
        """Rotation fine de +1 degré"""
        self.rotation_angle = min(180, self.rotation_angle + 1.0)
        self.angle_slider.set(self.rotation_angle)

    def get_safe_zoom_level(self):
        """Calcule un zoom qui garantit que l'image rentre dans le canvas"""
        if not self.original_image:
            return 1.0
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1:
            canvas_width = 800
        if canvas_height <= 1:
            canvas_height = 600
        
        canvas_width -= 20
        canvas_height -= 20
        
        scale_x = canvas_width / self.image_width
        scale_y = canvas_height / self.image_height
        
        return min(scale_x, scale_y, 1.0)

    def on_angle_slider_change(self, value):
        """Appelé quand le slider est modifié"""
        try:
            self.rotation_angle = float(value)
            self.angle_label.config(text=f"{self.rotation_angle:.1f}°")
            self.angle_entry.delete(0, tk.END)
            self.angle_entry.insert(0, f"{self.rotation_angle:.1f}")
            self.display_image_on_canvas()
        except (ValueError, tk.TclError):
            pass

    def on_angle_entry_change(self, event):
        """Appelé quand l'utilisateur appuie sur Entrée dans le champ texte"""
        self.apply_angle_from_entry()

    def apply_angle_from_entry(self):
        """Applique l'angle saisi dans le champ texte"""
        try:
            angle = float(self.angle_entry.get())
            angle = max(-180, min(180, angle))
            self.rotation_angle = angle
            self.angle_slider.set(angle)
            self.angle_label.config(text=f"{angle:.1f}°")
            self.display_image_on_canvas()
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer un nombre valide (ex: 45.5)")
            self.angle_entry.delete(0, tk.END)
            self.angle_entry.insert(0, f"{self.rotation_angle:.1f}")

    def apply_rotation_permanent(self):
        """Applique la rotation de manière permanente à l'image"""
        if not self.original_image:
            messagebox.showwarning("Attention", "Chargez une image d'abord")
            return
        
        if self.rotation_angle == 0:
            messagebox.showinfo("Info", "Aucune rotation à appliquer")
            return
        
        try:
            self.original_image = self.original_image.rotate(-self.rotation_angle, expand=True, resample=Image.Resampling.BICUBIC)
            self.image_width, self.image_height = self.original_image.size
            self.rotation_angle = 0.0
            self.angle_slider.set(0)
            self.crop_box = None
            self.display_image_on_canvas()
            messagebox.showinfo("Succès", "Rotation appliquée!")
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de rotationner: {e}")

    def display_image_on_canvas(self):
        if not self.original_image:
            self.canvas.delete("all")
            self.canvas.create_text(400, 300, text="Chargez une image pour commencer", 
                                   font=("Arial", 20), fill="gray", anchor="center")
            return

        working_image = self.original_image
        if self.rotation_angle != 0:
            working_image = self.original_image.rotate(-self.rotation_angle, expand=True, resample=Image.Resampling.BICUBIC)
        
        img_width, img_height = working_image.size
        
        current_zoom = self.zoom_level
        
        min_zoom = self.get_safe_zoom_level()
        if current_zoom < min_zoom:
            display_zoom = min_zoom
        else:
            display_zoom = current_zoom
        
        display_width = int(img_width * display_zoom)
        display_height = int(img_height * display_zoom)

        self.display_image = working_image.resize((display_width, display_height), Image.Resampling.LANCZOS)
        self.scale_factor = display_zoom

        if self.crop_box:
            draw_img = self.display_image.copy()
            draw = ImageDraw.Draw(draw_img, 'RGBA')
            scaled_box = (
                int(self.crop_box[0] * display_zoom),
                int(self.crop_box[1] * display_zoom),
                int(self.crop_box[2] * display_zoom),
                int(self.crop_box[3] * display_zoom)
            )
            draw.rectangle(scaled_box, outline='red', width=3)
            draw.rectangle([x+1 for x in scaled_box], outline='yellow', width=1)
            self.photo_image = ImageTk.PhotoImage(draw_img)
        else:
            self.photo_image = ImageTk.PhotoImage(self.display_image)

        self.canvas.delete("all")
        self.canvas_image = self.canvas.create_image(0, 0, image=self.photo_image, anchor="nw")

    def on_mouse_down(self, event):
        self.start_x = max(0, int(event.x / self.scale_factor))
        self.start_y = max(0, int(event.y / self.scale_factor))

    def on_mouse_drag(self, event):
        if self.start_x is None or self.start_y is None:
            return

        current_x = max(0, int(event.x / self.scale_factor))
        current_y = max(0, int(event.y / self.scale_factor))

        working_image = self.original_image
        if self.rotation_angle != 0:
            working_image = self.original_image.rotate(-self.rotation_angle, expand=True, resample=Image.Resampling.BICUBIC)
        
        img_width, img_height = working_image.size
        
        current_x = min(current_x, img_width)
        current_y = min(current_y, img_height)

        left = min(self.start_x, current_x)
        top = min(self.start_y, current_y)
        right = max(self.start_x, current_x)
        bottom = max(self.start_y, current_y)

        if right > left and bottom > top:
            self.crop_box = (left, top, right, bottom)
            self.update_coords_label()
            self.display_image_on_canvas()

    def on_mouse_up(self, event):
        pass

    def on_mouse_move(self, event):
        pass

    def update_coords_label(self):
        if self.crop_box:
            left, top, right, bottom = self.crop_box
            width = right - left
            height = bottom - top
            self.coords_label.config(text=f"L:{left} T:{top} R:{right} B:{bottom}\n{width}×{height}px")
        else:
            self.coords_label.config(text="Aucune sélection")

    def clear_selection(self):
        self.crop_box = None
        self.start_x = None
        self.start_y = None
        self.update_coords_label()
        self.display_image_on_canvas()

    def crop_image(self):
        if not self.image_path or not self.crop_box:
            messagebox.showwarning("Attention", "Veuillez sélectionner une région à cropper")
            return

        try:
            image_to_crop = self.original_image
            if self.rotation_angle != 0:
                image_to_crop = self.original_image.rotate(-self.rotation_angle, expand=True, resample=Image.Resampling.BICUBIC)
            
            cropped = image_to_crop.crop(self.crop_box)
            if cropped.mode == 'RGBA':
                cropped = cropped.convert('RGB')
            cropped.save(self.image_path)
            messagebox.showinfo("Succès", f"Image croppée et sauvegardée:\n{self.image_path}")
            
            self.original_image = Image.open(self.image_path)
            self.image_width, self.image_height = self.original_image.size
            self.crop_box = None
            self.rotation_angle = 0.0
            self.angle_slider.set(0)
            self.display_image_on_canvas()
            self.update_info()
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de cropper l'image: {e}")

    def next_image(self):
        if self.image_list:
            self.load_image_by_index((self.current_index + 1) % len(self.image_list))

    def prev_image(self):
        if self.image_list:
            self.load_image_by_index((self.current_index - 1) % len(self.image_list))

    def zoom_in(self):
        new_zoom = self.zoom_level + 0.1
        self.zoom_level = min(new_zoom, 3.0)
        self.zoom_label_display.config(text=f"{int(self.zoom_level * 100)}%")
        self.display_image_on_canvas()

    def zoom_out(self):
        min_zoom = self.get_safe_zoom_level()
        new_zoom = self.zoom_level - 0.1
        self.zoom_level = max(new_zoom, min_zoom)
        self.zoom_label_display.config(text=f"{int(self.zoom_level * 100)}%")
        self.display_image_on_canvas()

    def zoom_reset(self):
        self.zoom_level = self.get_safe_zoom_level()
        self.zoom_label_display.config(text=f"{int(self.zoom_level * 100)}%")
        self.display_image_on_canvas()

    def update_info(self):
        if self.image_path:
            filename = os.path.basename(self.image_path)
            size = f"{self.image_width}×{self.image_height}px"
            counter = f"{self.current_index + 1}/{len(self.image_list)}" if self.image_list else ""
            self.counter_label.config(text=counter)
            self.info_label.config(text=f"📄 {filename}\n{size}")

if __name__ == "__main__":
    import sys
    root = tk.Tk()
    app = CropTool(root)
    
    if len(sys.argv) > 1:
        folder_arg = sys.argv[1]
        if os.path.isdir(folder_arg):
            root.after(500, lambda: app.load_folder_directly(folder_arg))
    
    root.mainloop()
