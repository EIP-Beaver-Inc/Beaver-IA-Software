#!/usr/bin/env python3

import json
import pathlib
import os
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

from PIL import Image, ImageDraw, ImageTk


class WoodDefectAnnotator:
    """Editeur d'annotations, utilisable seul ou monte dans une autre fenetre.

    `parent` peut etre une fenetre (Tk, Toplevel) ou n'importe quel conteneur.
    Dans le second cas on ne touche ni au titre ni a la geometrie : c'est ce qui
    permet au poste de travail BEAVER de l'afficher dans sa propre zone, sans
    ouvrir de seconde fenetre.
    """

    def __init__(self, root):
        self.root = root
        self.toplevel = root.winfo_toplevel()
        self.autonome = isinstance(root, (tk.Tk, tk.Toplevel))
        if self.autonome:
            self.toplevel.title("Outil d'Annotation - Défauts du Bois")
            self.toplevel.geometry("1400x900")
            # En dessous, la colonne de gauche prend tout et il ne reste plus
            # de place pour l'image. Elle defile, donc rien n'est perdu, mais
            # annoter sur un timbre-poste n'a pas de sens.
            self.toplevel.minsize(900, 560)

        self.classes = {
            0: "Blue_Stain",
            1: "Crack",
            2: "Dead_Knot",
            3: "Knot_missing",
            4: "Live_Knot",
            5: "Marrow",
            6: "Quartzity",
            7: "knot_with_crack",
            8: "resin",
            9: "patte_de_chat",
            10: "aubier_altere",
            11: "poche_ecorce",
            12: "noeud_pourri",
            13: "pourriture"
        }

        #: Ce qui s'affiche a l'ecran. Les noms de `self.classes` partent dans
        #: les annotations enregistrees, puis dans le dataset YOLO et dans le
        #: modele : les traduire casserait tout ce qui est deja annote. On
        #: traduit donc l'affichage seul, et le disque garde ses noms.
        #:
        #: Les termes sont ceux du glossaire EN 1310 deja employe dans le
        #: depot (EN975_NOTES de build_external_dataset.py), pour qu'un nom
        #: designe la meme chose d'un bout a l'autre du projet.
        self.libelles = {
            0: "Bleuissement",
            1: "Fente",
            2: "Nœud mort",
            3: "Nœud sauté",
            4: "Nœud sain",
            5: "Moelle",
            # Terme du jeu Kodytek sans equivalent francais etabli, et hors
            # norme EN 975-1. Le laisser tel quel vaut mieux qu'en inventer un.
            6: "Quartzity",
            7: "Nœud fendu",
            8: "Poche de résine",
            9: "Patte de chat",
            10: "Aubier altéré",
            11: "Poche d'écorce",
            12: "Nœud pourri",
            13: "Pourriture",
        }

        self.colors = {
            0: "#0000FF",
            1: "#FF0000",
            2: "#8B4513",
            3: "#FFD700",
            4: "#00FF00",
            5: "#FF1493",
            6: "#808080",
            7: "#FFA500",
            8: "#800080",
            9: "#00CED1",
            10: "#D2B48C",
            11: "#6B4423",
            12: "#4B0082",
            13: "#556B2F"
        }

        self.image_path = None
        self.original_image = None
        self.display_image = None
        self.photo_image = None
        self.canvas_image = None

        self.annotations = []
        self.current_box = None
        self.start_x = None
        self.start_y = None
        self.current_rect = None
        self.selected_class = 4

        self.scale_factor = 1.0
        self.zoom_level = 1.0
        self.image_width = 0
        self.image_height = 0

        self.pan_start_x = None
        self.pan_start_y = None
        self.pan_data = {'x': 0, 'y': 0}

        self.folder_path = None
        self.image_list = []
        self.current_index = 0

        self.model = None
        self.model_path = None
        self.model_conf = 0.25
        self.auto_predict = False
        self.selected_index = None

        # L'aubier sain ne se cadre pas : il suit les cernes, donc sa forme est
        # irreguliere. La norme n'en demande que la presence et la FACE — pas
        # la rive : X s'il est sur une face, XX sur les deux. Avec une seule
        # camera on ne voit qu'une face : on constate donc la presence, et le
        # moteur signale que XX reste indecidable sans la seconde vue.
        self.aubier = tk.StringVar(value="aucun")

        #: Vrai des que l'operateur a touche a l'image en cours. C'est ce qui
        #: distingue une planche examinee et saine d'une planche seulement
        #: survolee : la premiere merite d'etre enregistree, la seconde non.
        self.modifiee = False

        self.setup_ui()

    def _colonne_defilante(self, parent):
        """Colonne de gauche qui defile, et renvoie le cadre ou tout se range.

        Quatorze classes de defaut, l'aubier, la pre-annotation IA, le zoom,
        la navigation et l'enregistrement ne tiennent pas dans 900 pixels de
        haut : les boutons du bas sortaient de l'ecran sans aucun moyen de les
        atteindre. Le contenu garde donc sa hauteur naturelle et c'est la vue
        qui se deplace dessus.
        """
        colonne = tk.Frame(parent, width=320, bg='#f0f0f0')
        colonne.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        colonne.pack_propagate(False)

        barre = tk.Scrollbar(colonne, orient=tk.VERTICAL)
        barre.pack(side=tk.RIGHT, fill=tk.Y)

        volet = tk.Canvas(colonne, bg='#f0f0f0', highlightthickness=0,
                          yscrollcommand=barre.set)
        volet.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        barre.config(command=volet.yview)

        dedans = tk.Frame(volet, bg='#f0f0f0')
        fenetre = volet.create_window((0, 0), window=dedans, anchor='nw')

        def ajuster(_event=None):
            volet.configure(scrollregion=volet.bbox('all'))
            # Sans cette largeur imposee, le cadre interieur se reduit a son
            # contenu et les boutons cessent de remplir la colonne.
            volet.itemconfigure(fenetre, width=volet.winfo_width())

        dedans.bind('<Configure>', ajuster)
        volet.bind('<Configure>', ajuster)

        def molette(event):
            pas = 1 if getattr(event, 'num', 0) == 5 or event.delta < 0 else -1
            volet.yview_scroll(pas, 'units')

        # La molette n'est captee que sous le pointeur : l'image a sa propre
        # molette pour le zoom, et la lui voler rendrait l'annotation penible.
        def saisir(_event=None):
            for sequence in ('<MouseWheel>', '<Button-4>', '<Button-5>'):
                volet.bind_all(sequence, molette)

        def relacher(_event=None):
            for sequence in ('<MouseWheel>', '<Button-4>', '<Button-5>'):
                volet.unbind_all(sequence)

        colonne.bind('<Enter>', saisir)
        colonne.bind('<Leave>', relacher)
        return dedans

    def setup_ui(self):
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_frame = self._colonne_defilante(main_frame)

        title = tk.Label(left_frame, text="Outil d'Annotation", font=('Arial', 16, 'bold'), bg='#f0f0f0')
        title.pack(pady=10)

        btn_load_folder = tk.Button(left_frame, text="Charger Dossier", command=self.load_folder,
                                   font=('Arial', 12, 'bold'), bg='#4CAF50', fg='white', padx=20, pady=10)
        btn_load_folder.pack(pady=10, padx=10, fill=tk.X)

        btn_load_image = tk.Button(left_frame, text="Charger Image", command=self.load_image,
                                  font=('Arial', 12, 'bold'), bg='#2196F3', fg='white', padx=20, pady=10)
        btn_load_image.pack(pady=10, padx=10, fill=tk.X)

        class_label = tk.Label(left_frame, text="Sélectionner la classe:",
                              font=('Arial', 11, 'bold'), bg='#f0f0f0')
        class_label.pack(pady=(10, 5))

        self.class_buttons_frame = tk.Frame(left_frame, bg='#f0f0f0')
        self.class_buttons_frame.pack(pady=5, padx=10, fill=tk.X)

        self.class_buttons = {}
        for class_id in self.classes:
            btn = tk.Button(self.class_buttons_frame,
                          text=f"{class_id}: {self.libelles[class_id]}",
                          command=lambda cid=class_id: self.select_class(cid),
                          font=('Arial', 9),
                          bg=self.colors[class_id],
                          fg='white' if class_id != 3 else 'black',
                          relief=tk.RAISED,
                          padx=5, pady=5)
            btn.pack(fill=tk.X, pady=2)
            self.class_buttons[class_id] = btn

        self.class_buttons[self.selected_class].config(relief=tk.SUNKEN, borderwidth=3)

        tk.Frame(left_frame, height=2, bg='gray').pack(fill=tk.X, pady=15, padx=10)

        annotations_label = tk.Label(left_frame, text="Annotations:", font=('Arial', 11, 'bold'), bg='#f0f0f0')
        annotations_label.pack(pady=(0, 5))

        list_frame = tk.Frame(left_frame, bg='#f0f0f0')
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.annotations_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, font=('Arial', 9), height=8)
        self.annotations_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.annotations_listbox.yview)

        sain_frame = tk.LabelFrame(left_frame, text="Planche sans défaut",
                                   font=('Arial', 9, 'bold'))
        sain_frame.pack(fill=tk.X, pady=(6, 4))
        tk.Button(sain_frame, text="Aucun défaut sur cette planche",
                  command=self.marquer_sans_defaut,
                  font=('Arial', 9, 'bold'), bg='#3D6E4C', fg='white',
                  pady=6).pack(fill=tk.X, padx=4, pady=(4, 2))
        self.lbl_sans_defaut = tk.Label(sain_frame, text="", font=('Arial', 7),
                                        fg='#666', wraplength=250,
                                        justify=tk.LEFT, anchor="w")
        self.lbl_sans_defaut.pack(fill=tk.X, padx=4, pady=(0, 2))
        tk.Label(sain_frame, font=('Arial', 7), fg='#666', justify=tk.LEFT,
                 wraplength=250, anchor="w",
                 text="Une planche saine est un exemple utile : elle apprend "
                      "au modèle à ne pas inventer de défaut. Sans ce bouton, "
                      "rien ne la distingue d'une planche non regardée."
                 ).pack(fill=tk.X, padx=4, pady=(0, 5))

        aub_frame = tk.LabelFrame(left_frame, text="Aubier sain (suffixe X / XX)",
                                  font=('Arial', 9, 'bold'))
        aub_frame.pack(fill=tk.X, pady=(6, 4))
        for val, lib in (("aucun", "Aucun sur cette face"),
                         ("une_face", "Présent sur cette face  →  X"),
                         ("deux_faces", "Présent sur les 2 faces  →  XX")):
            tk.Radiobutton(aub_frame, text=lib, value=val, variable=self.aubier,
                           font=('Arial', 9), anchor="w",
                           command=self._aubier_change).pack(fill=tk.X, padx=6)
        tk.Label(aub_frame, font=('Arial', 7), fg='#666', justify=tk.LEFT,
                 wraplength=250, anchor="w",
                 text="L'aubier est la bande PÂLE le long d'un bord ; le cœur "
                      "est plus foncé, et la limite suit les cernes donc elle "
                      "est courbe. Planche uniformément foncée = aucun.\n\n"
                      "Avec une seule caméra, cochez « sur cette face » : XX "
                      "ne se vérifie qu'avec la seconde face.\n\n"
                      "Pas de boîte à tracer. Seul l'aubier ALTÉRÉ se cadre."
                 ).pack(fill=tk.X, padx=6, pady=(2, 5))

        ia_frame = tk.LabelFrame(left_frame, text="Pré-annotation IA",
                                 font=('Arial', 9, 'bold'))
        ia_frame.pack(fill=tk.X, pady=(6, 4))

        self.btn_predict = tk.Button(
            ia_frame, text="Pré-annoter  (A)", command=self.predict_current,
            bg='#2E7591', fg='white', font=('Arial', 9, 'bold'), state=tk.DISABLED)
        self.btn_predict.pack(fill=tk.X, padx=4, pady=(4, 2))

        self.btn_validate = tk.Button(
            ia_frame, text="Tout valider  (V)", command=self.validate_all,
            bg='#3D6E4C', fg='white', font=('Arial', 9))
        self.btn_validate.pack(fill=tk.X, padx=4, pady=(0, 2))

        self.model_label = tk.Label(ia_frame, text="aucun modèle",
                                    font=('Arial', 8), fg='#888')
        self.model_label.pack(padx=4, pady=(0, 4))

        btn_delete = tk.Button(left_frame, text="Supprimer Sélection", command=self.delete_annotation,
                              font=('Arial', 10), bg='#f44336', fg='white', padx=10, pady=5)
        btn_delete.pack(pady=5, padx=10, fill=tk.X)

        tk.Frame(left_frame, height=2, bg='gray').pack(fill=tk.X, pady=15, padx=10)

        zoom_label = tk.Label(left_frame, text="Zoom:", font=('Arial', 11, 'bold'), bg='#f0f0f0')
        zoom_label.pack(pady=(0, 5))

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

        btn_save = tk.Button(left_frame, text="SAUVEGARDER JSON", command=self.save_json,
                           font=('Arial', 13, 'bold'), bg='#2196F3', fg='white', 
                           padx=20, pady=12, relief=tk.RAISED, borderwidth=3)
        btn_save.pack(pady=15, padx=10, fill=tk.X)

        save_hint = tk.Label(left_frame, text="(Raccourci: Ctrl+S)", 
                           font=('Arial', 8, 'italic'), bg='#f0f0f0', fg='#666')
        save_hint.pack(pady=(0, 10))

        self.info_label = tk.Label(left_frame, text="", font=('Arial', 9), bg='#f0f0f0', 
                                  wraplength=280, justify=tk.LEFT)
        self.info_label.pack(pady=10, padx=10)

        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        canvas_frame = tk.Frame(right_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        h_scrollbar = tk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        v_scrollbar = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas = tk.Canvas(canvas_frame, xscrollcommand=h_scrollbar.set, 
                               yscrollcommand=v_scrollbar.set, bg='#333333', cursor='cross')
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        h_scrollbar.config(command=self.canvas.xview)
        v_scrollbar.config(command=self.canvas.yview)

        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Shift-ButtonPress-1>", self.on_pan_start)
        self.canvas.bind("<Shift-B1-Motion>", self.on_panning)
        self.canvas.bind("<Shift-ButtonRelease-1>", self.on_pan_stop)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)
        
        self.canvas.bind("<Button-2>", self.on_pan_start)
        self.canvas.bind("<B2-Motion>", self.on_panning)
        self.canvas.bind("<ButtonRelease-2>", self.on_pan_stop)

        self.toplevel.bind("<Delete>", lambda e: self.delete_annotation())
        self.toplevel.bind("<a>", lambda e: self.predict_current())
        self.toplevel.bind("<v>", lambda e: self.validate_all())
        self.annotations_listbox.bind("<<ListboxSelect>>", self._on_select)
        self.toplevel.bind("<Control-s>", lambda e: self.save_json())
        self.toplevel.bind("<Control-o>", lambda e: self.load_image())
        self.toplevel.bind("<Delete>", lambda e: self.delete_annotation())
        self.toplevel.bind("<Left>", lambda e: self.prev_image())
        self.toplevel.bind("<Right>", lambda e: self.next_image())

        instructions = tk.Label(right_frame,
                              text="Cliquez et glissez pour créer une bounding box | "
                                   "Shift+Drag ou Molette+Drag pour se déplacer | "
                                   "Molette souris pour zoomer | "
                                   "Ctrl+S: Sauvegarder | Ctrl+O: Ouvrir | Delete: Supprimer | "
                                   "Flèches: Navigation",
                              font=('Arial', 10), bg='#e0e0e0', pady=5)
        instructions.pack(fill=tk.X)

    def select_class(self, class_id):
        self.class_buttons[self.selected_class].config(relief=tk.RAISED, borderwidth=1)
        self.selected_class = class_id
        self.class_buttons[class_id].config(relief=tk.SUNKEN, borderwidth=3)
        self.update_info()

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Sélectionner une image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"),
                      ("Tous les fichiers", "*.*")]
        )

        if not file_path:
            return

        self.image_path = Path(file_path)
        self.annotations = []
        self.selected_index = None
        self.modifiee = False
        if hasattr(self, "lbl_sans_defaut"):
            self.lbl_sans_defaut.config(text="")
        if hasattr(self, "aubier"):
            self.aubier.set("aucun")
        self.annotations_listbox.delete(0, tk.END)

        self.original_image = Image.open(self.image_path)
        self.image_width, self.image_height = self.original_image.size

        self.zoom_level = 1.0

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width > 1 and canvas_height > 1:
            scale_x = canvas_width / self.image_width
            scale_y = canvas_height / self.image_height
            self.scale_factor = min(scale_x, scale_y, 1.0)
        else:
            self.scale_factor = 1.0

        self.load_existing_json()
        if self.auto_predict and self.model is not None and not self.annotations:
            self.predict_current()

        self.display_image = self.original_image.copy()
        self.redraw_image()
        self.update_info()
        self.update_zoom_label()

    def redraw_image(self):
        if self.original_image is None:
            return

        img = self.original_image.copy()
        draw = ImageDraw.Draw(img)

        for i, ann in enumerate(self.annotations):
            bbox = ann['bbox']
            class_id = ann['class_id']
            color = self.colors[class_id]
            auto = ann.get('source') == 'auto'
            box = [bbox['x_min'], bbox['y_min'], bbox['x_max'], bbox['y_max']]

            if i == self.selected_index:
                draw.rectangle([box[0] - 3, box[1] - 3, box[2] + 3, box[3] + 3],
                               outline='#FFFFFF', width=2)

            if auto:
                self._dashed_rect(draw, box, color, width=3, dash=14)
                label = f"[IA {ann.get('confidence', 0) * 100:.0f}%] {self.libelles[class_id]}"
            else:
                draw.rectangle(box, outline=color, width=3)
                label = f"{class_id}: {self.libelles[class_id]}"

            draw.text((bbox['x_min'], max(0, bbox['y_min'] - 20)), label, fill=color)

        final_scale = self.scale_factor * self.zoom_level
        display_width = int(self.image_width * final_scale)
        display_height = int(self.image_height * final_scale)
        img_resized = img.resize((display_width, display_height), Image.Resampling.LANCZOS)

        self.photo_image = ImageTk.PhotoImage(img_resized)

        if self.canvas_image:
            self.canvas.delete(self.canvas_image)

        self.canvas_image = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

    @staticmethod
    def _dashed_rect(draw, box, color, width=3, dash=14):
        """Rectangle en pointille : marque une proposition non encore validee."""
        x1, y1, x2, y2 = box
        for x in range(int(x1), int(x2), dash * 2):
            draw.line([x, y1, min(x + dash, x2), y1], fill=color, width=width)
            draw.line([x, y2, min(x + dash, x2), y2], fill=color, width=width)
        for y in range(int(y1), int(y2), dash * 2):
            draw.line([x1, y, x1, min(y + dash, y2)], fill=color, width=width)
            draw.line([x2, y, x2, min(y + dash, y2)], fill=color, width=width)

    def load_folder(self):
        folder_path = filedialog.askdirectory(title="Sélectionner un dossier d'images")
        if not folder_path:
            return

        self.folder_path = Path(folder_path).resolve()
        self.image_list = []
        self.current_index = 0

        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']:
            self.image_list.extend(self.folder_path.glob(ext))

        if not self.image_list:
            all_files = list(self.folder_path.glob('*'))
            extensions = [f.suffix.lower() for f in all_files if f.is_file()]
            unique_ext = set(extensions)
            
            debug_info = f"Dossier: {self.folder_path}\n"
            debug_info += f"Total fichiers: {len(all_files)}\n"
            debug_info += f"Extensions trouvées: {', '.join(sorted(unique_ext))}\n\n"
            debug_info += "Premiers fichiers:\n"
            for f in all_files[:10]:
                debug_info += f"  - {f.name}\n"
            
            messagebox.showerror("Erreur - Aucun fichier image trouvé", debug_info)
            return

        self.image_list.sort()

        self.load_image_by_index(0)

    def load_folder_directly(self, folder):
        self.folder_path = Path(folder).resolve()
        self.image_list = []
        self.current_index = 0

        extensions = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP")
        for root, dirs, files in os.walk(str(self.folder_path)):
            for file in files:
                file_lower = file.lower()
                if any(file_lower.endswith(ext) for ext in extensions):
                    file_path = os.path.join(root, file)
                    self.image_list.append(Path(file_path))

        self.image_list = sorted(self.image_list)

        if not self.image_list:
            print(f"Aucune image trouvée dans {folder}")
            return

        self.load_image_by_index(0)

    def load_image_by_index(self, index):
        if not self.image_list or index < 0 or index >= len(self.image_list):
            return

        self.current_index = index
        self.image_path = self.image_list[index]
        self.annotations = []
        self.selected_index = None
        self.modifiee = False
        if hasattr(self, "lbl_sans_defaut"):
            self.lbl_sans_defaut.config(text="")
        if hasattr(self, "aubier"):
            self.aubier.set("aucun")
        self.annotations_listbox.delete(0, tk.END)

        self.original_image = Image.open(self.image_path)
        self.image_width, self.image_height = self.original_image.size

        self.zoom_level = 1.0

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width > 1 and canvas_height > 1:
            scale_x = canvas_width / self.image_width
            scale_y = canvas_height / self.image_height
            self.scale_factor = min(scale_x, scale_y, 1.0)
        else:
            self.scale_factor = 1.0

        self.load_existing_json()
        if self.auto_predict and self.model is not None and not self.annotations:
            self.predict_current()

        self.display_image = self.original_image.copy()
        self.redraw_image()
        self.update_info()
        self.update_zoom_label()
        self.update_counter()

    def _aubier_change(self):
        """L'aubier est une donnee a part entiere, pas un accessoire du noeud.

        Il se renseigne sur une planche sans le moindre defaut, et c'est lui
        qui decide du suffixe X / XX. Le noter doit donc suffire a declencher
        l'enregistrement.
        """
        self.modifiee = True
        self.update_info()

    def marquer_sans_defaut(self):
        """Declare la planche examinee et saine, et l'enregistre aussitot.

        Une planche propre ne se distingue autrement pas d'une planche non
        regardee : dans les deux cas la liste est vide. Ce bouton est ce qui
        fait la difference, et il evite d'avoir a forcer SAUVEGARDER.
        """
        if self.image_path is None:
            return
        if self.annotations and not messagebox.askyesno(
                "Annotations presentes",
                f"{len(self.annotations)} annotation(s) sur cette planche.\n"
                "Les supprimer et la declarer saine ?"):
            return
        self.annotations = []
        self.selected_index = None
        self.modifiee = True
        self.refresh_listbox()
        self.redraw_image()
        self._autosave()
        self.update_info()
        self.lbl_sans_defaut.config(
            text=f"✓ {self.image_path.name} enregistrée sans défaut", fg="#3D6E4C")

    def _autosave(self):
        """Ecrit le JSON sans dialogue, avant de changer d'image.

        Sans cela, passer a l'image suivante perdait le travail en cours : rien
        n'etait ecrit tant qu'on n'avait pas clique sur SAUVEGARDER.

        Une liste de defauts vide ne vaut pas « rien a enregistrer ». Une
        planche propre est un exemple negatif, que `convert_json_to_yolo.py`
        traduit en etiquette vide et qui apprend au modele a ne pas inventer
        de defaut ; et l'aubier se renseigne meme sans le moindre noeud. Le
        premier jet s'arretait sur `not self.annotations` et perdait les deux.

        Ce qui decide d'ecrire, c'est donc d'avoir touche a l'image, pas d'y
        avoir trouve quelque chose. Defiler sans rien faire n'ecrit rien :
        sinon chaque planche survolee serait declaree saine sans avoir ete
        regardee, et le jeu se remplirait de faux negatifs.
        """
        if self.image_path is None:
            return
        cible = self.image_path.with_suffix(".json")
        if not self.modifiee and not cible.exists():
            return
        data = {
            "image": self.image_path.name,
            "image_size": {"width": self.image_width, "height": self.image_height},
            "annotations": self.annotations,
            "aubier": self.aubier.get(),
            "sans_defaut": not self.annotations,
            "classes": self.classes,
        }
        try:
            cible.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                             encoding="utf-8")
            self.modifiee = False
        except OSError:
            pass

    def prev_image(self):
        if not self.image_list:
            return
        self._autosave()
        new_index = (self.current_index - 1) % len(self.image_list)
        self.load_image_by_index(new_index)

    def next_image(self):
        if not self.image_list:
            return
        self._autosave()
        new_index = (self.current_index + 1) % len(self.image_list)
        self.load_image_by_index(new_index)

    def update_counter(self):
        if self.image_list:
            self.counter_label.config(text=f"Image {self.current_index + 1} / {len(self.image_list)}")
        else:
            self.counter_label.config(text="")

    def on_mouse_down(self, event):
        if self.original_image is None:
            return

        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        final_scale = self.scale_factor * self.zoom_level
        self.start_x = int(canvas_x / final_scale)
        self.start_y = int(canvas_y / final_scale)

        if self.current_rect:
            self.canvas.delete(self.current_rect)

        self.current_rect = self.canvas.create_rectangle(
            canvas_x, canvas_y, canvas_x, canvas_y,
            outline=self.colors[self.selected_class], width=2
        )

    def on_mouse_drag(self, event):
        if self.original_image is None or self.start_x is None:
            return

        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        final_scale = self.scale_factor * self.zoom_level
        start_canvas_x = self.start_x * final_scale
        start_canvas_y = self.start_y * final_scale

        self.canvas.coords(self.current_rect,
                          start_canvas_x, start_canvas_y,
                          canvas_x, canvas_y)

    def on_mouse_up(self, event):
        if self.original_image is None or self.start_x is None:
            return

        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        final_scale = self.scale_factor * self.zoom_level
        end_x = int(canvas_x / final_scale)
        end_y = int(canvas_y / final_scale)

        x_min = min(self.start_x, end_x)
        x_max = max(self.start_x, end_x)
        y_min = min(self.start_y, end_y)
        y_max = max(self.start_y, end_y)

        if (x_max - x_min) < 5 or (y_max - y_min) < 5:
            if self.current_rect:
                self.canvas.delete(self.current_rect)
                self.current_rect = None
            self.start_x = None
            self.start_y = None
            return

        self.annotations.append(self._make_annotation(
            self.selected_class, x_min, y_min, x_max, y_max, source="manual"))
        self.modifiee = True
        self.refresh_listbox()

        if self.current_rect:
            self.canvas.delete(self.current_rect)
            self.current_rect = None
        self.start_x = None
        self.start_y = None

        self.redraw_image()
        self.update_info()


    def load_model(self, model_path, conf=0.25):
        """Charge le détecteur qui servira de pré-annotateur."""
        try:
            from ultralytics import YOLO
        except ImportError:
            self.model_label.config(text="ultralytics absent", fg="#9B3A2B")
            return
        try:
            self.model = YOLO(str(model_path))
            self.model_path = pathlib.Path(model_path)
            self.model_conf = conf
            self.btn_predict.config(state=tk.NORMAL)
            self.model_label.config(
                text=f"{self.model_path.name}  (seuil {conf:.2f})", fg="#2E7591")
        except Exception as exc:
            self.model_label.config(text=f"echec: {exc}"[:40], fg="#9B3A2B")

    def predict_current(self):
        """Propose des boîtes sur l'image courante.

        Les propositions sont marquées `source="auto"` : affichées en pointillé,
        préfixées [IA] dans la liste, et comptées à part. Tant qu'elles n'ont pas
        été validées, elles signalent qu'un humain n'a pas encore tranché.
        """
        if self.model is None or self.original_image is None:
            return
        import numpy as np
        img = np.array(self.original_image.convert("RGB"))[:, :, ::-1]
        try:
            res = self.model.predict(img, conf=self.model_conf, verbose=False)[0]
        except Exception as exc:
            messagebox.showerror("Erreur", f"Inference impossible:\n{exc}")
            return

        existantes = [(a["bbox"]["x_min"], a["bbox"]["y_min"],
                       a["bbox"]["x_max"], a["bbox"]["y_max"]) for a in self.annotations]
        ajoutees = 0
        for b in res.boxes:
            cid = int(b.cls[0])
            if cid not in self.classes:
                continue
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
            if x2 - x1 < 5 or y2 - y1 < 5:
                continue
            if any(self._iou((x1, y1, x2, y2), e) > 0.5 for e in existantes):
                continue
            self.annotations.append(self._make_annotation(
                cid, x1, y1, x2, y2, source="auto", confidence=float(b.conf[0])))
            ajoutees += 1

        self.refresh_listbox()
        self.redraw_image()
        self.update_info()
        if ajoutees == 0:
            self.model_label.config(text="aucune proposition nouvelle", fg="#8E6210")
        else:
            self.model_label.config(
                text=f"{ajoutees} proposition(s) — a verifier", fg="#2E7591")

    @staticmethod
    def _iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        aa = (a[2] - a[0]) * (a[3] - a[1])
        bb = (b[2] - b[0]) * (b[3] - b[1])
        return inter / float(aa + bb - inter)

    def _make_annotation(self, class_id, x_min, y_min, x_max, y_max,
                         source="manual", confidence=None):
        ann = {
            "class": self.classes[class_id],
            "class_id": class_id,
            "bbox": {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max},
            "bbox_normalized": {
                "x_center": round(((x_min + x_max) / 2) / self.image_width, 6),
                "y_center": round(((y_min + y_max) / 2) / self.image_height, 6),
                "width": round((x_max - x_min) / self.image_width, 6),
                "height": round((y_max - y_min) / self.image_height, 6),
            },
            "area": (x_max - x_min) * (y_max - y_min),
            "source": source,
        }
        if confidence is not None:
            ann["confidence"] = round(confidence, 4)
        return ann

    def validate_all(self):
        """Accepte toutes les propositions restantes."""
        n = 0
        for a in self.annotations:
            if a.get("source") == "auto":
                a["source"] = "manual"
                a.pop("confidence", None)
                n += 1
        if n:
            self.modifiee = True
            self.refresh_listbox()
            self.redraw_image()
            self.update_info()
            self.model_label.config(text=f"{n} proposition(s) validee(s)", fg="#3D6E4C")

    def refresh_listbox(self):
        self.annotations_listbox.delete(0, tk.END)
        for i, a in enumerate(self.annotations, 1):
            # Repli sur le nom enregistre si l'identifiant manque : une
            # annotation venue d'un ancien fichier doit rester lisible.
            nom = self.libelles.get(a.get("class_id"), a["class"])
            if a.get("source") == "auto":
                c = a.get("confidence", 0) * 100
                self.annotations_listbox.insert(tk.END, f"{i}. [IA {c:.0f}%] {nom}")
                self.annotations_listbox.itemconfig(i - 1, fg="#8E6210")
            else:
                self.annotations_listbox.insert(
                    tk.END, f"{i}. {nom} - {a['area']}px2")

    def _on_select(self, _event=None):
        sel = self.annotations_listbox.curselection()
        self.selected_index = sel[0] if sel else None
        self.redraw_image()

    def load_existing_json(self):
        """Recharge les annotations deja faites sur cette image.

        Sans ca, naviguer vers l'image suivante puis revenir effacait tout le
        travail : l'outil repartait d'une liste vide a chaque chargement.
        """
        if self.image_path is None:
            return
        jp = self.image_path.with_suffix(".json")
        if not jp.exists():
            return
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        self.aubier.set(data.get("aubier", "aucun"))
        for a in data.get("annotations", []):
            if "bbox" in a and "class_id" in a:
                a.setdefault("source", "manual")
                self.annotations.append(a)
        self.refresh_listbox()

    def delete_annotation(self):
        selection = self.annotations_listbox.curselection()
        if not selection:
            return

        index = selection[0]
        del self.annotations[index]
        self.modifiee = True
        self.selected_index = None
        self.refresh_listbox()

        self.redraw_image()
        self.update_info()

    def save_json(self):
        if self.original_image is None:
            messagebox.showerror("Erreur", "Aucune image chargée!")
            return

        if not self.annotations and not messagebox.askyesno(
                "Planche sans défaut",
                "Aucune annotation sur cette planche.\n\n"
                "L'enregistrer la déclare saine — c'est un exemple utile à "
                "l'entraînement. Continuer ?"):
            return

        data = {
            "image": self.image_path.name,
            "image_size": {
                "width": self.image_width,
                "height": self.image_height
            },
            "annotations": self.annotations,
            "aubier": self.aubier.get(),
            "sans_defaut": not self.annotations,
            "classes": self.classes
        }

        json_path = self.image_path.with_suffix('.json')

        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            messagebox.showinfo("Succès",
                f"JSON sauvegardé avec succès!\n\n"
                f"Fichier: {json_path.name}\n"
                f"Annotations: {len(self.annotations)}")

            self.modifiee = False
            self.update_info()
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de sauvegarder:\n{str(e)}")

    def update_info(self):
        if self.original_image:
            info = f"Image: {self.image_path.name}\n"
            info += f"Taille: {self.image_width}x{self.image_height}\n"
            n_auto = sum(1 for a in self.annotations if a.get("source") == "auto")
            info += f"Annotations: {len(self.annotations)}\n"
            if n_auto:
                info += f"  validées: {len(self.annotations) - n_auto}"
                info += f"   à vérifier: {n_auto}\n"
            if self.image_list:
                info += f"Position: {self.current_index + 1}/{len(self.image_list)}\n"
            info += (f"\nClasse sélectionnée:\n"
                     f"{self.selected_class}: {self.libelles[self.selected_class]}")
        else:
            info = "Aucune image chargée.\n\nChargez une image pour commencer.\n(Ctrl+O)"

        self.info_label.config(text=info)

    def update_zoom_label(self):
        zoom_percent = int(self.zoom_level * 100)
        self.zoom_label_display.config(text=f"{zoom_percent}%")

    def zoom_in(self):
        if self.original_image is None:
            return

        if self.zoom_level < 5.0:
            self.zoom_level *= 1.25
            if self.zoom_level > 5.0:
                self.zoom_level = 5.0
            self.redraw_image()
            self.update_zoom_label()

    def zoom_out(self):
        if self.original_image is None:
            return

        if self.zoom_level > 0.25:
            self.zoom_level /= 1.25
            if self.zoom_level < 0.25:
                self.zoom_level = 0.25
            self.redraw_image()
            self.update_zoom_label()

    def zoom_reset(self):
        if self.original_image is None:
            return

        self.zoom_level = 1.0
        self.redraw_image()
        self.update_zoom_label()

    def on_mouse_wheel(self, event):
        if self.original_image is None:
            return

        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        old_zoom = self.zoom_level

        if event.num == 4 or event.delta > 0:
            if self.zoom_level < 5.0:
                self.zoom_level = min(self.zoom_level * 1.25, 5.0)
        elif event.num == 5 or event.delta < 0:
            if self.zoom_level > 0.1:
                self.zoom_level = max(self.zoom_level / 1.25, 0.1)

        if self.zoom_level == old_zoom:
            return

        self.redraw_image()
        self.update_zoom_label()

        zoom_ratio = self.zoom_level / old_zoom
        final_scale = self.scale_factor * self.zoom_level
        new_total_width = int(self.image_width * final_scale)
        new_total_height = int(self.image_height * final_scale)

        if new_total_width > 0 and new_total_height > 0:
            new_scroll_x = (canvas_x * zoom_ratio - event.x) / new_total_width
            new_scroll_y = (canvas_y * zoom_ratio - event.y) / new_total_height
            self.canvas.xview_moveto(max(0.0, new_scroll_x))
            self.canvas.yview_moveto(max(0.0, new_scroll_y))

    def on_pan_start(self, event):
        """Démarre le panning"""
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        self.canvas.config(cursor='hand2')

    def on_panning(self, event):
        """Déplace l'image"""
        if self.pan_start_x is None or self.pan_start_y is None:
            return

        dx = event.x - self.pan_start_x
        dy = event.y - self.pan_start_y

        final_scale = self.scale_factor * self.zoom_level
        total_width = int(self.image_width * final_scale)
        total_height = int(self.image_height * final_scale)

        if total_width > 0 and total_height > 0:
            new_x = self.canvas.xview()[0] - dx / total_width
            new_y = self.canvas.yview()[0] - dy / total_height
            self.canvas.xview_moveto(max(0.0, min(1.0, new_x)))
            self.canvas.yview_moveto(max(0.0, min(1.0, new_y)))

        self.pan_start_x = event.x
        self.pan_start_y = event.y

    def on_pan_stop(self, event):
        """Arrête le panning"""
        self.pan_start_x = None
        self.pan_start_y = None
        self.canvas.config(cursor='cross')

def main():
    import argparse

    ap = argparse.ArgumentParser(
        description="Annotation des defauts, avec pre-annotation par un modele",
        epilog="Exemple : python BOBER/scripts/annotation_tool.py BOBER/extracted "
               "--model BOBER/model/weights/BOBERv1.6.pt --auto")
    ap.add_argument("dossier", nargs="?", help="dossier d'images a annoter")
    ap.add_argument("--model", default=None,
                    help="modele de pre-annotation (ex. BOBERv1.6.pt)")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="seuil de confiance des propositions (defaut 0.25)")
    ap.add_argument("--auto", action="store_true",
                    help="pre-annote automatiquement chaque image ouverte")
    args = ap.parse_args()

    root = tk.Tk()
    app = WoodDefectAnnotator(root)

    if args.model:
        root.after(100, lambda: app.load_model(args.model, args.conf))
    app.auto_predict = args.auto
    
    if args.dossier and os.path.isdir(args.dossier):
        root.after(600, lambda: app.load_folder_directly(args.dossier))

    root.mainloop()

if __name__ == "__main__":
    main()
