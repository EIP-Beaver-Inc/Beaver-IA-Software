"""
La fenêtre unique et ses vues.

Reprend la structure du logiciel principal : barre latérale sombre à gauche
avec le logo, zone de contenu crème à droite, état actif en ambre. Les vues
sont empilées et échangées, jamais ouvertes à côté.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, font as tkfont, messagebox, ttk

from . import theme as T
from . import workspace as W
from .taches import (CATALOGUE, ESSENCES_FEUILLUS, ESSENCES_RESINEUX, Param,
                     Tache, etat_projet)

RACINE = Path(__file__).resolve().parent.parent
LOGO = RACINE / "assets" / "beaver-logo.png"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Beaver — atelier IA")
        self.geometry("1360x860")
        self.minsize(1100, 700)
        self.configure(bg=T.CONTENT_BG)

        self.ws: Path | None = W.resoudre()
        self.session = W.session()
        self.proc: subprocess.Popen | None = None
        self.flux: queue.Queue[str] = queue.Queue()
        self.vues: dict[str, tk.Frame] = {}
        self.nav: dict[str, tuple[tk.Frame, tk.Label]] = {}
        self.active = ""
        self.champs: dict[str, tk.Variable] = {}
        self.tache: Tache | None = None
        self.annotateur = None
        self._photo_logo = None

        self.fam = T.police_disponible(self)
        self.f_titre = tkfont.Font(family=self.fam, size=20, weight="bold")
        self.f_section = tkfont.Font(family=self.fam, size=14, weight="bold")
        self.f_corps = tkfont.Font(family=self.fam, size=10)
        self.f_petit = tkfont.Font(family=self.fam, size=9)
        self.f_micro = tkfont.Font(family=self.fam, size=8)
        self.f_mono = tkfont.Font(
            family=T.police_disponible(self, T.FAMILLE_MONO), size=9)

        try:
            ttk.Style(self).theme_use("clam")
        except tk.TclError:
            pass

        if self.ws is None:
            self._ecran_configuration()
        else:
            self._construire()
        self.protocol("WM_DELETE_WINDOW", self._quitter)
        #: Relance de la boucle de lecture de la console. On en garde l'ident
        #: pour l'annuler en partant : sans cela, le rappel en attente se
        #: déclenche sur une fenêtre déjà détruite et Tcl se plaint à chaque
        #: fermeture.
        self._pompe = self.after(80, self._pomper)

    # ---- première ouverture : où est le dépôt IA ? --------------------------

    def _ecran_configuration(self):
        self.config_frame = tk.Frame(self, bg=T.CONTENT_BG)
        self.config_frame.pack(fill=tk.BOTH, expand=True)
        boite = tk.Frame(self.config_frame, bg=T.CARD_BG,
                         highlightthickness=1, highlightbackground=T.CARD_BORDURE)
        boite.place(relx=0.5, rely=0.5, anchor="center", width=660)

        tk.Label(boite, text="Beaver", bg=T.CARD_BG, fg=T.PRIMARY,
                 font=self.f_titre).pack(pady=(26, 4))
        tk.Label(boite, text="Où se trouve le dépôt IA ?", bg=T.CARD_BG,
                 fg=T.TEXT_MAIN, font=self.f_section).pack()
        tk.Label(boite, bg=T.CARD_BG, fg=T.TEXT_SEC, font=self.f_petit,
                 wraplength=560, justify=tk.CENTER,
                 text="Ce logiciel ne contient que l'interface. Les modèles, "
                      "les jeux de données et les scripts vivent dans le dépôt "
                      "ORGA-IA_BEAVER. Indiquez-le une fois, il sera retenu.").pack(
            padx=40, pady=(8, 18))

        self.var_ws = tk.StringVar()
        ligne = tk.Frame(boite, bg=T.CARD_BG)
        ligne.pack(fill=tk.X, padx=40)
        tk.Entry(ligne, textvariable=self.var_ws, font=self.f_corps).pack(
            side=tk.LEFT, fill=tk.X, expand=True, ipady=4)
        tk.Button(ligne, text="Parcourir", command=self._choisir_ws,
                  bg=T.PRIMARY, fg="white", relief=tk.FLAT, font=self.f_petit,
                  padx=14, cursor="hand2").pack(side=tk.LEFT, padx=(8, 0))

        for c in W.candidats():
            tk.Button(boite, text=f"Utiliser  {c}", command=lambda p=c: self._valider_ws(p),
                      bg=T.TIP_BG, fg=T.PRIMARY, relief=tk.FLAT,
                      font=self.f_petit, cursor="hand2", pady=6).pack(
                fill=tk.X, padx=40, pady=(10, 0))

        self.lbl_diag = tk.Label(boite, text="", bg=T.CARD_BG, fg=T.ERREUR,
                                 font=self.f_petit, wraplength=560,
                                 justify=tk.LEFT)
        self.lbl_diag.pack(padx=40, pady=(10, 0))

        tk.Button(boite, text="Continuer",
                  command=lambda: self._valider_ws(Path(self.var_ws.get())),
                  bg=T.ACTIVE_BG, fg="white", relief=tk.FLAT,
                  font=tkfont.Font(family=self.fam, size=11, weight="bold"),
                  pady=9, cursor="hand2").pack(fill=tk.X, padx=40, pady=(16, 28))

    def _choisir_ws(self):
        d = filedialog.askdirectory(title="Dossier du dépôt IA")
        if d:
            self.var_ws.set(d)

    def _valider_ws(self, chemin: Path):
        if not str(chemin).strip():
            self.lbl_diag.config(text="Indiquez un dossier.")
            return
        soucis = W.diagnostic(Path(chemin))
        if soucis:
            self.lbl_diag.config(text="Ce n'est pas le dépôt IA.\n" + "\n".join(soucis))
            return
        self.ws = Path(chemin).resolve()
        W.enregistrer(self.ws)
        self.config_frame.destroy()
        self._construire()

    # ---- ossature -----------------------------------------------------------

    def _construire(self):
        sys.path.insert(0, str(RACINE / "outils"))

        corps = tk.Frame(self, bg=T.CONTENT_BG)
        corps.pack(fill=tk.BOTH, expand=True)

        barre = tk.Frame(corps, bg=T.SIDEBAR_BG, width=T.LARGEUR_SIDEBAR)
        barre.pack(side=tk.LEFT, fill=tk.Y)
        barre.pack_propagate(False)
        self._logo(barre)

        pages = [
            ("accueil", "Accueil", "⌂"),
            ("defauts", "Défauts", "◈"),
            ("essence", "Essence", "❖"),
            ("roi", "Planche", "▭"),
            ("donnees", "Données", "▤"),
            ("entrainer", "Entraîner", "◉"),
            ("analyser", "Analyser", "▶"),
        ]
        for cle, lib, ico in pages:
            self._bouton_nav(barre, cle, lib, ico)

        tk.Frame(barre, bg=T.SIDEBAR_BG).pack(fill=tk.BOTH, expand=True)
        tk.Label(barre, text="atelier IA", bg=T.SIDEBAR_BG, fg="#3D6B4E",
                 font=self.f_micro).pack(pady=(0, 10))

        self.zone = tk.Frame(corps, bg=T.CONTENT_BG)
        self.zone.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._aller("accueil")

    def _logo(self, parent):
        cadre = tk.Frame(parent, bg=T.SIDEBAR_BG)
        cadre.pack(pady=(18, 14))
        if LOGO.exists():
            try:
                from PIL import Image, ImageTk
                im = Image.open(LOGO)
                im.thumbnail((62, 62), Image.Resampling.LANCZOS)
                self._photo_logo = ImageTk.PhotoImage(im)
                tk.Label(cadre, image=self._photo_logo, bg=T.SIDEBAR_BG).pack()
                return
            except Exception:  # noqa: BLE001
                pass
        tk.Label(cadre, text="BEAVER", bg=T.SIDEBAR_BG, fg="white",
                 font=tkfont.Font(family=self.fam, size=11, weight="bold")).pack()

    def _bouton_nav(self, parent, cle, libelle, icone):
        case = tk.Frame(parent, bg=T.SIDEBAR_BG, cursor="hand2", height=58)
        case.pack(fill=tk.X, padx=7, pady=2)
        case.pack_propagate(False)
        li = tk.Label(case, text=icone, bg=T.SIDEBAR_BG, fg=T.INACTIVE_CLR,
                      font=tkfont.Font(family=self.fam, size=15))
        li.pack(pady=(7, 0))
        lt = tk.Label(case, text=libelle, bg=T.SIDEBAR_BG, fg=T.INACTIVE_CLR,
                      font=self.f_micro)
        lt.pack()
        self.nav[cle] = (case, lt, li)
        for w in (case, li, lt):
            w.bind("<Button-1>", lambda _e, k=cle: self._aller(k))

    def _aller(self, cle: str):
        if cle == self.active:
            return
        for k, (case, lt, li) in self.nav.items():
            actif = k == cle
            fond = T.ACTIVE_BG if actif else T.SIDEBAR_BG
            coul = T.ACTIVE_CLR if actif else T.INACTIVE_CLR
            case.config(bg=fond)
            lt.config(bg=fond, fg=coul,
                      font=tkfont.Font(family=self.fam, size=8,
                                       weight="bold" if actif else "normal"))
            li.config(bg=fond, fg=coul)
        for v in self.vues.values():
            v.pack_forget()
        if cle not in self.vues:
            self.vues[cle] = self._batir(cle)
        self.vues[cle].pack(fill=tk.BOTH, expand=True)
        self.active = cle
        if cle == "accueil":
            self._rafraichir()

    def _batir(self, cle: str) -> tk.Frame:
        if cle == "accueil":
            return self._vue_accueil()
        if cle == "defauts":
            return self._vue_defauts()
        if cle == "essence":
            return self._vue_essence()
        return self._vue_taches(cle)

    def _entete(self, parent, titre: str, sous: str) -> tk.Frame:
        h = tk.Frame(parent, bg=T.PRIMARY, height=76)
        h.pack(fill=tk.X)
        h.pack_propagate(False)
        tk.Label(h, text=titre, bg=T.PRIMARY, fg="white",
                 font=self.f_titre, anchor="w").pack(
            fill=tk.X, padx=26, pady=(14, 0))
        tk.Label(h, text=sous, bg=T.PRIMARY, fg="#B7D4C2",
                 font=self.f_petit, anchor="w").pack(fill=tk.X, padx=26)
        return h

    # ---- accueil ------------------------------------------------------------

    def _vue_accueil(self) -> tk.Frame:
        f = tk.Frame(self.zone, bg=T.CONTENT_BG)
        self._entete(f, "Où en est le projet",
                     "Ce qui est prêt, et ce qui bloque la suite")
        self.cartes = tk.Frame(f, bg=T.CONTENT_BG)
        self.cartes.pack(fill=tk.X, padx=22, pady=18)

        guide = tk.Frame(f, bg=T.CARD_BG, highlightthickness=1,
                         highlightbackground=T.CARD_BORDURE)
        guide.pack(fill=tk.BOTH, expand=True, padx=22, pady=(0, 22))
        tk.Label(guide, text="Par où commencer", bg=T.CARD_BG, fg=T.TEXT_MAIN,
                 font=self.f_section, anchor="w").pack(
            fill=tk.X, padx=20, pady=(16, 8))
        etapes = [
            ("Annoter l'essence",
             "Sans elle aucune planche n'est notée : l'essence choisit le "
             "référentiel et la règle de mesure des nœuds.", "essence"),
            ("Annoter les défauts",
             "L'IA propose, vous validez ou corrigez.", "defauts"),
            ("Entraîner",
             "Quand il y a assez d'images annotées.", "entrainer"),
            ("Analyser une vidéo",
             "La chaîne complète, jusqu'à la note EN 975-1.", "analyser"),
        ]
        for n, (t, d, dest) in enumerate(etapes, 1):
            li = tk.Frame(guide, bg=T.CARD_BG, cursor="hand2")
            li.pack(fill=tk.X, padx=20, pady=2)
            tk.Label(li, text=str(n), bg=T.TIP_BG, fg=T.PRIMARY,
                     font=tkfont.Font(family=self.fam, size=11, weight="bold"),
                     width=3).pack(side=tk.LEFT, padx=(0, 12), ipady=4)
            tx = tk.Frame(li, bg=T.CARD_BG)
            tx.pack(side=tk.LEFT, fill=tk.X, expand=True)
            tk.Label(tx, text=t, bg=T.CARD_BG, fg=T.TEXT_MAIN,
                     font=tkfont.Font(family=self.fam, size=10, weight="bold"),
                     anchor="w").pack(fill=tk.X)
            tk.Label(tx, text=d, bg=T.CARD_BG, fg=T.TEXT_SEC, font=self.f_petit,
                     anchor="w", wraplength=820, justify=tk.LEFT).pack(
                fill=tk.X, pady=(0, 8))
            for w in (li, tx, *tx.winfo_children()):
                w.bind("<Button-1>", lambda _e, k=dest: self._aller(k))
        return f

    def _rafraichir(self):
        for w in self.cartes.winfo_children():
            w.destroy()
        for i, (lib, val, niv) in enumerate(etat_projet(self.ws)):
            fond = {"ok": T.TIP_BG, "alerte": T.ALERTE_BG, "erreur": T.ERREUR_BG}[niv]
            txt = {"ok": T.SUCCES, "alerte": T.ALERTE, "erreur": T.ERREUR}[niv]
            c = tk.Frame(self.cartes, bg=T.CARD_BG, highlightthickness=1,
                         highlightbackground=T.CARD_BORDURE)
            c.grid(row=i // 3, column=i % 3, sticky="ew", padx=5, pady=5)
            self.cartes.columnconfigure(i % 3, weight=1)
            tk.Label(c, text=lib.upper(), bg=T.CARD_BG, fg=T.TEXT_TERTIAIRE,
                     font=self.f_micro, anchor="w").pack(
                fill=tk.X, padx=14, pady=(12, 3))
            tk.Label(c, text=val, bg=fond, fg=txt,
                     font=tkfont.Font(family=self.fam, size=10, weight="bold"),
                     anchor="w", padx=9, pady=5).pack(fill=tk.X, padx=14,
                                                      pady=(0, 13))

    # ---- annotation des défauts, montée dans la fenêtre ---------------------

    def _vue_defauts(self) -> tk.Frame:
        f = tk.Frame(self.zone, bg=T.CONTENT_BG)
        self._entete(f, "Annotation des défauts",
                     "A proposer · V tout valider · Suppr supprimer · souris pour ajouter")
        hote = tk.Frame(f, bg=T.CONTENT_BG)
        hote.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        try:
            from annotation_tool import WoodDefectAnnotator
            self.annotateur = WoodDefectAnnotator(hote)
            modele = self.ws / "BOBER/model/weights/BOBERv1.5.pt"
            if modele.exists():
                self.after(300,
                           lambda: self.annotateur.load_model(str(modele), 0.10))
            dossier = self.ws / "BOBER" / "extracted"
            if dossier.is_dir():
                self.after(600,
                           lambda: self.annotateur.load_folder_directly(str(dossier)))
        except Exception as exc:  # noqa: BLE001
            tk.Label(hote, text=f"Éditeur indisponible : {exc}",
                     bg=T.CONTENT_BG, fg=T.ERREUR, font=self.f_corps).pack(pady=40)
        return f

    # ---- annotation d'essence ----------------------------------------------

    def _vue_essence(self) -> tk.Frame:
        f = tk.Frame(self.zone, bg=T.CONTENT_BG)
        self._entete(f, "Annotation de l'essence",
                     "Chêne et hêtre sont les seules essences couvertes par EN 975-1")
        self.esp_fichiers: list[Path] = []
        self.esp_index = 0

        info = tk.Frame(f, bg=T.ALERTE_BG)
        info.pack(fill=tk.X, padx=22, pady=(14, 0))
        tk.Label(info, bg=T.ALERTE_BG, fg=T.ALERTE, font=self.f_petit,
                 anchor="w", text="  Ne devinez pas : un mauvais label est pire "
                 "qu'un label manquant. Utilisez Ignorer en cas de doute.").pack(
            fill=tk.X, pady=7)

        self.esp_canvas = tk.Canvas(f, bg="#20291F", highlightthickness=0)
        self.esp_canvas.pack(fill=tk.BOTH, expand=True, padx=22, pady=12)

        bas = tk.Frame(f, bg=T.CONTENT_BG)
        bas.pack(fill=tk.X, padx=22, pady=(0, 16))
        for titre, liste, coul in (("Feuillus", ESSENCES_FEUILLUS, T.PRIMARY),
                                   ("Résineux", ESSENCES_RESINEUX, T.PRIMARY_CLAIR)):
            bloc = tk.Frame(bas, bg=T.CONTENT_BG)
            bloc.pack(fill=tk.X, pady=3)
            tk.Label(bloc, text=titre.upper(), bg=T.CONTENT_BG,
                     fg=T.TEXT_TERTIAIRE, font=self.f_micro, width=10,
                     anchor="w").pack(side=tk.LEFT)
            for sp in liste:
                tk.Button(bloc, text=sp, bg=coul, fg="white", relief=tk.FLAT,
                          font=self.f_petit, padx=12, pady=6, cursor="hand2",
                          activebackground=T.ACTIVE_BG, activeforeground="white",
                          command=lambda s=sp: self._esp_choisir(s)).pack(
                    side=tk.LEFT, padx=3)
        act = tk.Frame(bas, bg=T.CONTENT_BG)
        act.pack(fill=tk.X, pady=(10, 0))
        tk.Button(act, text="Ignorer", command=lambda: self._esp_choisir(None),
                  bg=T.CARD_BG, fg=T.TEXT_MAIN, relief=tk.FLAT, padx=16,
                  pady=7, cursor="hand2", font=self.f_petit).pack(side=tk.LEFT)
        tk.Button(act, text="Changer de dossier", command=self._esp_charger,
                  bg=T.CARD_BG, fg=T.TEXT_MAIN, relief=tk.FLAT, padx=16,
                  pady=7, cursor="hand2", font=self.f_petit).pack(
            side=tk.LEFT, padx=8)
        self.esp_info = tk.Label(act, text="", bg=T.CONTENT_BG, fg=T.TEXT_SEC,
                                 font=self.f_petit)
        self.esp_info.pack(side=tk.LEFT, padx=14)
        self.esp_compteur = tk.Label(act, text="", bg=T.CONTENT_BG,
                                     fg=T.PRIMARY, font=self.f_petit)
        self.esp_compteur.pack(side=tk.RIGHT)

        self.after(250, lambda: self._esp_charger(
            self.ws / "BOBER" / "horizontal_samples"))
        return f

    def _esp_charger(self, dossier=None):
        if dossier is None:
            d = filedialog.askdirectory(initialdir=str(self.ws))
            if not d:
                return
            dossier = Path(d)
        dossier = Path(dossier)
        if not dossier.is_dir():
            return
        self.esp_fichiers = [p for p in sorted(dossier.glob("*.json"))
                             if p.name != "MANIFEST.json"]
        self.esp_index = -1
        self._esp_suivant()

    def _esp_suivant(self):
        self.esp_index += 1
        while self.esp_index < len(self.esp_fichiers):
            f = self.esp_fichiers[self.esp_index]
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self.esp_index += 1
                continue
            if d.get("species"):
                self.esp_index += 1
                continue
            img = next((f.with_suffix(e) for e in (".jpg", ".jpeg", ".png", ".bmp")
                        if f.with_suffix(e).exists()), None)
            if img is None:
                self.esp_index += 1
                continue
            self._esp_afficher(img)
            self.esp_info.config(text=img.name)
            self._esp_compter()
            return
        self.esp_canvas.delete("all")
        self.esp_canvas.create_text(
            max(self.esp_canvas.winfo_width() // 2, 300), 180,
            text="Toutes les images de ce dossier sont annotées.",
            fill="white", font=self.f_section)
        self.esp_info.config(text="")
        self._esp_compter()

    def _esp_compter(self):
        fait = 0
        for f in self.esp_fichiers:
            try:
                if json.loads(f.read_text(encoding="utf-8")).get("species"):
                    fait += 1
            except (OSError, json.JSONDecodeError):
                pass
        self.esp_compteur.config(text=f"{fait} / {len(self.esp_fichiers)} annotées")

    def _esp_afficher(self, chemin: Path):
        from PIL import Image, ImageTk
        self.esp_canvas.update_idletasks()
        cw = max(self.esp_canvas.winfo_width(), 700)
        ch = max(self.esp_canvas.winfo_height(), 320)
        im = Image.open(chemin)
        r = min(cw / im.width, ch / im.height)
        im = im.resize((max(1, int(im.width * r)), max(1, int(im.height * r))),
                       Image.Resampling.LANCZOS)
        self._esp_photo = ImageTk.PhotoImage(im)
        self.esp_canvas.delete("all")
        self.esp_canvas.create_image(cw // 2, ch // 2, image=self._esp_photo)

    def _esp_choisir(self, essence: str | None):
        if self.esp_index >= len(self.esp_fichiers):
            return
        if essence is not None:
            f = self.esp_fichiers[self.esp_index]
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
                d["species"] = essence
                d["wood_family"] = ("feuillu" if essence in ESSENCES_FEUILLUS
                                    else "resineux")
                f.write_text(json.dumps(d, indent=2, ensure_ascii=False),
                             encoding="utf-8")
            except (OSError, json.JSONDecodeError) as exc:
                messagebox.showerror("Erreur", str(exc))
                return
        self._esp_suivant()

    # ---- tâches : formulaire + console intégrée ----------------------------

    def _colonne_defilante(self, parent: tk.Frame, largeur: int) -> tk.Frame:
        """Colonne qui défile, et renvoie le cadre où empiler les cartes.

        `pack` ne prévient pas quand il manque de place : il rogne la dernière
        carte sans rien dire. « Données » perdait ainsi une tâche à la taille
        par défaut et trois à la taille minimale, sans que rien ne signale
        qu'il en existait d'autres. La barre, elle, se voit.
        """
        colonne = tk.Frame(parent, bg=T.CONTENT_BG, width=largeur)
        colonne.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 14))
        colonne.pack_propagate(False)

        barre = tk.Scrollbar(colonne, orient=tk.VERTICAL)
        volet = tk.Canvas(colonne, bg=T.CONTENT_BG, highlightthickness=0,
                          yscrollcommand=barre.set)
        barre.config(command=volet.yview)
        volet.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        dedans = tk.Frame(volet, bg=T.CONTENT_BG)
        fenetre = volet.create_window((0, 0), window=dedans, anchor="nw")

        def ajuster(_e=None):
            volet.configure(scrollregion=volet.bbox("all"))
            volet.itemconfigure(fenetre, width=volet.winfo_width())
            # La barre ne s'affiche que si elle sert : sur « Entraîner », qui
            # tient à l'aise, elle ne vient pas manger la largeur des cartes.
            if dedans.winfo_reqheight() > volet.winfo_height():
                if not barre.winfo_ismapped():
                    barre.pack(side=tk.RIGHT, fill=tk.Y, before=volet)
            elif barre.winfo_ismapped():
                barre.pack_forget()

        dedans.bind("<Configure>", ajuster)
        volet.bind("<Configure>", ajuster)

        def molette(e):
            if dedans.winfo_reqheight() > volet.winfo_height():
                volet.yview_scroll(
                    1 if getattr(e, "num", 0) == 5 or e.delta < 0 else -1,
                    "units")

        # La molette n'est captée que sous le pointeur, pour laisser la
        # console et le reste de la vue garder la leur.
        roulettes = ("<MouseWheel>", "<Button-4>", "<Button-5>")
        colonne.bind("<Enter>",
                     lambda _e: [volet.bind_all(s, molette) for s in roulettes])
        colonne.bind("<Leave>",
                     lambda _e: [volet.unbind_all(s) for s in roulettes])
        return dedans

    def _vue_taches(self, domaine: str) -> tk.Frame:
        titres = {"roi": ("Localisation de la planche",
                          "Annoter les coins, tester le modèle ROI"),
                  "donnees": ("Données", "Constituer et préparer les jeux"),
                  "entrainer": ("Entraînement", "Détecteurs et reconnaissance d'essence"),
                  "analyser": ("Analyse", "Vidéo, classement, comparaison")}
        f = tk.Frame(self.zone, bg=T.CONTENT_BG)
        self._entete(f, *titres[domaine])

        corps = tk.Frame(f, bg=T.CONTENT_BG)
        corps.pack(fill=tk.BOTH, expand=True, padx=18, pady=14)

        gauche = self._colonne_defilante(corps, 256)
        for t in CATALOGUE[domaine]:
            c = tk.Frame(gauche, bg=T.CARD_BG, highlightthickness=1,
                         highlightbackground=T.CARD_BORDURE, cursor="hand2")
            c.pack(fill=tk.X, pady=3)
            tk.Label(c, text=t.nom, bg=T.CARD_BG, fg=T.TEXT_MAIN,
                     font=tkfont.Font(family=self.fam, size=10, weight="bold"),
                     anchor="w", wraplength=220, justify=tk.LEFT).pack(
                fill=tk.X, padx=11, pady=(9, 2))
            tk.Label(c, text=t.resume, bg=T.CARD_BG, fg=T.TEXT_SEC,
                     font=self.f_micro, anchor="w", wraplength=220,
                     justify=tk.LEFT).pack(fill=tk.X, padx=11, pady=(0, 10))
            for w in (c, *c.winfo_children()):
                w.bind("<Button-1>", lambda _e, tt=t: self._choisir_tache(tt))

        droite = tk.Frame(corps, bg=T.CONTENT_BG)
        droite.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.form = tk.Frame(droite, bg=T.CARD_BG, highlightthickness=1,
                             highlightbackground=T.CARD_BORDURE)
        self.form.pack(fill=tk.X)
        tk.Label(self.form, text="Choisissez une tâche à gauche.",
                 bg=T.CARD_BG, fg=T.TEXT_TERTIAIRE, font=self.f_corps).pack(
            padx=20, pady=26)

        boite = tk.Frame(droite, bg=T.CONSOLE_BG)
        boite.pack(fill=tk.BOTH, expand=True, pady=(12, 0))
        sb = tk.Scrollbar(boite)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.console = tk.Text(boite, bg=T.CONSOLE_BG, fg=T.CONSOLE_TXT,
                               font=self.f_mono, wrap=tk.NONE, relief=tk.FLAT,
                               yscrollcommand=sb.set, padx=10, pady=8)
        self.console.pack(fill=tk.BOTH, expand=True)
        sb.config(command=self.console.yview)
        self.console.tag_config("err", foreground=T.CONSOLE_ERR)
        self.console.tag_config("info", foreground=T.CONSOLE_INFO)
        return f

    def _choisir_tache(self, t: Tache):
        self.tache = t
        self.champs.clear()
        for w in self.form.winfo_children():
            w.destroy()
        tk.Label(self.form, text=t.nom, bg=T.CARD_BG, fg=T.TEXT_MAIN,
                 font=self.f_section, anchor="w").pack(
            fill=tk.X, padx=20, pady=(16, 3))
        tk.Label(self.form, text=t.resume, bg=T.CARD_BG, fg=T.TEXT_SEC,
                 font=self.f_petit, anchor="w", wraplength=700,
                 justify=tk.LEFT).pack(fill=tk.X, padx=20)
        if t.fenetre_propre:
            fp = tk.Frame(self.form, bg=T.TIP_BG)
            fp.pack(fill=tk.X, padx=20, pady=(9, 0))
            tk.Label(fp, bg=T.TIP_BG, fg=T.PRIMARY, font=self.f_petit,
                     anchor="w", padx=10, pady=7, wraplength=680,
                     justify=tk.LEFT,
                     text="Cet outil ouvre sa propre fenêtre : il repose sur "
                          "OpenCV et n'a pas pu être intégré sans être "
                          "réécrit. Fermez-la pour revenir ici.").pack(fill=tk.X)

        if t.note:
            n = tk.Frame(self.form, bg=T.ALERTE_BG)
            n.pack(fill=tk.X, padx=20, pady=(9, 0))
            tk.Label(n, text=t.note, bg=T.ALERTE_BG, fg=T.ALERTE,
                     font=self.f_petit, anchor="w", wraplength=680,
                     justify=tk.LEFT, padx=10, pady=7).pack(fill=tk.X)

        g = tk.Frame(self.form, bg=T.CARD_BG)
        g.pack(fill=tk.X, padx=20, pady=12)
        g.columnconfigure(1, weight=1)
        memo = self.session.get(t.cle, {})
        ligne = 0
        for p in ([t.positionnel] if t.positionnel else []) + t.params:
            ligne = self._champ(g, p, ligne, memo)

        act = tk.Frame(self.form, bg=T.CARD_BG)
        act.pack(fill=tk.X, padx=20, pady=(0, 16))
        self.btn_run = tk.Button(
            act, text="Lancer", command=self._lancer, bg=T.PRIMARY, fg="white",
            relief=tk.FLAT, font=tkfont.Font(family=self.fam, size=10, weight="bold"),
            padx=26, pady=8, cursor="hand2", activebackground=T.ACTIVE_BG,
            activeforeground="white")
        self.btn_run.pack(side=tk.LEFT)
        self.btn_stop = tk.Button(act, text="Arrêter", command=self._arreter,
                                  bg=T.ERREUR, fg="white", relief=tk.FLAT,
                                  font=self.f_petit, padx=18, pady=8,
                                  state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=9)

    def _champ(self, parent, p: Param, ligne: int, memo: dict) -> int:
        cle = p.flag or "_pos"
        val = memo.get(cle, p.defaut)
        tk.Label(parent, text=p.label, bg=T.CARD_BG, fg=T.TEXT_MAIN,
                 font=self.f_petit, anchor="w").grid(
            row=ligne, column=0, sticky="w", pady=4, padx=(0, 12))
        if p.type == "drapeau":
            var = tk.BooleanVar(value=bool(val))
            tk.Checkbutton(parent, variable=var, bg=T.CARD_BG,
                           activebackground=T.CARD_BG).grid(
                row=ligne, column=1, sticky="w")
        elif p.type == "choix":
            var = tk.StringVar(value=val)
            ttk.Combobox(parent, textvariable=var, values=list(p.choix),
                         state="readonly", width=20).grid(
                row=ligne, column=1, sticky="w")
        else:
            var = tk.StringVar(value=val)
            cadre = tk.Frame(parent, bg=T.CARD_BG)
            cadre.grid(row=ligne, column=1, sticky="ew")
            cadre.columnconfigure(0, weight=1)
            tk.Entry(cadre, textvariable=var, font=self.f_petit).grid(
                row=0, column=0, sticky="ew", ipady=2)
            if p.type in ("fichier", "dossier"):
                tk.Button(cadre, text="…", width=3, relief=tk.FLAT,
                          bg=T.TIP_BG, fg=T.PRIMARY, cursor="hand2",
                          command=lambda v=var, tt=p.type: self._parcourir(v, tt)).grid(
                    row=0, column=1, padx=(5, 0))
        self.champs[cle] = var
        if p.aide:
            tk.Label(parent, text=p.aide, bg=T.CARD_BG, fg=T.TEXT_TERTIAIRE,
                     font=self.f_micro, anchor="w").grid(
                row=ligne + 1, column=1, sticky="w")
            return ligne + 2
        return ligne + 1

    def _parcourir(self, var: tk.StringVar, type_: str):
        c = (filedialog.askdirectory(initialdir=str(self.ws)) if type_ == "dossier"
             else filedialog.askopenfilename(initialdir=str(self.ws)))
        if c:
            try:
                c = str(Path(c).relative_to(self.ws))
            except ValueError:
                pass
            var.set(c)

    # ---- exécution ----------------------------------------------------------

    def _lancer(self):
        if self.proc is not None:
            messagebox.showinfo("En cours", "Une tâche tourne déjà.")
            return
        t = self.tache
        # Les outils de l'atelier vivent ici, le runtime dans l'espace de
        # travail. Le prefixe du chemin dit lequel des deux on lance.
        script = (RACINE / t.script if t.script.startswith("outils/")
                  else self.ws / t.script)
        if not script.exists():
            self._ecrire(f"Script introuvable : {script}\n", True, "err")
            return
        cmd = [sys.executable, str(script)]
        memo = {}
        if t.positionnel:
            v = self.champs["_pos"].get().strip()
            memo["_pos"] = v
            if not v:
                messagebox.showwarning("Paramètre manquant",
                                       f"« {t.positionnel.label} » est requis.")
                return
            cmd += v.split()
        for p in t.params:
            var = self.champs[p.flag]
            if p.type == "drapeau":
                memo[p.flag] = bool(var.get())
                if var.get():
                    cmd.append(p.flag)
            else:
                v = str(var.get()).strip()
                memo[p.flag] = v
                if v:
                    cmd += [p.flag, *v.split()]
        self.session[t.cle] = memo
        W.enregistrer_session(self.session)

        self._ecrire("", True)
        self._ecrire("$ " + " ".join(cmd[1:]) + "\n\n", False, "info")
        try:
            self.proc = subprocess.Popen(
                cmd, cwd=str(self.ws), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                errors="replace", bufsize=1,
                env={**os.environ, "PYTHONUNBUFFERED": "1",
                     "PYTHONIOENCODING": "utf-8",
                     "BEAVER_WORKSPACE": str(self.ws)})
        except OSError as exc:
            self._ecrire(f"Lancement impossible : {exc}\n", False, "err")
            return
        self.btn_run.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        threading.Thread(target=self._lire, daemon=True).start()

    def _lire(self):
        try:
            for ligne in self.proc.stdout:
                self.flux.put(ligne)
        except (OSError, ValueError):
            pass
        self.flux.put(f"\x00{self.proc.wait() if self.proc else -1}")

    def _pomper(self):
        try:
            while True:
                item = self.flux.get_nowait()
                if item.startswith("\x00"):
                    code = item[1:]
                    self._ecrire(f"\n[terminé, code {code}]\n", False,
                                 "info" if code == "0" else "err")
                    self.proc = None
                    try:
                        self.btn_run.config(state=tk.NORMAL)
                        self.btn_stop.config(state=tk.DISABLED)
                    except tk.TclError:
                        pass
                    if self.active == "accueil":
                        self._rafraichir()
                else:
                    self._ecrire(item)
        except queue.Empty:
            pass
        self._pompe = self.after(80, self._pomper)

    def _ecrire(self, texte: str, effacer: bool = False, tag: str = ""):
        if not hasattr(self, "console"):
            return
        try:
            self.console.config(state=tk.NORMAL)
            if effacer:
                self.console.delete("1.0", tk.END)
            if texte:
                self.console.insert(tk.END, texte, tag)
                self.console.see(tk.END)
            self.console.config(state=tk.DISABLED)
        except tk.TclError:
            pass

    def _arreter(self):
        if self.proc:
            self.proc.terminate()
            self._ecrire("\n[arrêt demandé]\n", False, "err")

    def _quitter(self):
        if self.annotateur is not None:
            try:
                self.annotateur._autosave()
            except Exception:  # noqa: BLE001
                pass
        if self.proc and not messagebox.askyesno(
                "Tâche en cours", "Une tâche tourne. Quitter quand même ?"):
            return
        if self.proc:
            self.proc.terminate()
        if self._pompe is not None:
            self.after_cancel(self._pompe)
            self._pompe = None
        self.destroy()
