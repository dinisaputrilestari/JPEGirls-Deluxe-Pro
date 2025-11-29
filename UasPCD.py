import tkinter as tk
from tkinter import Menu, filedialog, messagebox, simpledialog, Scale, Button, Label, Toplevel, Frame
from PIL import Image, ImageTk, ImageOps, ImageFilter, ImageEnhance, ImageDraw
import numpy as np
import cv2
from scipy import ndimage
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import webbrowser
import math
import random
import platform

# =========================
# JPEGirls Theme / Styling Constants
# =========================
# Palette taken from JPEGirls.py (fresh green)
C_BG = "#f4f7f6"         # overall background
C_TOP_BAR = "#006266"    # top bar (toska)
C_PANEL = "#ffffff"      # panels (white)
C_TEXT_DARK = "#2d3436"
C_TEXT_LIGHT = "#ffffff"
C_BTN = "#00b894"
C_BTN_ACTIVE = "#00897b"
C_BTN_ACCENT = "#81ecec"
C_CANVAS_BG = "#ffffff"
C_CANVAS_BORDER = "#dfe6e9"
C_SLIDER_TROUGH = "#b2bec3"
C_TEXT_SECONDARY = "#747d8c"

FONT_TITLE = ("Segoe UI", 14, "bold")
FONT_SUB = ("Segoe UI", 10)
FONT_SMALL = ("Segoe UI", 9)

class JPEGirlsDeluxePro_UI:
    def __init__(self, root):
        self.root = root
        self.root.title("✨ JPEGirls Deluxe Pro ✨")
        # Try to maximize nicely across platforms
        try:
            self.root.state("zoomed")
        except Exception:
            self.root.geometry("1260x820")
        self.root.configure(bg=C_BG)
        self.root.minsize(1000, 720)

        # state / images
        self.original_image = None
        self.processed_image = None
        self.temp_image = None
        self.image_path = None

        # undo/redo stacks could be added later if needed
        self.zoom = 1.0
        self.rotate_val = 0

        # build UI
        self._build_topbar()
        self._build_layout()
        self._build_menubar()
        self._bind_wheel_events()

    # ---------- Top bar ----------
    def _build_topbar(self):
        top = tk.Frame(self.root, bg=C_TOP_BAR, height=64)
        top.pack(fill="x", side="top")

        lbl = tk.Label(top, text="✨ JPEGirls Deluxe Pro ✨", bg=C_TOP_BAR, fg=C_TEXT_LIGHT,
                       font=("Segoe UI", 16, "bold"))
        lbl.pack(side="left", padx=18)

        sub = tk.Label(top, text="Stylish • Fast • Simple", bg=C_TOP_BAR, fg="#dfe6e9",
                       font=("Segoe UI", 10, "italic"))
        sub.pack(side="right", padx=18)

    # ---------- Menubar ----------
    def _build_menubar(self):
        menubar = Menu(self.root)
        # File
        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open Image...", command=self.open_image)
        filemenu.add_command(label="Save", command=self.save_image)
        filemenu.add_command(label="Save As...", command=self.save_as_image)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.exit_app)
        menubar.add_cascade(label="File", menu=filemenu)

        # Basic ops (keep many entries from original)
        basic = Menu(menubar, tearoff=0)
        basic.add_command(label="Negative", command=self.negative)

        # Arithmetic submenu
        arith = Menu(basic, tearoff=0)
        arith.add_command(label="Add (+)", command=self.arithmetic_add)
        arith.add_command(label="Subtract (-)", command=self.arithmetic_subtract)
        arith.add_command(label="Multiply (*)", command=self.arithmetic_multiply)
        arith.add_command(label="Divide (/)", command=self.arithmetic_divide)
        basic.add_cascade(label="Arithmetic", menu=arith)

        # Boolean submenu
        boolean = Menu(basic, tearoff=0)
        boolean.add_command(label="NOT", command=self.boolean_not)
        boolean.add_command(label="AND", command=self.boolean_and)
        boolean.add_command(label="OR", command=self.boolean_or)
        boolean.add_command(label="XOR", command=self.boolean_xor)
        basic.add_cascade(label="Boolean", menu=boolean)

        # Geometrics submenu
        geom = Menu(basic, tearoff=0)
        geom.add_command(label="Translation", command=self.geometric_translation)
        geom.add_command(label="Rotation", command=self.geometric_rotation)
        geom.add_command(label="Zooming", command=self.geometric_zooming)
        geom.add_command(label="Flipping", command=self.geometric_flipping)
        geom.add_command(label="Cropping", command=self.geometric_cropping)
        basic.add_cascade(label="Geometrics", menu=geom)

        basic.add_command(label="Thresholding", command=self.thresholding)
        basic.add_command(label="Convolution", command=self.convolution)
        basic.add_command(label="Fourier Transform", command=self.fourier_transform)

        # Colouring submenu
        colouring = Menu(basic, tearoff=0)
        colouring.add_command(label="Binary", command=self.color_binary)
        colouring.add_command(label="Grayscale", command=self.color_grayscale)
        colouring.add_command(label="RGB", command=self.color_rgb)
        colouring.add_command(label="HSV", command=self.color_hsv)
        colouring.add_command(label="CMY", command=self.color_cmy)
        colouring.add_command(label="YUV", command=self.color_yuv)
        colouring.add_command(label="YIQ", command=self.color_yiq)
        colouring.add_command(label="Pseudo", command=self.color_pseudo)
        basic.add_cascade(label="Colouring", menu=colouring)

        menubar.add_cascade(label="Basic Ops", menu=basic)

        # Enhancement
        enhancement = Menu(menubar, tearoff=0)
        enhancement.add_command(label="Brightness", command=self.enhance_brightness)
        enhancement.add_command(label="Contrast", command=self.enhance_contrast)
        enhancement.add_command(label="Hist. Equalization", command=self.histogram_equalization)

        smoothing = Menu(enhancement, tearoff=0)
        spatial = Menu(smoothing, tearoff=0)
        spatial.add_command(label="Lowpass Filtering", command=self.smoothing_lowpass)
        spatial.add_command(label="Median Filtering", command=self.smoothing_median)
        smoothing.add_cascade(label="Spatial Domain", menu=spatial)
        freq = Menu(smoothing, tearoff=0)
        freq.add_command(label="ILPF", command=self.smoothing_ilpf)
        freq.add_command(label="BLPF", command=self.smoothing_blpf)
        smoothing.add_cascade(label="Frequency Domain", menu=freq)
        enhancement.add_cascade(label="Smoothing", menu=smoothing)

        sharpening = Menu(enhancement, tearoff=0)
        sh_sp = Menu(sharpening, tearoff=0)
        sh_sp.add_command(label="Highpass Filtering", command=self.sharpening_highpass)
        sh_sp.add_command(label="Highboost Filtering", command=self.sharpening_highboost)
        sharpening.add_cascade(label="Spatial Domain", menu=sh_sp)
        sh_fr = Menu(sharpening, tearoff=0)
        sh_fr.add_command(label="IHPF", command=self.sharpening_ihpf)
        sh_fr.add_command(label="BHPF", command=self.sharpening_bhpf)
        sharpening.add_cascade(label="Frequency Domain", menu=sh_fr)
        enhancement.add_cascade(label="Sharpening", menu=sharpening)

        enhancement.add_command(label="Geometrics Correction", command=self.geometric_correction)
        menubar.add_cascade(label="Enhancement", menu=enhancement)

        # Noise
        noise = Menu(menubar, tearoff=0)
        noise.add_command(label="Gaussian Noise", command=self.noise_gaussian)
        noise.add_command(label="Rayleigh Noise", command=self.noise_rayleigh)
        noise.add_command(label="Erlang (Gamma) Noise", command=self.noise_erlang)
        noise.add_command(label="Exponential Noise", command=self.noise_exponential)
        noise.add_command(label="Uniform Noise", command=self.noise_uniform)
        noise.add_command(label="Impulse Noise", command=self.noise_impulse)
        menubar.add_cascade(label="Noise", menu=noise)

        # Edge Detection
        edge = Menu(menubar, tearoff=0)
        first = Menu(edge, tearoff=0)
        first.add_command(label="Sobel", command=self.edge_sobel)
        first.add_command(label="Prewitt", command=self.edge_prewitt)
        first.add_command(label="Robert", command=self.edge_robert)
        edge.add_cascade(label="1st Differential Gradient", menu=first)
        second = Menu(edge, tearoff=0)
        second.add_command(label="Laplacian", command=self.edge_laplacian)
        second.add_command(label="LoG", command=self.edge_log)
        second.add_command(label="Canny", command=self.edge_canny)
        edge.add_cascade(label="2nd Differential Gradient", menu=second)
        edge.add_command(label="Compass", command=self.edge_compass)
        menubar.add_cascade(label="Edge Detection", menu=edge)

        # Segmentation
        seg = Menu(menubar, tearoff=0)
        seg.add_command(label="Region Growing", command=self.segmentation_region_growing)
        seg.add_command(label="Watershed", command=self.segmentation_watershed)
        menubar.add_cascade(label="Segmentation", menu=seg)

        # About
        about = Menu(menubar, tearoff=0)
        about.add_command(label="Info Tim Developer", command=self.show_info)
        about.add_separator()
        about.add_command(label="Tutorial: Link Github", command=self.open_github)
        about.add_command(label="Tutorial: Link Youtube", command=self.open_youtube)
        menubar.add_cascade(label="About", menu=about)

        self.root.config(menu=menubar)

    # ---------- Layout ----------
    def _build_layout(self):
        container = tk.Frame(self.root, bg=C_BG)
        container.pack(fill="both", expand=True, padx=12, pady=12)

        # Left tools panel (white)
        left_panel = tk.Frame(container, bg=C_PANEL, width=320, bd=0)
        left_panel.pack(side="left", fill="y", padx=(0,12))
        left_panel.pack_propagate(False)

        tk.Label(left_panel, text="Controls", font=FONT_TITLE, bg=C_PANEL, fg=C_TEXT_DARK).pack(pady=(14,6))

        btn_open = tk.Button(left_panel, text="Open Image", command=self.open_image, bg=C_BTN, fg=C_TEXT_LIGHT,
                             activebackground=C_BTN_ACTIVE, font=FONT_SUB, width=20)
        btn_open.pack(pady=8)

        btn_save = tk.Button(left_panel, text="Save", command=self.save_image, bg=C_BTN, fg=C_TEXT_LIGHT,
                             activebackground=C_BTN_ACTIVE, font=FONT_SUB, width=20)
        btn_save.pack(pady=6)

        btn_saveas = tk.Button(left_panel, text="Save As...", command=self.save_as_image, bg=C_BTN_ACCENT, fg=C_TEXT_DARK,
                               activebackground=C_BTN_ACTIVE, font=FONT_SUB, width=20)
        btn_saveas.pack(pady=6)

        btn_reset = tk.Button(left_panel, text="Reset to Original", command=self.reset_to_original, bg=C_BTN, fg=C_TEXT_LIGHT,
                              activebackground=C_BTN_ACTIVE, font=FONT_SUB, width=20)
        btn_reset.pack(pady=6)

        # Quick ops group
        tk.Label(left_panel, text="Quick Tools", font=FONT_SUB, bg=C_PANEL, fg=C_TEXT_SECONDARY).pack(pady=(12,6))
        tk.Button(left_panel, text="Grayscale", command=self.color_grayscale, width=20).pack(pady=4)
        tk.Button(left_panel, text="Histogram EQ", command=self.histogram_equalization, width=20).pack(pady=4)
        tk.Button(left_panel, text="Canny Edge", command=self.edge_canny, width=20).pack(pady=4)

        # Middle and right: canvases area (use a frame with two cards)
        canvases_frame = tk.Frame(container, bg=C_BG)
        canvases_frame.pack(side="left", fill="both", expand=True)

        # original canvas card (left)
        card_orig = tk.Frame(canvases_frame, bg=C_PANEL, bd=1, relief="solid")
        card_orig.pack(side="left", padx=(0,12), pady=6, fill="y")
        card_orig.pack_propagate(False)
        card_orig.configure(width=420, height=720)

        tk.Label(card_orig, text="Original Image", font=("Segoe UI", 12, "bold"), bg=C_PANEL, fg=C_TEXT_DARK).pack(pady=(12,6))
        self.canvas_original = tk.Canvas(card_orig, bg=C_CANVAS_BG, width=380, height=660, highlightthickness=0)
        self.canvas_original.pack(padx=12, pady=(4,12))
        # allow large scroll region so image can go outside visible area
        self.canvas_original.config(scrollregion=(0, 0, 5000, 5000))

        # processed canvas card (right)
        card_proc = tk.Frame(canvases_frame, bg=C_PANEL, bd=1, relief="solid")
        card_proc.pack(side="left", fill="both", expand=True, pady=6)
        card_proc.pack_propagate(False)
        card_proc.configure(width=760, height=720)

        tk.Label(card_proc, text="Processed Image", font=("Segoe UI", 12, "bold"), bg=C_PANEL, fg=C_TEXT_DARK).pack(pady=(12,6))
        self.canvas_processed = tk.Canvas(card_proc, bg=C_CANVAS_BG, width=720, height=660, highlightthickness=0)
        self.canvas_processed.pack(padx=12, pady=(4,12), fill="both", expand=True)
        # allow large scroll region so image can go outside visible area
        self.canvas_processed.config(scrollregion=(0, 0, 5000, 5000))

        # footer small
        footer = tk.Frame(self.root, bg=C_BG, height=36)
        footer.pack(fill="x", side="bottom")
        footer.pack_propagate(False)
        tk.Label(footer, text="Tim: PixA", font=FONT_SMALL, bg=C_BG, fg=C_TEXT_SECONDARY).pack(side="left", padx=12)
        tk.Label(footer, text="Tips: Open > pilih gambar > pilih operasi", font=FONT_SMALL, bg=C_BG, fg=C_TEXT_SECONDARY).pack(side="right", padx=12)

    def _bind_wheel_events(self):
        system = platform.system()
        if system in ["Windows", "Darwin"]:
            self.canvas_processed.bind_all("<MouseWheel>", self.on_mouse_wheel)
        else:
            self.canvas_processed.bind_all("<Button-4>", lambda e: self._zoom(1.1))
            self.canvas_processed.bind_all("<Button-5>", lambda e: self._zoom(0.9))

    def on_mouse_wheel(self, event):
        if event.delta > 0:
            self._zoom(1.1)
        else:
            self._zoom(0.9)

    def _zoom(self, factor):
        self.zoom *= factor
        # only affects display resizing; keep processed image intact
        self.display_images()
        # update scrollregion after zoom so image can overflow
        try:
            bbox = self.canvas_processed.bbox("all")
            if bbox:
                self.canvas_processed.config(scrollregion=bbox)
        except Exception:
            pass

    # ========== Utility: slider dialog (styled JPEGirls) ==========
    def create_slider_dialog(self, title, label_text, min_val, max_val, default_val, resolution=1, callback=None):
        dialog = Toplevel(self.root)
        dialog.title(title)
        dialog.geometry("460x220")
        dialog.resizable(False, False)
        dialog.configure(bg=C_BG)
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(dialog, text=label_text, font=("Segoe UI", 11, "bold"), bg=C_BG, fg=C_TEXT_DARK).pack(pady=(12,6))
        value_var = tk.DoubleVar(value=default_val)
        result = {'value': None, 'confirmed': False}

        value_label = tk.Label(dialog, text=f"Value: {default_val}", font=FONT_SUB, bg=C_BG, fg=C_TEXT_SECONDARY)
        value_label.pack(pady=6)

        def on_slider_change(val):
            try:
                v = float(val)
                display = f"{v:.2f}" if isinstance(resolution, float) else f"{int(round(v))}"
            except:
                display = val
            value_label.config(text=f"Value: {display}")
            if callback:
                try:
                    callback(float(val))
                except Exception:
                    pass

        slider = Scale(dialog, from_=min_val, to=max_val, orient=tk.HORIZONTAL,
                       resolution=resolution, length=380, variable=value_var,
                       command=on_slider_change, bg=C_BG, troughcolor=C_SLIDER_TROUGH)
        slider.pack(pady=6)

        btn_frame = tk.Frame(dialog, bg=C_BG)
        btn_frame.pack(pady=12)

        def on_ok():
            result['value'] = value_var.get()
            result['confirmed'] = True
            dialog.destroy()

        def on_reset():
            result['value'] = None
            result['confirmed'] = False
            dialog.destroy()

        btn_ok = tk.Button(btn_frame, text="OK", command=on_ok, width=12, bg=C_BTN, fg=C_TEXT_LIGHT, activebackground=C_BTN_ACTIVE)
        btn_ok.pack(side=tk.LEFT, padx=10)
        btn_reset = tk.Button(btn_frame, text="Reset", command=on_reset, width=12, bg=C_BTN_ACCENT, fg=C_TEXT_DARK, activebackground=C_BTN_ACTIVE)
        btn_reset.pack(side=tk.LEFT, padx=10)

        dialog.wait_window()
        return result

    # ========== File functions ==========
    def open_image(self):
        file_path = filedialog.askopenfilename(
            title="Pilih Gambar",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All Files", "*.*")]
        )
        if file_path:
            try:
                self.image_path = file_path
                self.original_image = Image.open(file_path).convert("RGB")
                self.processed_image = self.original_image.copy()
                self.temp_image = None
                self.zoom = 1.0
                self.display_images()
            except Exception as e:
                messagebox.showerror("Error", f"Gagal membuka gambar: {e}")

    def save_image(self):
        if self.processed_image:
            if self.image_path:
                try:
                    self.processed_image.save(self.image_path)
                    messagebox.showinfo("Success", "Image saved successfully!")
                except Exception as e:
                    messagebox.showerror("Error", f"Gagal menyimpan: {e}")
            else:
                self.save_as_image()
        else:
            messagebox.showwarning("Warning", "No processed image to save!")

    def save_as_image(self):
        if self.processed_image:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp"), ("All Files", "*.*")]
            )
            if file_path:
                try:
                    self.processed_image.save(file_path)
                    messagebox.showinfo("Success", "Image saved successfully!")
                except Exception as e:
                    messagebox.showerror("Error", f"Gagal menyimpan: {e}")
        else:
            messagebox.showwarning("Warning", "No processed image to save!")

    def exit_app(self):
        if messagebox.askokcancel("Exit", "Do you want to exit?"):
            self.root.destroy()

    def reset_to_original(self):
        if self.original_image:
            self.processed_image = self.original_image.copy()
            self.temp_image = None
            self.zoom = 1.0
            self.display_images()
            # update scrollregion
            try:
                bbox = self.canvas_processed.bbox("all")
                if bbox:
                    self.canvas_processed.config(scrollregion=bbox)
            except Exception:
                pass

    # ========== Display helpers ==========
    def resize_for_canvas(self, image, max_width, max_height):
        img_width, img_height = image.size
        ratio = min(max_width/img_width, max_height/img_height)
        new_size = (int(img_width*ratio*self.zoom), int(img_height*ratio*self.zoom))
        if new_size[0] < 1 or new_size[1] < 1:
            new_size = (1,1)
        return image.resize(new_size, Image.Resampling.LANCZOS)

    def display_images(self, no_fit=False):
        # clear canvases
        try:
            self.canvas_original.delete("all")
            self.canvas_processed.delete("all")
        except Exception:
            pass

        if self.original_image:
            # fit original into left canvas
            w = max(10, self.canvas_original.winfo_width()) or 380
            h = max(10, self.canvas_original.winfo_height()) or 600
            orig_resized = self.resize_for_canvas(self.original_image, w-20, h-20)
            orig_photo = ImageTk.PhotoImage(orig_resized)
            self.canvas_original.create_rectangle(2,2,w-2,h-2, outline=C_CANVAS_BORDER, width=1)
            # place image centered (but scrollregion allows overflow)
            self.canvas_original.create_image(self.canvas_original.winfo_width()//2, self.canvas_original.winfo_height()//2,
                                              image=orig_photo, anchor="center")
            self.canvas_original.image = orig_photo
            # update scrollregion to include full image area (if available)
            try:
                bbox = self.canvas_original.bbox("all")
                if bbox:
                    self.canvas_original.config(scrollregion=bbox)
            except Exception:
                pass

        if self.processed_image:
            w = max(10, self.canvas_processed.winfo_width()) or 720
            h = max(10, self.canvas_processed.winfo_height()) or 600
            proc_resized = self.resize_for_canvas(self.processed_image, w-20, h-20)
            proc_photo = ImageTk.PhotoImage(proc_resized)
            self.canvas_processed.create_rectangle(2,2,w-2,h-2, outline=C_CANVAS_BORDER, width=1)
            # place image centered (but allow overflow via scrollregion)
            self.canvas_processed.create_image(self.canvas_processed.winfo_width()//2, self.canvas_processed.winfo_height()//2,
                                               image=proc_photo, anchor="center")
            self.canvas_processed.image = proc_photo
            # update scrollregion to include full image area (if available)
            try:
                bbox = self.canvas_processed.bbox("all")
                if bbox:
                    self.canvas_processed.config(scrollregion=bbox)
                else:
                    # fallback: set a generous area so image can overflow
                    self.canvas_processed.config(scrollregion=(0,0,5000,5000))
            except Exception:
                pass

    def display_temp_image(self, no_fit=False):
        # show temp image to processed canvas
        if self.temp_image is None:
            return
        try:
            self.canvas_processed.delete("all")
        except Exception:
            pass
        w = max(10, self.canvas_processed.winfo_width()) or 720
        h = max(10, self.canvas_processed.winfo_height()) or 600
        proc_resized = self.temp_image  # tampilkan ukuran asli tanpa auto-fit
        proc_photo = ImageTk.PhotoImage(proc_resized)
        self.canvas_processed.create_rectangle(2,2,w-2,h-2, outline=C_CANVAS_BORDER, width=1)
        self.canvas_processed.create_image(self.canvas_processed.winfo_width()//2, self.canvas_processed.winfo_height()//2,
                                           image=proc_photo, anchor="center")
        self.canvas_processed.image = proc_photo
        # update scrollregion after preview
        try:
            bbox = self.canvas_processed.bbox("all")
            if bbox:
                self.canvas_processed.config(scrollregion=bbox)
        except Exception:
            pass

    def check_image_loaded(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return False
        return True

    # ========== BASIC OPS & ALL PROCESSING FUNCTIONS (copied & preserved from original) ==========
    # Negative
    def negative(self):
        if not self.check_image_loaded(): return

        def preview_negative(val):
            strength = val / 100.0
            img_array = np.array(self.original_image.convert("RGB"), dtype=np.float32)
            inverted = 255 - img_array
            result = img_array + strength * (inverted - img_array)
            result = np.clip(result, 0, 255).astype(np.uint8)
            self.temp_image = Image.fromarray(result)
            self.display_temp_image()

        result = self.create_slider_dialog("Negative", "Negative: 0-100%", 0, 100, 100, 1, preview_negative)

        if result['confirmed'] and result['value'] is not None:
            strength = result['value'] / 100.0
            img_array = np.array(self.original_image.convert("RGB"), dtype=np.float32)
            inverted = 255 - img_array
            final_result = img_array + strength * (inverted - img_array)
            final_result = np.clip(final_result, 0, 255).astype(np.uint8)
            self.processed_image = Image.fromarray(final_result)
        else:
            self.processed_image = self.original_image.copy()

        self.display_images()

    # Arithmetic
    def arithmetic_add(self):
        if not self.check_image_loaded(): return

        def preview_add(val):
            img_array = np.array(self.original_image.convert("RGB"), dtype=np.float32)
            result = np.clip(img_array + val, 0, 255).astype(np.uint8)
            self.temp_image = Image.fromarray(result)
            self.display_temp_image()

        result = self.create_slider_dialog("Add", "Add Value: 0-255", 0, 255, 50, 1, preview_add)
        if result['confirmed'] and result['value'] is not None:
            img_array = np.array(self.original_image.convert("RGB"), dtype=np.float32)
            final_result = np.clip(img_array + result['value'], 0, 255).astype(np.uint8)
            self.processed_image = Image.fromarray(final_result)
        else:
            self.processed_image = self.original_image.copy()
        self.display_images()

    def arithmetic_subtract(self):
        if not self.check_image_loaded(): return

        def preview_subtract(val):
            img_array = np.array(self.original_image.convert("RGB"), dtype=np.float32)
            result = np.clip(img_array - val, 0, 255).astype(np.uint8)
            self.temp_image = Image.fromarray(result)
            self.display_temp_image()

        result = self.create_slider_dialog("Subtract", "Subtract Value: 0-255", 0, 255, 50, 1, preview_subtract)
        if result['confirmed'] and result['value'] is not None:
            img_array = np.array(self.original_image.convert("RGB"), dtype=np.float32)
            final_result = np.clip(img_array - result['value'], 0, 255).astype(np.uint8)
            self.processed_image = Image.fromarray(final_result)
        else:
            self.processed_image = self.original_image.copy()
        self.display_images()

    def arithmetic_multiply(self):
        if not self.check_image_loaded(): return

        def preview_multiply(val):
            img_array = np.array(self.original_image.convert("RGB"), dtype=np.float32)
            result = np.clip(img_array * val, 0, 255).astype(np.uint8)
            self.temp_image = Image.fromarray(result)
            self.display_temp_image()

        result = self.create_slider_dialog("Multiply", "Multiply Factor: 0.1-5.0", 0.1, 5.0, 1.0, 0.1, preview_multiply)
        if result['confirmed'] and result['value'] is not None:
            img_array = np.array(self.original_image.convert("RGB"), dtype=np.float32)
            final_result = np.clip(img_array * result['value'], 0, 255).astype(np.uint8)
            self.processed_image = Image.fromarray(final_result)
        else:
            self.processed_image = self.original_image.copy()
        self.display_images()

    def arithmetic_divide(self):
        if not self.check_image_loaded(): return

        def preview_divide(val):
            img_array = np.array(self.original_image.convert("RGB"), dtype=np.float32)
            if val == 0: val = 1e-3
            result = np.clip(img_array / val, 0, 255).astype(np.uint8)
            self.temp_image = Image.fromarray(result)
            self.display_temp_image()

        result = self.create_slider_dialog("Divide", "Divide Factor: 0.1-5.0", 0.1, 5.0, 1.0, 0.1, preview_divide)
        if result['confirmed'] and result['value'] is not None:
            val = result['value']
            if val == 0: val = 1e-3
            img_array = np.array(self.original_image.convert("RGB"), dtype=np.float32)
            final_result = np.clip(img_array / val, 0, 255).astype(np.uint8)
            self.processed_image = Image.fromarray(final_result)
        else:
            self.processed_image = self.original_image.copy()
        self.display_images()

    # Boolean ops
    def boolean_not(self):
        if not self.check_image_loaded(): return

        def preview_not(val):
            strength = val / 100.0
            img_array = np.array(self.original_image.convert("L"), dtype=np.float32)
            inverted = 255 - img_array
            result = img_array + strength * (inverted - img_array)
            result = np.clip(result, 0, 255).astype(np.uint8)
            self.temp_image = Image.fromarray(result)
            self.display_temp_image()

        result = self.create_slider_dialog("Boolean NOT", "NOT Strength: 0-100%", 0, 100, 100, 1, preview_not)
        if result['confirmed'] and result['value'] is not None:
            strength = result['value'] / 100.0
            img_array = np.array(self.original_image.convert("L"), dtype=np.float32)
            inverted = 255 - img_array
            final_result = img_array + strength * (inverted - img_array)
            final_result = np.clip(final_result, 0, 255).astype(np.uint8)
            self.processed_image = Image.fromarray(final_result)
        else:
            self.processed_image = self.original_image.copy()
        self.display_images()

    def boolean_and(self):
        if not self.check_image_loaded(): return
        file_path = filedialog.askopenfilename(
            title="Pilih Gambar Kedua untuk Operasi AND",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All Files", "*.*")]
        )
        if file_path:
            img2 = Image.open(file_path).convert("L")
            img2 = img2.resize(self.original_image.size)
            img1_gray = np.array(self.original_image.convert("L"))
            img2_gray = np.array(img2)
            result = np.bitwise_and(img1_gray, img2_gray)
            self.processed_image = Image.fromarray(result)
            self.display_images()

    def boolean_or(self):
        if not self.check_image_loaded(): return
        file_path = filedialog.askopenfilename(
            title="Pilih Gambar Kedua untuk Operasi OR",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All Files", "*.*")]
        )
        if file_path:
            img2 = Image.open(file_path).convert("L")
            img2 = img2.resize(self.original_image.size)
            img1_gray = np.array(self.original_image.convert("L"))
            img2_gray = np.array(img2)
            result = np.bitwise_or(img1_gray, img2_gray)
            self.processed_image = Image.fromarray(result)
            self.display_images()

    def boolean_xor(self):
        if not self.check_image_loaded(): return
        file_path = filedialog.askopenfilename(
            title="Pilih Gambar Kedua untuk Operasi XOR",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All Files", "*.*")]
        )
        if file_path:
            img2 = Image.open(file_path).convert("L")
            img2 = img2.resize(self.original_image.size)
            img1_gray = np.array(self.original_image.convert("L"))
            img2_gray = np.array(img2)
            result = np.bitwise_xor(img1_gray, img2_gray)
            self.processed_image = Image.fromarray(result)
            self.display_images()

    # Geometric
    def geometric_translation(self):
        if not self.check_image_loaded(): return

        result_x = self.create_slider_dialog("Translation X", "X Translation: -500 to 500", -500, 500, 0, 1)
        if not result_x['confirmed']:
            return

        result_y = self.create_slider_dialog("Translation Y", "Y Translation: -500 to 500", -500, 500, 0, 1)
        if not result_y['confirmed']:
            return

        tx = int(result_x['value'])
        ty = int(result_y['value'])
        w, h = self.original_image.size
        mat = (1, 0, tx, 0, 1, ty)
        # keep same output size but allow content to be shifted; update display and scrollregion after
        self.processed_image = self.original_image.transform((w, h), Image.AFFINE, mat)
        self.display_images()
        # ensure scrollregion updated so shifted content can be outside visible area
        try:
            bbox = self.canvas_processed.bbox("all")
            if bbox:
                self.canvas_processed.config(scrollregion=bbox)
            else:
                self.canvas_processed.config(scrollregion=(0,0,5000,5000))
        except Exception:
            pass

    def geometric_rotation(self):
        if not self.check_image_loaded(): return

        def preview_rotation(val):
            rotated = self.original_image.rotate(val, expand=True)
            self.temp_image = rotated
            self.display_temp_image()

        result = self.create_slider_dialog("Rotation", "Rotation Angle: -360 to 360", -360, 360, 0, 1, preview_rotation)
        if result['confirmed'] and result['value'] is not None:
            self.processed_image = self.original_image.rotate(result['value'], expand=True)
        else:
            self.processed_image = self.original_image.copy()
        self.display_images()
        try:
            bbox = self.canvas_processed.bbox("all")
            if bbox:
                self.canvas_processed.config(scrollregion=bbox)
        except Exception:
            pass

    def geometric_zooming(self):
        if not self.check_image_loaded(): return

        self.force_no_autofit = True

        def preview_zoom(val):
            new_size = (int(self.original_image.width * val), int(self.original_image.height * val))
            zoomed = self.original_image.resize(new_size, Image.Resampling.LANCZOS)
            self.temp_image = zoomed
            self.display_temp_image(no_fit=True)

        result = self.create_slider_dialog("Zooming", "Zoom Factor: 0.1-5.0", 0.1, 5.0, 1.0, 0.1, preview_zoom)
        if result['confirmed'] and result['value'] is not None:
            new_size = (int(self.original_image.width * result['value']), int(self.original_image.height * result['value']))
            self.processed_image = self.original_image.resize(new_size, Image.Resampling.LANCZOS)
        else:
            self.processed_image = self.original_image.copy()
        self.display_images(no_fit=True)
        # update scrollregion so enlarged image can overflow canvas
        try:
            bbox = self.canvas_processed.bbox("all")
            if bbox:
                self.canvas_processed.config(scrollregion=bbox)
            else:
                self.canvas_processed.config(scrollregion=(0,0,5000,5000))
        except Exception:
            pass

    def geometric_flipping(self):
        if not self.check_image_loaded(): return

        dialog = Toplevel(self.root)
        dialog.title("Flipping")
        dialog.geometry("320x150")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.configure(bg=C_BG)

        tk.Label(dialog, text="Select Flip Direction:", font=("Segoe UI", 10, "bold"), bg=C_BG, fg=C_TEXT_DARK).pack(pady=12)
        result = {'value': None}

        def on_horizontal():
            result['value'] = 'horizontal'
            dialog.destroy()

        def on_vertical():
            result['value'] = 'vertical'
            dialog.destroy()

        btn_frame = tk.Frame(dialog, bg=C_BG)
        btn_frame.pack(pady=6)
        tk.Button(btn_frame, text="Horizontal", command=on_horizontal, width=12, bg=C_BTN, fg=C_TEXT_LIGHT).pack(side=tk.LEFT, padx=8)
        tk.Button(btn_frame, text="Vertical", command=on_vertical, width=12, bg=C_BTN, fg=C_TEXT_LIGHT).pack(side=tk.LEFT, padx=8)

        dialog.wait_window()

        if result['value'] == 'horizontal':
            self.processed_image = self.original_image.transpose(Image.FLIP_LEFT_RIGHT)
        elif result['value'] == 'vertical':
            self.processed_image = self.original_image.transpose(Image.FLIP_TOP_BOTTOM)
        else:
            self.processed_image = self.original_image.copy()
        self.display_images()
        try:
            bbox = self.canvas_processed.bbox("all")
            if bbox:
                self.canvas_processed.config(scrollregion=bbox)
        except Exception:
            pass

    def geometric_cropping(self):
        if not self.check_image_loaded(): return

        x1 = simpledialog.askinteger("Crop", "Enter X1 (left):", initialvalue=0)
        if x1 is None: return
        y1 = simpledialog.askinteger("Crop", "Enter Y1 (top):", initialvalue=0)
        if y1 is None: return
        x2 = simpledialog.askinteger("Crop", "Enter X2 (right):", initialvalue=self.original_image.width)
        if x2 is None: return
        y2 = simpledialog.askinteger("Crop", "Enter Y2 (bottom):", initialvalue=self.original_image.height)
        if y2 is None: return

        self.processed_image = self.original_image.crop((x1, y1, x2, y2))
        self.display_images()
        try:
            bbox = self.canvas_processed.bbox("all")
            if bbox:
                self.canvas_processed.config(scrollregion=bbox)
        except Exception:
            pass

    # Thresholding & Convolution & Fourier
    def thresholding(self):
        if not self.check_image_loaded(): return

        def preview_threshold(val):
            img_gray = np.array(self.original_image.convert("L"))
            _, result = cv2.threshold(img_gray, int(val), 255, cv2.THRESH_BINARY)
            self.temp_image = Image.fromarray(result)
            self.display_temp_image()

        result = self.create_slider_dialog("Thresholding", "Threshold Value: 0-255", 0, 255, 127, 1, preview_threshold)
        if result['confirmed'] and result['value'] is not None:
            img_gray = np.array(self.original_image.convert("L"))
            _, final_result = cv2.threshold(img_gray, int(result['value']), 255, cv2.THRESH_BINARY)
            self.processed_image = Image.fromarray(final_result)
        else:
            self.processed_image = self.original_image.copy()
        self.display_images()

    def convolution(self):
        if not self.check_image_loaded(): return
        kernel = np.array([[-1, -1, -1],
                          [-1,  8, -1],
                          [-1, -1, -1]])
        img_array = np.array(self.original_image.convert("L"), dtype=np.float32)
        result = ndimage.convolve(img_array, kernel)
        result = np.clip(result, 0, 255).astype(np.uint8)
        self.processed_image = Image.fromarray(result)
        self.display_images()

    def fourier_transform(self):
        if not self.check_image_loaded(): return
        img_gray = np.array(self.original_image.convert("L"), dtype=np.float32)
        f = fft2(img_gray)
        fshift = fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        magnitude_spectrum = np.clip(magnitude_spectrum, 0, 255).astype(np.uint8)
        self.processed_image = Image.fromarray(magnitude_spectrum)
        self.display_images()

    # ========== COLOR OPERATIONS ==========
    def color_binary(self):
        if not self.check_image_loaded(): return
        def preview_binary(val):
            img_gray = np.array(self.original_image.convert("L"))
            _, result = cv2.threshold(img_gray, int(val), 255, cv2.THRESH_BINARY)
            self.temp_image = Image.fromarray(result)
            self.display_temp_image()
        result = self.create_slider_dialog("Binary", "Threshold: 0-255", 0, 255, 127, 1, preview_binary)
        if result['confirmed'] and result['value'] is not None:
            img_gray = np.array(self.original_image.convert("L"))
            _, final_result = cv2.threshold(img_gray, int(result['value']), 255, cv2.THRESH_BINARY)
            self.processed_image = Image.fromarray(final_result)
        else:
            self.processed_image = self.original_image.copy()
        self.display_images()

    def color_grayscale(self):
        if not self.check_image_loaded(): return
        self.processed_image = self.original_image.convert("L")
        self.display_images()

    def color_rgb(self):
        if not self.check_image_loaded(): return
        self.processed_image = self.original_image.convert("RGB")
        self.display_images()

    def color_hsv(self):
        if not self.check_image_loaded(): return
        img_rgb = np.array(self.original_image.convert("RGB"))
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        self.processed_image = Image.fromarray(img_hsv)
        self.display_images()

    def color_cmy(self):
        if not self.check_image_loaded(): return
        img_rgb = np.array(self.original_image.convert("RGB"), dtype=np.float32) / 255.0
        img_cmy = 1.0 - img_rgb
        img_cmy = (img_cmy * 255).astype(np.uint8)
        self.processed_image = Image.fromarray(img_cmy)
        self.display_images()

    def color_yuv(self):
        if not self.check_image_loaded(): return
        img_rgb = np.array(self.original_image.convert("RGB"))
        img_yuv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
        self.processed_image = Image.fromarray(img_yuv)
        self.display_images()

    def color_yiq(self):
        if not self.check_image_loaded(): return
        img_rgb = np.array(self.original_image.convert("RGB"), dtype=np.float32) / 255.0
        transform_matrix = np.array([[0.299, 0.587, 0.114],
                                    [0.596, -0.275, -0.321],
                                    [0.212, -0.523, 0.311]])
        img_yiq = np.dot(img_rgb, transform_matrix.T)
        img_yiq = np.clip(img_yiq * 255, 0, 255).astype(np.uint8)
        self.processed_image = Image.fromarray(img_yiq)
        self.display_images()

    def color_pseudo(self):
        if not self.check_image_loaded(): return
        img_gray = np.array(self.original_image.convert("L"))
        img_colored = cv2.applyColorMap(img_gray, cv2.COLORMAP_JET)
        img_colored = cv2.cvtColor(img_colored, cv2.COLOR_BGR2RGB)
        self.processed_image = Image.fromarray(img_colored)
        self.display_images()

    # ========== ENHANCEMENT ==========
    def enhance_brightness(self):
        if not self.check_image_loaded(): return
        def preview_brightness(val):
            enhancer = ImageEnhance.Brightness(self.original_image)
            img = enhancer.enhance(val)
            self.temp_image = img
            self.display_temp_image()
        result = self.create_slider_dialog("Brightness", "Brightness: 0.1-3.0", 0.1, 3.0, 1.0, 0.1, preview_brightness)
        if result['confirmed'] and result['value'] is not None:
            enhancer = ImageEnhance.Brightness(self.original_image)
            self.processed_image = enhancer.enhance(result['value'])
        else:
            self.processed_image = self.original_image.copy()
        self.display_images()

    def enhance_contrast(self):
        if not self.check_image_loaded(): return
        def preview_contrast(val):
            enhancer = ImageEnhance.Contrast(self.original_image)
            img = enhancer.enhance(val)
            self.temp_image = img
            self.display_temp_image()
        result = self.create_slider_dialog("Contrast", "Contrast: 0.1-3.0", 0.1, 3.0, 1.0, 0.1, preview_contrast)
        if result['confirmed'] and result['value'] is not None:
            enhancer = ImageEnhance.Contrast(self.original_image)
            self.processed_image = enhancer.enhance(result['value'])
        else:
            self.processed_image = self.original_image.copy()
        self.display_images()

    def histogram_equalization(self):
        if not self.check_image_loaded(): return
        img_gray = np.array(self.original_image.convert("L"))
        eq = cv2.equalizeHist(img_gray)
        self.processed_image = Image.fromarray(eq)
        self.display_images()

    # ========== SMOOTHING ==========
    def smoothing_lowpass(self):
        if not self.check_image_loaded(): return
        def preview_lowpass(val):
            k = int(max(1, round(val)))
            img = np.array(self.original_image.convert("RGB"))
            result = cv2.blur(img, (k, k))
            self.temp_image = Image.fromarray(result)
            self.display_temp_image()
        result = self.create_slider_dialog("Lowpass", "Kernel size (odd): 1-31", 1, 31, 3, 1, preview_lowpass)
        if result['confirmed'] and result['value'] is not None:
            k = int(max(1, round(result['value'])))
            if k % 2 == 0: k += 1
            img = np.array(self.original_image.convert("RGB"))
            final = cv2.blur(img, (k, k))
            self.processed_image = Image.fromarray(final)
        else:
            self.processed_image = self.original_image.copy()
        self.display_images()

    def smoothing_median(self):
        if not self.check_image_loaded(): return
        def preview_median(val):
            k = int(max(1, round(val)))
            if k % 2 == 0: k += 1
            img = cv2.cvtColor(np.array(self.original_image), cv2.COLOR_RGB2BGR)
            result = cv2.medianBlur(img, k)
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            self.temp_image = Image.fromarray(result)
            self.display_temp_image()
        result = self.create_slider_dialog("Median", "Kernel size (odd): 1-31", 1, 31, 3, 1, preview_median)
        if result['confirmed'] and result['value'] is not None:
            k = int(max(1, round(result['value'])))
            if k % 2 == 0: k += 1
            img = cv2.cvtColor(np.array(self.original_image), cv2.COLOR_RGB2BGR)
            final = cv2.medianBlur(img, k)
            final = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
            self.processed_image = Image.fromarray(final)
        else:
            self.processed_image = self.original_image.copy()
        self.display_images()

    def smoothing_ilpf(self):
        if not self.check_image_loaded(): return
        def preview_ilpf(val):
            d0 = float(val)
            img = np.array(self.original_image.convert("L"), dtype=np.float32)
            rows, cols = img.shape
            d0 = max(1.0, d0)
            f = fft2(img)
            fshift = fftshift(f)
            crow, ccol = rows//2, cols//2
            mask = np.zeros_like(img)
            Y, X = np.ogrid[:rows, :cols]
            mask_area = (X - ccol)**2 + (Y - crow)**2 <= d0*d0
            mask[mask_area] = 1
            fshift_filtered = fshift * mask
            f_ishift = ifftshift(fshift_filtered)
            img_back = np.real(ifft2(f_ishift))
            img_back = np.clip(img_back, 0, 255).astype(np.uint8)
            self.temp_image = Image.fromarray(img_back)
            self.display_temp_image()
        result = self.create_slider_dialog("ILPF", "Cutoff radius: 1-200", 1, 200, 30, 1, preview_ilpf)
        if result['confirmed'] and result['value'] is not None:
            d0 = float(result['value'])
            img = np.array(self.original_image.convert("L"), dtype=np.float32)
            rows, cols = img.shape
            f = fft2(img); fshift = fftshift(f)
            crow, ccol = rows//2, cols//2
            mask = np.zeros_like(img)
            Y, X = np.ogrid[:rows, :cols]
            mask_area = (X - ccol)**2 + (Y - crow)**2 <= d0*d0
            mask[mask_area] = 1
            fshift_filtered = fshift * mask
            f_ishift = ifftshift(fshift_filtered)
            img_back = np.real(ifft2(f_ishift))
            img_back = np.clip(img_back, 0, 255).astype(np.uint8)
            self.processed_image = Image.fromarray(img_back)
        else:
            self.processed_image = self.original_image.copy()
        self.display_images()

    def smoothing_blpf(self):
        if not self.check_image_loaded(): return
        def preview_blpf(val):
            d0 = float(val)
            n = 2
            img = np.array(self.original_image.convert("L"), dtype=np.float32)
            rows, cols = img.shape
            f = fft2(img); fshift = fftshift(f)
            crow, ccol = rows//2, cols//2
            Y, X = np.ogrid[:rows, :cols]
            D = np.sqrt((X - ccol)**2 + (Y - crow)**2)
            H = 1 / (1 + (D / (d0+1e-6))**(2*n))
            fshift_filtered = fshift * H
            f_ishift = ifftshift(fshift_filtered)
            img_back = np.real(ifft2(f_ishift))
            img_back = np.clip(img_back, 0, 255).astype(np.uint8)
            self.temp_image = Image.fromarray(img_back)
            self.display_temp_image()
        result = self.create_slider_dialog("BLPF", "Cutoff radius: 1-200", 1, 200, 30, 1, preview_blpf)
        if result['confirmed'] and result['value'] is not None:
            d0 = float(result['value']); n = 2
            img = np.array(self.original_image.convert("L"), dtype=np.float32)
            rows, cols = img.shape
            f = fft2(img); fshift = fftshift(f)
            crow, ccol = rows//2, cols//2
            Y, X = np.ogrid[:rows, :cols]
            D = np.sqrt((X - ccol)**2 + (Y - crow)**2)
            H = 1 / (1 + (D / (d0+1e-6))**(2*n))
            fshift_filtered = fshift * H
            f_ishift = ifftshift(fshift_filtered)
            img_back = np.real(ifft2(f_ishift))
            img_back = np.clip(img_back, 0, 255).astype(np.uint8)
            self.processed_image = Image.fromarray(img_back)
        else:
            self.processed_image = self.original_image.copy()
        self.display_images()

    # ========== SHARPENING ==========
    def sharpening_highpass(self):
        if not self.check_image_loaded(): return
        img = np.array(self.original_image.convert("L"), dtype=np.float32)
        kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
        res = ndimage.convolve(img, kernel)
        res = np.clip(res, 0, 255).astype(np.uint8)
        self.processed_image = Image.fromarray(res)
        self.display_images()

    def sharpening_highboost(self):
        if not self.check_image_loaded(): return
        def preview_highboost(val):
            A = float(val)
            img = np.array(self.original_image.convert("L"), dtype=np.float32)
            blurred = cv2.GaussianBlur(img, (3,3), 0)
            mask = img - blurred
            res = img + (A - 1) * mask
            res = np.clip(res, 0, 255).astype(np.uint8)
            self.temp_image = Image.fromarray(res)
            self.display_temp_image()
        result = self.create_slider_dialog("Highboost", "Boost factor A: 1.0-3.0", 1.0, 3.0, 1.5, 0.1, preview_highboost)
        if result['confirmed'] and result['value'] is not None:
            A = float(result['value'])
            img = np.array(self.original_image.convert("L"), dtype=np.float32)
            blurred = cv2.GaussianBlur(img, (3,3), 0)
            mask = img - blurred
            res = img + (A - 1) * mask
            res = np.clip(res, 0, 255).astype(np.uint8)
            self.processed_image = Image.fromarray(res)
        else:
            self.processed_image = self.original_image.copy()
        self.display_images()

    def sharpening_ihpf(self):
        if not self.check_image_loaded(): return
        def preview_ihpf(val):
            d0 = float(val)
            img = np.array(self.original_image.convert("L"), dtype=np.float32)
            rows, cols = img.shape
            f = fft2(img); fshift = fftshift(f)
            crow, ccol = rows//2, cols//2
            mask = np.ones_like(img)
            Y, X = np.ogrid[:rows, :cols]
            mask_area = (X - ccol)**2 + (Y - crow)**2 <= d0*d0
            mask[mask_area] = 0
            fshift_filtered = fshift * mask
            f_ishift = ifftshift(fshift_filtered)
            img_back = np.real(ifft2(f_ishift))
            img_back = np.clip(img_back, 0, 255).astype(np.uint8)
            self.temp_image = Image.fromarray(img_back)
            self.display_temp_image()
        result = self.create_slider_dialog("IHPF", "Cutoff radius: 1-200", 1, 200, 30, 1, preview_ihpf)
        if result['confirmed'] and result['value'] is not None:
            d0 = float(result['value'])
            img = np.array(self.original_image.convert("L"), dtype=np.float32)
            rows, cols = img.shape
            f = fft2(img); fshift = fftshift(f)
            crow, ccol = rows//2, cols//2
            mask = np.ones_like(img)
            Y, X = np.ogrid[:rows, :cols]
            mask_area = (X - ccol)**2 + (Y - crow)**2 <= d0*d0
            mask[mask_area] = 0
            fshift_filtered = fshift * mask
            f_ishift = ifftshift(fshift_filtered)
            img_back = np.real(ifft2(f_ishift))
            img_back = np.clip(img_back, 0, 255).astype(np.uint8)
            self.processed_image = Image.fromarray(img_back)
        else:
            self.processed_image = self.original_image.copy()
        self.display_images()

    def sharpening_bhpf(self):
        if not self.check_image_loaded(): return
        def preview_bhpf(val):
            d0 = float(val)
            n = 2
            img = np.array(self.original_image.convert("L"), dtype=np.float32)
            rows, cols = img.shape
            f = fft2(img); fshift = fftshift(f)
            crow, ccol = rows//2, cols//2
            Y, X = np.ogrid[:rows, :cols]
            D = np.sqrt((X - ccol)**2 + (Y - crow)**2)
            H = 1 / (1 + (d0 / (D + 1e-6))**(2*n))
            fshift_filtered = fshift * H
            f_ishift = ifftshift(fshift_filtered)
            img_back = np.real(ifft2(f_ishift))
            img_back = np.clip(img_back, 0, 255).astype(np.uint8)
            self.temp_image = Image.fromarray(img_back)
            self.display_temp_image()
        result = self.create_slider_dialog("BHPF", "Cutoff radius: 1-200", 1, 200, 30, 1, preview_bhpf)
        if result['confirmed'] and result['value'] is not None:
            d0 = float(result['value']); n = 2
            img = np.array(self.original_image.convert("L"), dtype=np.float32)
            rows, cols = img.shape
            f = fft2(img); fshift = fftshift(f)
            crow, ccol = rows//2, cols//2
            Y, X = np.ogrid[:rows, :cols]
            D = np.sqrt((X - ccol)**2 + (Y - crow)**2)
            H = 1 / (1 + (d0 / (D + 1e-6))**(2*n))
            fshift_filtered = fshift * H
            f_ishift = ifftshift(fshift_filtered)
            img_back = np.real(ifft2(f_ishift))
            img_back = np.clip(img_back, 0, 255).astype(np.uint8)
            self.processed_image = Image.fromarray(img_back)
        else:
            self.processed_image = self.original_image.copy()
        self.display_images()

    def geometric_correction(self):
        if not self.check_image_loaded(): return
        img = np.array(self.original_image.convert("RGB")).astype(np.float32)
        p2, p98 = np.percentile(img, (2, 98))
        img_rescale = np.clip((img - p2) * 255.0 / (p98 - p2 + 1e-6), 0, 255).astype(np.uint8)
        self.processed_image = Image.fromarray(img_rescale)
        self.display_images()

    # ========== NOISE ==========
    def noise_gaussian(self):
        if not self.check_image_loaded(): return
        def preview_gauss(val):
            var = val
            img = np.array(self.original_image.convert("RGB"), dtype=np.float32)
            mean = 0
            sigma = math.sqrt(var)
            gauss = np.random.normal(mean, sigma, img.shape)
            noisy = img + gauss
            noisy = np.clip(noisy, 0, 255).astype(np.uint8)
            self.temp_image = Image.fromarray(noisy)
            self.display_temp_image()
        result = self.create_slider_dialog("Gaussian Noise", "Variance: 0-2000", 0, 2000, 200, 1, preview_gauss)
        if result['confirmed'] and result['value'] is not None:
            var = result['value']
            img = np.array(self.original_image.convert("RGB"), dtype=np.float32)
            mean = 0
            sigma = math.sqrt(var)
            gauss = np.random.normal(mean, sigma, img.shape)
            noisy = img + gauss
            noisy = np.clip(noisy, 0, 255).astype(np.uint8)
            self.processed_image = Image.fromarray(noisy)
        else:
            self.processed_image = self.original_image.copy()
        self.display_images()

    def noise_rayleigh(self):
        if not self.check_image_loaded(): return
        def preview_rayleigh(val):
            scale = val
            img = np.array(self.original_image.convert("RGB"), dtype=np.float32)
            r = np.random.rayleigh(scale, img.shape)
            noisy = img + r
            noisy = np.clip(noisy, 0, 255).astype(np.uint8)
            self.temp_image = Image.fromarray(noisy)
            self.display_temp_image()
        result = self.create_slider_dialog("Rayleigh Noise", "Scale: 0.1-100", 0.1, 100.0, 10.0, 0.1, preview_rayleigh)
        if result['confirmed'] and result['value'] is not None:
            scale = result['value']
            img = np.array(self.original_image.convert("RGB"), dtype=np.float32)
            r = np.random.rayleigh(scale, img.shape)
            noisy = img + r
            noisy = np.clip(noisy, 0, 255).astype(np.uint8)
            self.processed_image = Image.fromarray(noisy)
        else:
            self.processed_image = self.original_image.copy()
        self.display_images()

    def noise_erlang(self):
        if not self.check_image_loaded(): return
        def preview_erlang(val):
            shape = max(1, int(val))
            scale = 10.0
            img = np.array(self.original_image.convert("RGB"), dtype=np.float32)
            e = np.random.gamma(shape, scale, img.shape)
            noisy = img + e
            noisy = np.clip(noisy, 0, 255).astype(np.uint8)
            self.temp_image = Image.fromarray(noisy)
            self.display_temp_image()
        result = self.create_slider_dialog("Erlang Noise", "Shape (k): 1-10", 1, 10, 2, 1, preview_erlang)
        if result['confirmed'] and result['value'] is not None:
            shape = int(result['value'])
            scale = 10.0
            img = np.array(self.original_image.convert("RGB"), dtype=np.float32)
            e = np.random.gamma(shape, scale, img.shape)
            noisy = img + e
            noisy = np.clip(noisy, 0, 255).astype(np.uint8)
            self.processed_image = Image.fromarray(noisy)
        else:
            self.processed_image = self.original_image.copy()
        self.display_images()

    def noise_exponential(self):
        if not self.check_image_loaded(): return
        def preview_exponential(val):
            scale = val
            img = np.array(self.original_image.convert("RGB"), dtype=np.float32)
            e = np.random.exponential(scale, img.shape)
            noisy = img + e
            noisy = np.clip(noisy, 0, 255).astype(np.uint8)
            self.temp_image = Image.fromarray(noisy)
            self.display_temp_image()
        result = self.create_slider_dialog("Exponential Noise", "Scale: 0.1-50", 0.1, 50.0, 5.0, 0.1, preview_exponential)
        if result['confirmed'] and result['value'] is not None:
            scale = result['value']
            img = np.array(self.original_image.convert("RGB"), dtype=np.float32)
            e = np.random.exponential(scale, img.shape)
            noisy = img + e
            noisy = np.clip(noisy, 0, 255).astype(np.uint8)
            self.processed_image = Image.fromarray(noisy)
        else:
            self.processed_image = self.original_image.copy()
        self.display_images()

    def noise_uniform(self):
        if not self.check_image_loaded(): return
        def preview_uniform(val):
            a = -val; b = val
            img = np.array(self.original_image.convert("RGB"), dtype=np.float32)
            u = np.random.uniform(a, b, img.shape)
            noisy = img + u
            noisy = np.clip(noisy, 0, 255).astype(np.uint8)
            self.temp_image = Image.fromarray(noisy)
            self.display_temp_image()
        result = self.create_slider_dialog("Uniform Noise", "Range: 0-200", 0, 200, 20, 1, preview_uniform)
        if result['confirmed'] and result['value'] is not None:
            a = -result['value']; b = result['value']
            img = np.array(self.original_image.convert("RGB"), dtype=np.float32)
            u = np.random.uniform(a, b, img.shape)
            noisy = img + u
            noisy = np.clip(noisy, 0, 255).astype(np.uint8)
            self.processed_image = Image.fromarray(noisy)
        else:
            self.processed_image = self.original_image.copy()
        self.display_images()

    def noise_impulse(self):
        if not self.check_image_loaded(): return
        prob = simpledialog.askfloat("Impulse Noise", "Noise probability (0.0 - 1.0):", minvalue=0.0, maxvalue=1.0, initialvalue=0.05)
        if prob is None: return
        img = np.array(self.original_image.convert("RGB"), dtype=np.uint8)
        out = img.copy()
        rnd = np.random.rand(*img[:,:,0].shape)
        out[rnd < prob/2] = 0
        out[(rnd >= prob/2) & (rnd < prob)] = 255
        self.processed_image = Image.fromarray(out)
        self.display_images()

    # ========== EDGE DETECTION ==========
    def edge_sobel(self):
        if not self.check_image_loaded(): return
        img = np.array(self.original_image.convert("L"))
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.hypot(gx, gy)
        mag = np.clip(mag / mag.max() * 255, 0, 255).astype(np.uint8)
        self.processed_image = Image.fromarray(mag)
        self.display_images()

    def edge_prewitt(self):
        if not self.check_image_loaded(): return
        kernelx = np.array([[ -1,0,1],[-1,0,1],[-1,0,1]])
        kernely = np.array([[ 1,1,1],[0,0,0],[-1,-1,-1]])
        img = np.array(self.original_image.convert("L"), dtype=np.float32)
        gx = ndimage.convolve(img, kernelx)
        gy = ndimage.convolve(img, kernely)
        mag = np.hypot(gx, gy)
        mag = np.clip(mag / mag.max() * 255, 0, 255).astype(np.uint8)
        self.processed_image = Image.fromarray(mag)
        self.display_images()

    def edge_robert(self):
        if not self.check_image_loaded(): return
        img = np.array(self.original_image.convert("L"), dtype=np.float32)
        gx = ndimage.convolve(img, np.array([[1,0],[0,-1]]))
        gy = ndimage.convolve(img, np.array([[0,1],[-1,0]]))
        mag = np.hypot(gx, gy)
        mag = np.clip(mag / mag.max() * 255, 0, 255).astype(np.uint8)
        self.processed_image = Image.fromarray(mag)
        self.display_images()

    def edge_laplacian(self):
        if not self.check_image_loaded(): return
        img = np.array(self.original_image.convert("L"), dtype=np.float32)
        lap = cv2.Laplacian(img, cv2.CV_64F)
        lap = np.clip(np.abs(lap) / np.abs(lap).max() * 255, 0, 255).astype(np.uint8)
        self.processed_image = Image.fromarray(lap)
        self.display_images()

    def edge_log(self):
        if not self.check_image_loaded(): return
        img = np.array(self.original_image.convert("L"), dtype=np.float32)
        blurred = cv2.GaussianBlur(img, (5,5), 0)
        log = cv2.Laplacian(blurred, cv2.CV_64F)
        log = np.clip(np.abs(log) / np.abs(log).max() * 255, 0, 255).astype(np.uint8)
        self.processed_image = Image.fromarray(log)
        self.display_images()

    def edge_canny(self):
        if not self.check_image_loaded(): return
        img = np.array(self.original_image.convert("L"))
        edges = cv2.Canny(img, 100, 200)
        self.processed_image = Image.fromarray(edges)
        self.display_images()

    def edge_compass(self):
        if not self.check_image_loaded(): return
        img = np.array(self.original_image.convert("L"), dtype=np.float32)
        kernels = [
            np.array([[ -1,-1,2],[-1,-1,2],[-1,-1,2]]),
            np.array([[ -1,2,2],[-1,-1,2],[-1,-1,-1]]),
            np.array([[2,2,2],[-1,-1,-1],[-1,-1,-1]]),
            np.array([[2,2,-1],[2,-1,-1],[-1,-1,-1]])
        ]
        responses = [np.abs(ndimage.convolve(img, k)) for k in kernels]
        mag = np.maximum.reduce(responses)
        mag = np.clip(mag / mag.max() * 255, 0, 255).astype(np.uint8)
        self.processed_image = Image.fromarray(mag)
        self.display_images()

    # ========== SEGMENTATION ==========
    def segmentation_region_growing(self):
        if not self.check_image_loaded(): return
        x = simpledialog.askinteger("Seed X", "Enter seed X:", initialvalue=self.original_image.width//2)
        if x is None: return
        y = simpledialog.askinteger("Seed Y", "Enter seed Y:", initialvalue=self.original_image.height//2)
        if y is None: return
        tol = simpledialog.askinteger("Tolerance", "Intensity tolerance (0-255):", minvalue=0, maxvalue=255, initialvalue=10)
        if tol is None: return
        img = np.array(self.original_image.convert("L"))
        h, w = img.shape
        visited = np.zeros_like(img, dtype=bool)
        seed_val = int(img[y, x])
        stack = [(y,x)]
        mask = np.zeros_like(img, dtype=np.uint8)
        while stack:
            cy, cx = stack.pop()
            if cy<0 or cx<0 or cy>=h or cx>=w or visited[cy,cx]:
                continue
            visited[cy,cx] = True
            if abs(int(img[cy,cx]) - seed_val) <= tol:
                mask[cy,cx] = 255
                neighbors = [(cy+1,cx),(cy-1,cx),(cy,cx+1),(cy,cx-1)]
                stack.extend(neighbors)
        self.processed_image = Image.fromarray(mask)
        self.display_images()

    def segmentation_watershed(self):
        if not self.check_image_loaded(): return
        img = cv2.cvtColor(np.array(self.original_image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2,5)
        ret2, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        ret3, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown==255] = 0
        markers = cv2.watershed(img, markers)
        img[markers == -1] = [255,0,0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.processed_image = Image.fromarray(img)
        self.display_images()

    # ========== ABOUT / HELP ==========
    def show_info(self):
        info_text = f"""
JPEGirls Deluxe Pro
© Tim JPEGirls (2025)

Anggota Tim:
1. Sofi Kumala Dina
2. Dini Saputri Letari

Mata Kuliah: Pengolahan Citra Digital
Dosen Pengampu: Feri Candra, S.T., M.T., Ph.D
Nanda Dwi Putra S.Kom., M.Kom
Universitas: Universitas Riau

Fitur (preserved):
- Basic Operations (Arithmetic, Boolean, Geometric)
- Image Enhancement (Brightness, Contrast, Filtering)
- Noise Addition and Removal
- Edge Detection (Sobel, Prewitt, Canny, etc.)
- Image Segmentation (Region Growing, Watershed)
- Frequency Domain Processing (FFT, Filters)
- Color Space Conversions
"""
        messagebox.showinfo("About - JPEGirls Deluxe Pro", info_text)

    def open_github(self):
        webbrowser.open("https://github.com")
        messagebox.showinfo("Tutorial", "Opening Github tutorial...")

    def open_youtube(self):
        webbrowser.open("https://youtube.com")
        messagebox.showinfo("Tutorial", "Opening Youtube tutorial...")

# ========== MAIN ==========
if __name__ == "__main__":
    root = tk.Tk()
    app = JPEGirlsDeluxePro_UI(root)
    root.mainloop()
