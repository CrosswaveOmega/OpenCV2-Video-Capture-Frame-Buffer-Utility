## This module is used for selecting camera indexes and defining dialog gui windows.

import tkinter as tk
from tkinter import ttk
from tkinter.simpledialog import Dialog
import cv2
import subprocess
import sys
import re


from tkinter import messagebox

# Functions related to listing cameras for all platforms


def detect_platform():
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform.startswith("darwin"):
        return "mac"
    return "unknown"


def list_cameras_windows():
    try:
        proc = subprocess.Popen(
            ["ffmpeg", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        output = proc.stderr.read()
        matches = re.findall(r'\[dshow .*?\] *"([^"]+)"', output)
        return [(i, name) for i, name in enumerate(matches)]
    except Exception:
        return []


def list_cameras_linux():
    try:
        proc = subprocess.Popen(
            ["v4l2-ctl", "--list-devices"], stdout=subprocess.PIPE, text=True
        )
        out = proc.stdout.read()
        devices = []
        current_name = None
        for line in out.splitlines():
            if not line.startswith("\t") and line.strip():
                current_name = line.strip()
            if "\t/dev/video" in line:
                dev = line.strip()
                idx = int(dev.replace("/dev/video", ""))
                devices.append((idx, current_name))
        return devices
    except Exception:
        return []


def list_cameras_mac():
    try:
        proc = subprocess.Popen(
            ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        output = proc.stderr.read()
        matches = re.findall(r"\[AVFoundation .*?\] \[([0-9]+)\] ([^\r\n]+)", output)
        return [(int(i), name) for i, name in matches]
    except Exception:
        return []


def fallback_opencv(max_index=10):
    devices = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            devices.append((i, f"Camera {i}"))
            cap.release()
    return devices


def list_available_cameras():
    platform = detect_platform()
    if platform == "windows":
        cams = list_cameras_windows()
        return cams if cams else fallback_opencv()
    if platform == "linux":
        cams = list_cameras_linux()
        return cams if cams else fallback_opencv()
    if platform == "mac":
        cams = list_cameras_mac()
        return cams if cams else fallback_opencv()
    return fallback_opencv()


def build_resolution_list():
    """Create a list of resolutions"""
    base_resolutions = [
        (1280, 720, "720p"),
        (1920, 1080, "1080p"),
        (2560, 1440, "1440p"),
        (3840, 2160, "4K"),
    ]

    refresh_rates = [60, 120]

    resolutions = []
    for w, h, label in base_resolutions:
        for hz in refresh_rates:
            resolutions.append(f"{w}x{h} {label} @ {hz}Hz")

    return resolutions


def open_camera_popup(root):
    """Popup that returns selected camera index"""
    popup = tk.Toplevel(root)
    popup.title("Select Camera")
    popup.geometry("350x350")

    result_var = tk.IntVar(value=-1)
    resolution_var = tk.StringVar(value="1920x1080 1080p @ 60Hz")

    ttk.Label(popup, text="Choose a video capture device:").pack(pady=5)

    devices = list_available_cameras()

    if not devices:
        ttk.Label(popup, text="No cameras detected").pack(pady=10)
        ttk.Button(popup, text="Close", command=popup.destroy).pack(pady=5)
        popup.wait_window()
        return -1, None

    labels = [f"{idx}: {name}" for idx, name in devices]
    selected = tk.StringVar(value=labels[0])

    dropdown = ttk.Combobox(
        popup, textvariable=selected, values=labels, state="readonly", width=40
    )
    dropdown.pack(pady=5)

    resolution_values = build_resolution_list()

    ttk.Label(popup, text="Choose a resolution:").pack(pady=5)

    resolution_dropdown = ttk.Combobox(
        popup,
        textvariable=resolution_var,
        values=resolution_values,
        state="readonly",
        width=40,
    )
    resolution_dropdown.pack(pady=5)

    def confirm():
        chosen_label = selected.get()
        idx = int(chosen_label.split(":")[0])
        result_var.set(idx)
        popup.destroy()

    ttk.Button(popup, text="OK", command=confirm).pack(pady=5)

    popup.wait_window()

    return result_var.get(), resolution_var.get()


# Simple text dialog, for displaying a help menu with h.
class TextDialog(Dialog):
    def __init__(
        self,
        parent,
        title,
        text,
        width=80,
        height=20,
        ok_text="OK",
    ):
        self.text = text
        self.width = width
        self.height = height
        self.ok_text = ok_text
        super().__init__(parent, title)

    def body(self, master):
        frame = tk.Frame(master)
        frame.pack(expand=1, fill=tk.BOTH)

        scrollbar = tk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        text = tk.Text(
            frame,
            wrap=tk.WORD,
            width=self.width,
            height=self.height,
            yscrollcommand=scrollbar.set,
        )
        text.insert("1.0", self.text)
        text.configure(state=tk.DISABLED)
        text.pack(side=tk.LEFT, expand=1, fill=tk.BOTH)

        scrollbar.config(command=text.yview)

        return text

    def buttonbox(self):
        box = tk.Frame(self)

        w = tk.Button(
            box,
            text=self.ok_text,
            width=10,
            command=self.ok,
            default=tk.ACTIVE,
        )
        w.pack(padx=5, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

        box.pack()

    def validate(self):
        return 1

    def apply(self):
        pass


# Resize dialog.
class ResizeThresholdDialog(Dialog):
    """
    Dialog for changing the rows/columns to be displayed, along with the
    frame disposal integer"""

    size_errormessage = "columns and rows must be integers."
    threshold_errormessage = "Value must be a number."

    def __init__(
        self,
        title,
        prompt,
        initial_columns=None,
        initial_rows=None,
        minvalue=None,
        maxvalue=None,
        initial_threshold=0.0,
        threshold_max=1.0,
        threshold_resolution=0.01,
        parent=None,
    ):
        self.prompt = prompt

        self.initial_columns = initial_columns
        self.initial_rows = initial_rows
        self.minvalue = minvalue
        self.maxvalue = maxvalue

        self.initial_threshold = initial_threshold
        self.threshold_max = threshold_max
        self.threshold_resolution = threshold_resolution

        self.result = None

        super().__init__(parent, title)

    def body(self, master):
        # Inital prompt
        lbl = tk.Label(master, text=self.prompt, justify=tk.LEFT)
        lbl.grid(row=0, column=0, columnspan=2, padx=5, pady=(5, 8), sticky="w")

        # columns / rows
        tk.Label(master, text="Columns:").grid(row=1, column=0, padx=5, sticky="w")
        tk.Label(master, text="Rows:").grid(row=2, column=0, padx=5, sticky="w")

        self.columns_entry = tk.Entry(master, name="columns")
        self.rows_entry = tk.Entry(master, name="rows")

        self.columns_entry.grid(row=1, column=1, padx=5, sticky="we")
        self.rows_entry.grid(row=2, column=1, padx=5, sticky="we")

        if self.initial_columns is not None:
            self.columns_entry.insert(0, self.initial_columns)
            self.columns_entry.select_range(0, tk.END)

        if self.initial_rows is not None:
            self.rows_entry.insert(0, self.initial_rows)
            self.rows_entry.select_range(0, tk.END)

        # Threshold
        tk.Label(master, text="Threshold:").grid(
            row=3, column=0, columnspan=2, padx=5, pady=(10, 0), sticky="w"
        )

        self.scale = tk.Scale(
            master,
            from_=0.0,
            to=self.threshold_max,
            orient=tk.HORIZONTAL,
            resolution=self.threshold_resolution,
            length=300,
        )
        self.scale.grid(row=4, column=0, columnspan=2, padx=5, sticky="we")
        self.scale.set(self.initial_threshold)

        master.columnconfigure(1, weight=1)
        return self.columns_entry

    def validate(self):
        # Validate columns / rows
        try:
            columns = int(self.columns_entry.get())
            rows = int(self.rows_entry.get())
        except ValueError:
            messagebox.showwarning(
                "Illegal value",
                self.size_errormessage + "\nPlease try again",
                parent=self,
            )
            return 0

        if self.minvalue is not None:
            if columns < self.minvalue or rows < self.minvalue:
                messagebox.showwarning(
                    "Too small",
                    f"The allowed minimum value is {self.minvalue}.",
                    parent=self,
                )
                return 0

        if self.maxvalue is not None:
            if columns > self.maxvalue or rows > self.maxvalue:
                messagebox.showwarning(
                    "Too large",
                    f"The allowed maximum value is {self.maxvalue}.",
                    parent=self,
                )
                return 0

        # Validate threshold
        try:
            threshold = float(self.scale.get())
        except Exception:
            messagebox.showwarning(
                "Illegal value",
                self.threshold_errormessage + "\nPlease try again",
                parent=self,
            )
            return 0

        if not (0.0 <= threshold <= self.threshold_max):
            messagebox.showwarning(
                "Out of range",
                f"Value must be between 0 and {self.threshold_max}.",
                parent=self,
            )
            return 0

        self.result = (columns, rows, threshold)
        return 1

    def apply(self):
        pass


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    TextDialog(
        root,
        title="Information",
        text=(
            "This is a static dialog based on Dialog.\n\n"
            "It supports multiline text, scrolling, and modal behavior.\n"
            "It closes when OK is pressed."
        ),
    )

    # device_index = open_camera_popup()
    # print("User selected:", device_index)
