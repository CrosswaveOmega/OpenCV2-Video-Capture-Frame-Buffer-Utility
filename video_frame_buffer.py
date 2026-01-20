import re
import cv2
from collections import deque
import queue
import math
import numpy as np
import os
import datetime
import threading
from typing import Tuple, Optional, List, Deque, Any

import configparser
from pathlib import Path
import tkinter as tk
from tkinter import simpledialog
import dialog_boxes


WINDOWNAME = "Last N Frame Difference Buffer- press H for help!"
HELP_TEXT = """

This is my custom frame buffer program!  It takes in a video feed from a capture card, and lets you select screenshots from that video feed.

Click on an image to save it.

Press N to set the folder to save the screenshots to.

Press R to realign the outputs to a new window.

Press M to change the number of rows/columns displayed in the preview, or the frame disposal threshold.

Press W to go up one buffer row.

Press S to go down one buffer row.


Press ESC to exit the program.



"""

def compute_frame_difference(
    prev_frame: Optional[np.ndarray],
    curr_frame: np.ndarray,
    pixel_change_min: int = 0,
    pixel_change_max: int = 256,
) -> Tuple[np.ndarray, int, float]:
    """
    Compute the difference between two frames via converting it to greyscale, and quantify the changes.
    Parameters:
        prev_frame (Optional[np.ndarray]): The previous frame to compare against.
                                        If None, returns a zeroed difference.
        curr_frame (np.ndarray): The current frame to compare.
        pixel_change_min (int): The minimum change in the greyscale image to consider
                                a pixel as changed. Default is 0.
        pixel_change_max (int): The maximum change in the greyscale image to consider
                                a pixel as changed. Default is 256.
    Returns:
        Tuple[np.ndarray, int, float]: A tuple containing:
            - The difference image as a numpy array.
            - The count of changed pixels.
            - The fraction of changed pixels relative to the total number of pixels.
    """

    if prev_frame is None:
        diff = curr_frame * 0
        changed_pixels = 0
        changed_frac = 0.0
    else:
        diff = cv2.absdiff(curr_frame, prev_frame)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        # Ignore tiny changes (noise)
        _, gray_thresh = cv2.threshold(
            gray, pixel_change_min, pixel_change_max, cv2.THRESH_TOZERO
        )

        changed_pixels = np.count_nonzero(gray_thresh)
        total_pixels = gray_thresh.size
        changed_frac = changed_pixels / total_pixels

        diff = gray_thresh

    return diff, changed_pixels, changed_frac


def highlight_diff(
    img_display: np.ndarray,
    img_orig: np.ndarray,
    color: Tuple[int, int, int] = (0, 0, 255),
    intensity: float = 0.7,
) -> np.ndarray:
    """Highlight the difference in the display window."""
    if len(img_display.shape) == 3:
        img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2GRAY)

    alpha = img_display.astype(np.float32) / 255.0

    alpha[alpha == 0] = 0.0

    alpha[(alpha != 0) & (alpha < 0.1)] = 0.1
    alpha = alpha * intensity
    alpha = np.expand_dims(alpha, axis=2)

    overlay = np.full_like(img_orig, color, dtype=np.uint8)

    return cv2.convertScaleAbs((1 - alpha) * img_orig + alpha * overlay)

class FrameViewerConfigMixin:
    """Mixin functions for frameviewer."""

    CONFIG_SECTION = "FrameViewer"

    def load_config(self, path: str = "config.ini") -> None:
        config = configparser.ConfigParser()
        path = Path(path)

        if not path.exists():
            return

        config.read(path)

        if self.CONFIG_SECTION not in config:
            return

        cfg = config[self.CONFIG_SECTION]

        if "name" in cfg:
            self._apply_new_name(cfg["name"])

        if "file_prefix" in cfg:
            self.file_prefix = cfg["file_prefix"]
            self.prefix = f"./{self.name}/{self.file_prefix}{self.name}"
            self._apply_new_name(cfg["name"])

        if "grid_rows" in cfg:
            self.grid_rows = int(cfg["grid_rows"])

        if "grid_cols" in cfg:
            self.grid_cols = int(cfg["grid_cols"])

        if "history_scrolling_limit" in cfg:
            self.history_scrolling_limit = int(cfg["history_scrolling_limit"])

        if "change_threshold" in cfg:
            self.change_threshold = float(cfg["change_threshold"])

        if "pixel_change_min" in cfg:
            self.pixel_change_min = int(cfg["pixel_change_min"])

        if "pixel_change_max" in cfg:
            self.pixel_change_max = int(cfg["pixel_change_max"])

    def save_config(self, path: str = "config.ini") -> None:
        config = configparser.ConfigParser()
        path = Path(path)

        if path.exists():
            config.read(path)

        if self.CONFIG_SECTION not in config:
            config[self.CONFIG_SECTION] = {}

        cfg = config[self.CONFIG_SECTION]

        cfg["name"] = self.name
        cfg["file_prefix"] = self.file_prefix
        cfg["grid_rows"] = str(self.grid_rows)
        cfg["grid_cols"] = str(self.grid_cols)
        cfg["change_threshold"] = str(self.change_threshold)
        cfg["pixel_change_min"] = str(self.pixel_change_min)
        cfg["pixel_change_max"] = str(self.pixel_change_max)
        cfg["history_scrolling_limit"] = str(self.history_scrolling_limit)

        with path.open("w") as f:
            config.write(f)

def command(name: str):
    def decorator(func):
        func._command_name = name
        return func
    return decorator

class FrameViewer(FrameViewerConfigMixin):
    """Frame Viewer Class.  This class displays the frame preview"""

    _command_handlers = {}

    
    def __init__(
        self,
        name: str = "FrameBufferOutput",
        grid_rows: int = 10,
        grid_cols: int = 8,
        history_scrolling_limit: int = 10,
        change_threshold: float = 0.1,
    ) -> None:
        # Folder name to save images to.
        self.name: str = ""
        self.prefix: str = ""
        self.file_prefix: str = "LiveScreenshot-"
        self._apply_new_name(name)
        os.makedirs(f"./video_frame_buffer_output/{name}/", exist_ok=True)

        self.name_request_q: Optional[queue.Queue] = None
        self.name_result_q: Optional[queue.Queue] = None

        self.scrollbar_width: int = 20
        self.bar_color: Tuple[int, int, int] = (0, 255, 0)
        self.alpha: float = 0.6

        self.frame_rects: List[Tuple[int, int, int, int, np.ndarray]] = []
        self.saved: int = 0
        self.yg_offset: int = 0

        # Where the mouse is hovering over.
        self.mouse_hovering_over_index = 0

        self.grid_rows: int = grid_rows
        self.grid_cols: int = grid_cols
        self.history_scrolling_limit: int = history_scrolling_limit
        self.change_threshold: float = change_threshold

        # frame_diff, frame_orig
        self.diffs: Deque[Tuple[np.ndarray, np.ndarray]] = deque(
            maxlen=(history_scrolling_limit + grid_rows) * grid_cols
        )
        self.last_frame: Optional[np.ndarray] = None
        self.last_frame_full: Optional[np.ndarray] = None

        self.target_size: Optional[Tuple[int, int]] = None
        self.pause_delay: bool = True
        self.pixel_change_min = 0
        self.pixel_change_max = 256
        self.last_frame_available: bool = False

        self.display_w: int = 0
        self.display_h: int = 0

        self.flag_resize = False
        self.KEYMAP = {
            ord("p"): "TOGGLE_PAUSE",
            ord("n"): "CHANGE_TARGET_FOLDER",
            ord("h"): "SHOW_HELP",
            ord("m"): "SHOW_RESIZE",
            ord("r"): "REORIENT",
            ord("w"): "SCROLL_UP",
            ord("s"): "SCROLL_DOWN",
            27:       "QUIT",  # ESC
        }
        handlers = {}
        for attr in self.__class__.__dict__.values():
            name = getattr(attr, "_command_name", None)
            if name:
                handlers[name] = attr
        self.__class__._command_handlers = handlers



    
    def _apply_new_name(self, target_folder: str) -> None:
        """Change the folder and prefix"""
        self.name = target_folder
        self.prefix = f"./video_frame_buffer_output/{target_folder}/{self.file_prefix}{target_folder}"
        print(f"File Prefix set to {self.prefix}")
        os.makedirs(f"./video_frame_buffer_output/{target_folder}/", exist_ok=True)

    def _resize(self, grid_cols=0, grid_rows=0) -> None:
        """Change the size of the displayed grid."""

        if grid_cols == self.grid_cols and grid_rows == self.grid_rows:
            return
        print("Changing sizes")
        print(grid_cols, grid_rows)
        self.yg_offset = 0
        ok = False
        if self.grid_cols > 0:
            self.grid_cols = grid_cols or self.grid_cols
            ok = True
        if self.grid_rows > 0:
            self.grid_rows = grid_rows or self.grid_rows
            ok = True
        if ok:
            self.flag_resize = True

    def mouse_callback(
        self, event: int, x: int, y: int, flags: int, param: Any
    ) -> None:
        """
        Mouse callback function that handles mouse events for frame selection.
        Parameters:
            event (int): The type of mouse event, this is a opencv MouseEventType Enum.
            x (int): The x-coordinate of the mouse event.
            y (int): The y-coordinate of the mouse event.
            flags (int): Additional flags associated with the mouse event.
            param (Any): Additional parameters passed to the callback.
        Functionality:
        - Updates the index of the frame currently being hovered over based on the mouse position.
        - Handles mouse click events to queue a save request for the selected frame.
        - Adjusts the vertical offset for frame display based on mouse scroll events.
        """

        # Set the mouse hovering over index: this is the frame that's surrounded
        # by a thick red border.
        for idx, (fx, fy_base, fw, fh, frame_to_save) in enumerate(self.frame_rects):
            fy = fy_base - self.yg_offset * fh
            if fx <= x < fx + fw and fy <= y < fy + fh:
                # print("mouse is hovering over:", idx)
                self.mouse_hovering_over_index = idx
                break

        if int(event) == cv2.EVENT_LBUTTONUP:  # click
            for idx, (fx, fy_base, fw, fh, frame_to_save) in enumerate(
                self.frame_rects
            ):
                fy = fy_base - self.yg_offset * fh
                if fx <= x < fx + fw and fy <= y < fy + fh:
                    self.name_request_q.put(
                        (
                            "ASKSAVE",
                            (frame_to_save.copy(), self.prefix, self.saved, idx),
                        )
                    )
                    self.saved += 1

        if int(event) == cv2.EVENT_MOUSEWHEEL:  # scroll up or down.
            delta_scroll = flags >> 16
            if delta_scroll > 0:
                self.yg_offset = max(0, self.yg_offset - 1)
            else:
                self.yg_offset += 1


    @command("CHANGE_TARGET_FOLDER")
    def _change_target_folder(self, frame, stop_event):
        if self.name_request_q:
            self.name_request_q.put(("ASK_FOLDER_NAME", None))
            # block until the main thread provides the name
            target_folder = self.name_result_q.get()
            if target_folder:
                self._apply_new_name(target_folder)

    @command("SHOW_HELP")
    def _show_help(self, frame, stop_event):
        if self.name_request_q:
            self.name_request_q.put(("HELP", None))


    @command("REORIENT")
    def _reorient_frames(self, frame, stop_event):
        h, w = frame.shape[:2]
        self.__make_target_size(h, w)
        self.last_frame = cv2.resize(frame.copy(), self.target_size)

        #frame_small = cv2.resize(frame, self.target_size)


    @command("SHOW_RESIZE_DIALOG")
    def _resize_dialog(self, frame, stop_event):
        if self.name_request_q:
            self.name_request_q.put(("RESIZE", None))
    @command("SCROLL_UP")
    def _cmd_scroll_up(self, frame, stop_event):
        self.yg_offset = max(0, self.yg_offset - 1)

    @command("SCROLL_DOWN")
    def _cmd_scroll_down(self, frame, stop_event):
        self.yg_offset = max(0, self.yg_offset - 1)

    def keyboard_callback(self, key, frame, stop_event):
        command = self.KEYMAP.get(key)

        if command:
            handler = self._command_handlers.get(command)
            if handler:
                handler(self, frame, stop_event)
                return 1

            if command == "QUIT":
                stop_event.set()
                return -1
        return 0


    def __make_target_size(self, h, w):
        """Redo the target size calculation that scales each frame in the preview window."""
        print("making target size", h, w)
        self.display_w = cv2.getWindowImageRect(WINDOWNAME)[2]
        self.display_h = cv2.getWindowImageRect(WINDOWNAME)[3]

        available_height = (self.display_h or 1080) - 80
        cell_h = available_height // (self.grid_rows)
        print(available_height, self.display_h)
        aspect = w / h
        cell_w = int(cell_h * aspect)

        self.target_size = (w // self.grid_cols, cell_h)
        print(self.target_size, available_height, self.display_h)

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        """Normalize image."""
        target_w, target_h = self.target_size
        img = img.copy()
        # Rescale.
        if img.shape[1] != target_w or img.shape[0] != target_h:
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        # normalize dimensions.
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # Change dtype if needed
        if img.dtype != np.uint8:
            print("changing dtype")
            img = img.astype(np.uint8)

        # Check if the mouse is hovering over THIS square.
        if self.mouse_hovering_over_index == self.current_normalized_id:
            h, w = img.shape[:2]
            pad = 2

            cv2.rectangle(
                img,
                (pad, pad),
                (w - pad - 1, h - pad - 1),
                (0, 255, 0),  # green (BGR)
                thickness=2,
                lineType=cv2.LINE_AA,
            )
        self.current_normalized_id += 1

        return img

    def run(self, q: queue.Queue, stop_event: threading.Event) -> None:
        try:
            self.run_main(q, stop_event)
        except Exception as e:
            print(e)
            stop_event.set()

    def run_main(self, q: queue.Queue, stop_event: threading.Event) -> None:
        cv2.namedWindow(WINDOWNAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOWNAME, self.mouse_callback)

        while not stop_event.is_set():
            try:
                ts, frame = q.get(timeout=0.1)
            except queue.Empty:
                continue
            # set target_frame size.
            self.display_w = cv2.getWindowImageRect(WINDOWNAME)[2]
            self.display_h = cv2.getWindowImageRect(WINDOWNAME)[3]
            if self.target_size is None:
                h, w = frame.shape[:2]
                self.__make_target_size(h, w)
            if self.flag_resize:
                self.flag_resize = False
                self.recalculate_all()
            frame_small = cv2.resize(frame, self.target_size)

            key = cv2.waitKey(1) & 0xFF
            value=self.keyboard_callback(key,frame,stop_event)
            if value==-1:
                break
            '''
            if key == ord("p"):
                self.pause_delay = not self.pause_delay
            elif key == ord("n"):
                # Change the target folder
                if self.name_request_q:
                    self.name_request_q.put(("ASK_FOLDER_NAME", None))
                    # block until the main thread provides the name
                    target_folder = self.name_result_q.get()
                    if target_folder:
                        self._apply_new_name(target_folder)
            elif key == ord("h"):
                #SHOW HELP
                if self.name_request_q:
                    self.name_request_q.put(("HELP", None))
            elif key == ord("m"):
                # SHOW RESIZE DIALOG
                if self.name_request_q:
                    self.name_request_q.put(("RESIZE", None))
            elif key == ord("r"):
                # Reorient the displayed frames.
                h, w = frame.shape[:2]
                self.__make_target_size(h, w)
                self.last_frame = cv2.resize(frame.copy(), self.target_size)

                frame_small = cv2.resize(frame, self.target_size)
            elif key == ord("w"):  # SCROLL UP ONE
                self.yg_offset = max(0, self.yg_offset - 1)
            elif key == ord("s"):  # SCROLL DOWN ONE
                self.yg_offset += 1
            elif key == 27:  # ESCAPE.
                stop_event.set()
                break
            '''
            # Check if the window is closed/open
            if cv2.getWindowProperty(WINDOWNAME, cv2.WND_PROP_VISIBLE) < 1:
                print("Window closed by user")
                stop_event.set()
                break
            if self.last_frame_available:
                self._process_frame(frame_small, frame, ts)

            self.last_frame = frame_small.copy()
            self.last_frame_available = True

        cv2.destroyAllWindows()

    def recalculate_all(self):
        """Resize all currently available diffs, recalculate the target size, and change the max length of self.diffs."""

        print("RECALCULATING!")

        _, frame = self.diffs[-1]
        h, w = frame.shape[:2]
        self.__make_target_size(h, w)

        self.last_frame = cv2.resize(frame, self.target_size)
        diffs = []
        for diff_bgr, diff in self.diffs:
            diffs.append((cv2.resize(diff, self.target_size), diff.copy()))
        self.diffs: Deque[Tuple[np.ndarray, np.ndarray]] = deque(
            maxlen=(self.history_scrolling_limit + self.grid_rows) * self.grid_cols
        )
        for d in diffs:
            self.diffs.append(d)

    def _process_frame(
        self, frame_small: np.ndarray, full_frame: np.ndarray, ts: datetime.datetime
    ) -> None:
        """Compute the difference between frames."""

        # Frame difference is the change in pixels between frames.
        diff_gray, changed_pixels, changed_frac = compute_frame_difference(
            self.last_frame, frame_small, self.pixel_change_min, self.pixel_change_max
        )

        if changed_frac > self.change_threshold:
            # only add to diffs if needed.
            diff_bgr = highlight_diff(
                cv2.cvtColor(diff_gray, cv2.COLOR_GRAY2BGR), frame_small
            )
            # Output text

            timestamp = ts.strftime("%H:%M:%S:%f")[:-3]
            text = f"{timestamp} - {changed_frac:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.5
            thickness = 1
            text_size = cv2.getTextSize(text, font, scale, thickness)[0]
            x_text = (diff_bgr.shape[1] - text_size[0]) // 2
            y_text = (diff_bgr.shape[0] + text_size[1]) // 2
            # Outline text
            outline_thickness = 2
            for dx, dy in [
                (-outline_thickness, 0),
                (outline_thickness, 0),
                (0, -outline_thickness),
                (0, outline_thickness),
            ]:
                cv2.putText(
                    diff_bgr,
                    str(text),
                    (x_text + dx, y_text + dy),
                    font,
                    scale,
                    (0, 0, 255),
                    thickness,
                    cv2.LINE_AA,
                )

            # Main text
            cv2.putText(
                diff_bgr,
                str(text),
                (x_text, y_text),
                font,
                scale,
                (255, 255, 255),  # White color for the main text
                thickness,
                cv2.LINE_AA,
            )

            self.diffs.append((diff_bgr, full_frame.copy()))

        if self.diffs:
            self._render_grid()

    def _render_grid(self) -> None:
        # Clear the frame_rects, this is a list of all
        # Frames to be passed into the mouse callback.
        self.frame_rects = []

        rows = math.ceil(len(self.diffs) / self.grid_cols)

        # All rows
        grid_rows: List[np.ndarray] = []

        diffs_list = list(self.diffs)

        self.current_normalized_id = 0
        # Create frame rects.
        for r in range(rows):
            start = r * self.grid_cols
            end = min(start + self.grid_cols, len(diffs_list))

            row_tuples = diffs_list[start:end]

            row_imgs = [self._normalize(t[0]) for t in row_tuples]

            if len(row_imgs) < self.grid_cols:
                h_frame, w_frame = row_imgs[0].shape[:2]
                for _ in range(self.grid_cols - len(row_imgs)):
                    row_imgs.append(np.zeros((h_frame, w_frame, 3), dtype=np.uint8))
                    row_tuples.append((row_imgs[-1], np.zeros_like(row_tuples[0][1])))

            # Create rows
            row_concat = cv2.hconcat(row_imgs)
            y_offset = r * row_imgs[0].shape[0]

            for c, (img_display, img_orig) in enumerate(row_tuples):
                x_offset = c * img_display.shape[1]
                # Start X, Start Y, Frame Size X, Frame Size Y
                self.frame_rects.append(
                    (
                        x_offset,
                        y_offset,
                        img_display.shape[1],
                        img_display.shape[0],
                        img_orig,
                    )
                )

            grid_rows.append(row_concat)

        # Ensure scroll bar never exceeds the bottom of the visible area.
        if self.yg_offset >= max(0, len(grid_rows) - self.grid_rows):
            self.yg_offset = max(0, len(grid_rows) - self.grid_rows)

        visible = grid_rows[self.yg_offset : self.yg_offset + self.grid_rows]

        if not visible:
            print("NOT VISIBLE.")
            return

        combined_grid = cv2.vconcat(visible)

        target_h = 1020
        h, w, c = combined_grid.shape

        if h < target_h:
            pad_h = target_h - h
            combined_grid = cv2.copyMakeBorder(
                combined_grid, 0, pad_h, 0, 0, cv2.BORDER_CONSTANT, value=0
            )

        # Create scroll bar.
        total_rows = len(grid_rows)
        view_rows = self.grid_rows
        indicator_h = max(self.grid_rows, int(h * view_rows / total_rows))
        indicator_y = int(h * self.yg_offset / total_rows)
        indicator_y = min(h - indicator_h, indicator_y)

        overlay = combined_grid.copy()
        cv2.rectangle(
            overlay,
            (w - self.scrollbar_width, indicator_y),
            (w - 1, indicator_y + indicator_h),
            self.bar_color,
            -1,
        )

        combined_grid_with_bar = cv2.addWeighted(
            overlay, self.alpha, combined_grid, 1 - self.alpha, 0
        )

        # Show rendered grid image on WINDOWNAME.
        cv2.imshow(WINDOWNAME, combined_grid_with_bar)


class FrameCapture:
    '''Class for capturing frames from a video feed.'''
    def __init__(self, cam_index: int = 0, resolution="1920x1080 1080p @ 60Hz") -> None:
        self.cap: cv2.VideoCapture = cv2.VideoCapture(cam_index)
        w, h, fr = self.parse_resolution(resolution)
        self.x = w
        self.y = h

    def parse_resolution(self, resolution: str) -> Tuple[int, int, int]:
        """ """
        pattern = r"(?P<w>\d+)\s*x\s*(?P<h>\d+).*?@\s*(?P<hz>\d+)\s*Hz"
        match = re.search(pattern, resolution)

        if not match:
            raise ValueError(f"Invalid resolution format: {resolution}")

        width = int(match.group("w"))
        height = int(match.group("h"))
        framerate = int(match.group("hz"))

        return width, height, framerate

    def run(self, q: queue.Queue, stop_event: threading.Event) -> None:
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.x)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.y)
        while not stop_event.is_set():
            try:
                ret, frame = self.cap.read()
                if not ret:
                    print("Frame capture failed.")
                    stop_event.set()
                    break
                ts = datetime.datetime.now()
                q.put((ts, frame))
            except Exception as e:
                print(e)

        self.cap.release()


def main() -> None:
    """Main running function."""
    root = tk.Tk()
    root.withdraw()

    # get camera index
    camera_index, resolution = dialog_boxes.open_camera_popup(root)
    if camera_index == -1:
        print("Invalid index")
        return

    cap = FrameCapture(cam_index=camera_index, resolution=resolution)
    viewer = FrameViewer()

    # load config file if it exists.
    viewer.load_config("frame_viewer_config.ini")

    # make queues and stop event
    name_request_q = queue.Queue()
    name_result_q = queue.Queue()

    viewer.name_request_q = name_request_q
    viewer.name_result_q = name_result_q

    main_frame_communication_queue = queue.Queue()
    stop_event = threading.Event()

    # create child threads.
    t1 = threading.Thread(
        target=cap.run, args=(main_frame_communication_queue, stop_event), daemon=True
    )
    t2 = threading.Thread(
        target=viewer.run,
        args=(main_frame_communication_queue, stop_event),
        daemon=True,
    )

    t1.start()
    t2.start()

    # create root for helper windows
    root = tk.Tk()
    root.withdraw()
    while not stop_event.is_set():
        try:
            req, im = name_request_q.get(timeout=0.05)
        except queue.Empty:
            continue
        print(req)
        # REQUESTS QUEUE
        if req == "ASK_FOLDER_NAME":
            newname = simpledialog.askstring(
                "Set Target Folder Name",
                "Enter name of folder to save to:",
                parent=root,
            )

            name_result_q.put(newname)
        if req == "ASKSAVE":
            if im is not None:
                newname = simpledialog.askstring(
                    "Set Image Name", "Enter name for image:", parent=root
                )
                frame_to_save, prefix, saved, idx = im
                if newname:
                    filename = f"{prefix}_{newname}.png"
                else:
                    filename = f"{prefix}_{saved}.png"
                cv2.imwrite(filename, frame_to_save)
                print(f"Saved frame {idx} as {filename}")
            else:
                print("NO IMAGE DATA RECIEVED!")

        if req == "RESIZE":
            newdial = dialog_boxes.ResizeThresholdDialog(
                parent=root,
                title="Change displayed rows, cols, or threshold.",
                prompt="Set new rows/cols.",
                initial_columns=viewer.grid_cols,
                initial_rows=viewer.grid_rows,
                initial_threshold=viewer.change_threshold,
                minvalue=1,
                maxvalue=20,
                threshold_max=1.0,
                threshold_resolution=0.01,
            )
            w, h, t = newdial.result
            viewer._resize(grid_cols=w, grid_rows=h)

            viewer.change_threshold = t

        if req == "HELP":
            dialog_boxes.TextDialog(
                root,
                text=(HELP_TEXT),
                title="Information",
            )

    t2.join()
    stop_event.set()
    viewer.save_config("frame_viewer_config.ini")


if __name__ == "__main__":
    main()
