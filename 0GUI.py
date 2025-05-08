import os
import threading
import tkinter as tk
from tkinter import filedialog
import dearpygui.dearpygui as dpg

# ---------------------------
# Callback functions
# ---------------------------

def open_folder_dialog(sender, app_data, user_data):
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory()
    root.destroy()
    if not folder:
        dpg.set_value("dir_path", "None")
        dpg.set_value("dir_path_repeat", "None")
        dpg.configure_item("contents_list", items=[])
        return
    parent = os.path.basename(folder)
    grandparent = os.path.basename(os.path.dirname(folder))
    two_level = f"{grandparent}\\{parent}"
    dpg.set_value("dir_path", two_level)
    dpg.set_value("dir_path_repeat", two_level)
    try:
        contents = os.listdir(folder)
    except Exception:
        contents = []
    dpg.configure_item("contents_list", items=contents)
    dpg.set_value("status_text", "Status: Ready")


def start_timer(sender, app_data, user_data):
    dpg.set_value("timer_count", "0")
    dpg.set_value("status_text", "Status: Running")
    for i in range(1, 6):
        threading.Timer(i, lambda i=i: dpg.set_value("timer_count", str(i))).start()
    threading.Timer(5.0, lambda: dpg.set_value("status_text", "Status: Done")).start()


def viewport_resize_callback(sender, app_data):
    # Called when viewport size changes; reposition and resize panes
    width, height = app_data
    half = (width - 30) // 2
    pane_height = height - 20
    # Resize left and right panes
    dpg.configure_item("left_window", width=half, height=pane_height)
    dpg.configure_item("right_window", pos=(half + 20, 10), width=half, height=pane_height)

# ---------------------------
# App initialization
# ---------------------------

dpg.create_context()

dpg.create_viewport(title='Three-Pane DearPyGui App', width=800, height=800)

dpg.set_viewport_resize_callback(viewport_resize_callback)

# ---------------------------
# Build UI
# ---------------------------

# Left pane: controls
with dpg.window(tag="left_window", label="Controls", pos=(10, 10), width=335, height=300):
    dpg.add_button(label="Open Folder", callback=open_folder_dialog, width=-1)
    dpg.add_text("Selected Folder:")
    dpg.add_text("None", tag="dir_path", wrap=310)
    dpg.add_text("")  # Spacer
    dpg.add_checkbox(label="eGFP cells", tag="opt_egfp", default_value=True)
    dpg.add_checkbox(label="WGA rips", tag="opt_wga", default_value=False)
    dpg.add_checkbox(label="Save MP details", tag="opt_save", default_value=True)
    # Initial status position updated on first show via resize callback
    dpg.add_text("Status: Ready", tag="status_text")

# Middle pane: folder contents below controls
with dpg.window(tag="contents_window", label="Folder Contents", pos=(10, 320), width=335, height=400):
    dpg.add_text("Selected Folder:")
    dpg.add_text("None", tag="dir_path_repeat", wrap=310)
    dpg.add_listbox(items=[], tag="contents_list", num_items=15, width=-1)

# Right pane: timer
with dpg.window(tag="right_window", label="Output", pos=(355, 10), width=335, height=710):
    dpg.add_text("Right panel timer")
    dpg.add_button(label="Start Timer", callback=start_timer, width=-1)
    dpg.add_text("0", tag="timer_count")

# ---------------------------
# Run app
# ---------------------------

dpg.setup_dearpygui()

dpg.show_viewport()

dpg.start_dearpygui()

dpg.destroy_context()
