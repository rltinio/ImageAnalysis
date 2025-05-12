import os
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import dearpygui.dearpygui as dpg
import nd2
from cellpose import models
from skimage.morphology import binary_erosion, disk
from skimage import img_as_ubyte
import cv2

current_folder = None
opened_file = None
channel_zstack = None
channel2_stack = None
gray_img = None
mask_array = None
colors = {}
selected_masks = []
boundaries_df = pd.DataFrame(columns=["filename","z_min","z_max"])
texture_cache = None
last_show_masks = True
last_selected = []

def to_8bit(arr):
    norm = arr.astype(np.float32)
    if norm.max() > 0:
        norm /= norm.max()
    return img_as_ubyte(norm)

def max_proj(channel_zstack):
    return np.max(channel_zstack, axis=0)

def draw_mask_outlines():
    print("\n[DEBUG] draw_mask_outlines called")
    global mask_array, gray_img, colors, selected_masks
    if mask_array is None or gray_img is None:
        return
    if not dpg.get_value("show_masks_checkbox"):
        return

    outline_rgba = np.zeros((*gray_img.shape, 4), dtype=np.float32)
    for m in np.unique(mask_array):
        if m == 0:
            continue
        mask = (mask_array == m).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        col = colors.get(m, np.random.rand(3))
        for contour in contours:
            cv2.polylines(outline_rgba, [contour], isClosed=True, color=(*col, 1.0), thickness=1)
            if m in selected_masks:
                cv2.fillPoly(outline_rgba, [contour], color=(*col, 0.4))

    base = to_8bit(gray_img).astype(np.float32) / 255.0
    rgba = np.zeros((*gray_img.shape, 4), dtype=np.float32)
    rgba[..., :3] = base[..., None]
    rgba[..., 3] = 1.0

    mask_alpha = outline_rgba[..., 3:4]
    rgba[..., :3] = (1 - mask_alpha) * rgba[..., :3] + mask_alpha * outline_rgba[..., :3]
    rgba[..., 3] = np.clip(rgba[..., 3] + outline_rgba[..., 3], 0, 1)

    dpg.set_value("dynamic_texture", rgba.flatten().tolist())


def blend_with_masks(gray):
    h, w = gray.shape
    base = to_8bit(gray).astype(np.float32) / 255.0
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    rgba[..., :3] = base[..., None]
    rgba[..., 3] = 1.0
    return rgba

def refresh_contents_list(sender=None, app_data=None, user_data=None):
    global current_folder
    if current_folder is None:
        dpg.configure_item("contents_list", items=[])
    else:
        files = sorted([f for f in os.listdir(current_folder) if f.lower().endswith('.nd2')])
        dpg.configure_item("contents_list", items=files)

def open_folder_dialog(sender, app_data, user_data):
    global current_folder, opened_file
    root = tk.Tk(); root.withdraw()
    folder = filedialog.askdirectory(); root.destroy()
    if not folder:
        return
    current_folder = folder
    opened_file = None
    two_level = f"{os.path.basename(os.path.dirname(folder))}\\{os.path.basename(folder)}"
    dpg.set_value("dir_path_repeat", two_level)
    refresh_contents_list()
    for tag in ("z_range_group","rip_group","wga_group"):
        if dpg.does_item_exist(tag):
            dpg.hide_item(tag)
    dpg.set_value("status_text", "Folder loaded")

def contents_list_callback(sender, app_data, user_data):
    sel = app_data
    show_z = (sel == opened_file)
    for tag in ("z_range_group","set_boundaries_button","z_min_slider","z_max_slider"):
        if dpg.does_item_exist(tag):
            dpg.show_item(tag) if show_z else dpg.hide_item(tag)
    show_rip = show_z and sel in boundaries_df['filename'].values
    if dpg.does_item_exist("rip_group"):
        dpg.show_item("rip_group") if show_rip else dpg.hide_item("rip_group")

def open_nd2_callback(sender, app_data, user_data):
    global opened_file, channel_zstack, channel2_stack, gray_img, mask_array, colors, selected_masks
    sel = dpg.get_value("contents_list")
    if not sel:
        dpg.set_value("status_text", "No file selected")
        return
    if sel != opened_file:
        dpg.set_value("status_text", f"Loading: {sel}")
        path = os.path.join(current_folder, sel)
        with nd2.ND2File(path) as f:
            stack8 = to_8bit(f.asarray())
        channel_zstack = stack8[:,0,:,:]
        channel2_stack = stack8[:,2,:,:]
        opened_file = sel
        add_z_range_widget("contents_window", channel_zstack.shape[0])
        dpg.set_value("contents_list", sel)
        gray_img = max_proj(channel_zstack)
        mask_array = np.zeros_like(gray_img, dtype=int)
        colors.clear(); selected_masks.clear()
        update_texture(gray_img, force=True)
        dpg.configure_item("wga_slider", min_value=0, max_value=channel2_stack.shape[0]-1)
        dpg.show_item("wga_group"); dpg.show_item("z_range_group")
        if sel in boundaries_df['filename'].values:
            dpg.show_item("rip_group")
        dpg.set_value("status_text", f"Loaded: {sel}")
    else:
        dpg.set_value("status_text", f"Already loaded: {sel}")

def z_slider_callback(sender, app_data, user_data):
    global gray_img
    if channel_zstack is None:
        return
    z0, z1 = dpg.get_value("z_min_slider"), dpg.get_value("z_max_slider")
    gray_img = max_proj(channel_zstack[z0:z1+1])
    update_texture(gray_img, force=True)

def set_boundaries_callback(sender, app_data, user_data):
    global boundaries_df, mask_array, selected_masks, colors, texture_cache
    if opened_file is None:
        return
    z0 = dpg.get_value("z_min_slider")
    z1 = dpg.get_value("z_max_slider")
    if opened_file in boundaries_df['filename'].values:
        boundaries_df.loc[boundaries_df['filename'] == opened_file, ['z_min', 'z_max']] = [z0, z1]
    else:
        boundaries_df.loc[len(boundaries_df)] = [opened_file, z0, z1]
    mask_array = None
    selected_masks.clear()
    colors.clear()
    texture_cache = None
    dpg.set_value("status_text", f"{opened_file}: {z0}-{z1}")
    if opened_file in boundaries_df["filename"].values:
        dpg.show_item("rip_group")
    else:
        dpg.hide_item("rip_group")
    update_texture(gray_img, force=True)


def rip_checkbox_callback(sender, app_data, user_data):
    dpg.set_value("show_masks_checkbox", False)
    global mask_array, colors, selected_masks, texture_cache
    dpg.configure_item("run_rip_button", show=app_data)
    dpg.configure_item("show_masks_checkbox", show=app_data)
    if not app_data:
        mask_array = None
        colors.clear()
        selected_masks.clear()
        texture_cache = None
        update_texture(gray_img, force=True)

def run_rip_detector_callback(sender, app_data, user_data):
    global boundaries_df, mask_array, colors, selected_masks, texture_cache

    if opened_file not in boundaries_df["filename"].values:
        dpg.set_value("status_text", "Set Z boundaries before running rip detector.")
        return

    z0 = dpg.get_value("z_min_slider")
    z1 = dpg.get_value("z_max_slider")
    boundaries_df.loc[boundaries_df["filename"] == opened_file, ["z_min", "z_max"]] = [z0, z1]

    mask_array = None
    selected_masks.clear()
    colors.clear()
    texture_cache = None

    dpg.set_value("status_text", "Rip detection started")
    dpg.show_item("show_masks_checkbox")
    dpg.set_value("show_masks_checkbox", True)
    update_texture(gray_img, force=True)

    model_path = 'CP_models/T5_DAPI_V4'
    cp_model = models.CellposeModel(gpu=True, pretrained_model=model_path)
    masks, _, _ = cp_model.eval(gray_img, diameter=None)

    mask_array = masks
    colors.update({m: np.random.rand(3) for m in np.unique(mask_array) if m > 0})
    selected_masks.clear()
    update_texture(gray_img, force=True)

    dpg.set_value("status_text", "Rip detection complete")



def wga_view_callback(sender, app_data, user_data):
    img = gray_img if not dpg.get_value("wga_checkbox") else channel2_stack[dpg.get_value("wga_slider")]
    update_texture(img, force=True)

def update_texture(base_img=None, force=False):
    global gray_img, channel2_stack, texture_cache, last_show_masks, last_selected, mask_array, selected_masks, colors
    print("[DEBUG] update_texture called")
    if base_img is None:
        base_img = gray_img if not dpg.get_value("wga_checkbox") else channel2_stack[dpg.get_value("wga_slider")]

    show_masks = dpg.get_value("show_masks_checkbox")
    if not force and texture_cache is not None and show_masks == last_show_masks and selected_masks == last_selected:
        dpg.set_value("dynamic_texture", texture_cache)
        return

    print(f"[DEBUG] Image min: {np.min(base_img)}, max: {np.max(base_img)}")
    base = to_8bit(base_img).astype(np.float32) / 255.0
    rgba = np.zeros((*base_img.shape, 4), dtype=np.float32)
    rgba[..., :3] = base[..., None]
    rgba[..., 3] = 1.0

    if show_masks and mask_array is not None:
        print("[DEBUG] embedding mask outlines in update_texture")
        outline_rgba = np.zeros((*base_img.shape, 4), dtype=np.float32)
        for m in np.unique(mask_array):
            if m == 0:
                continue
            mask = (mask_array == m).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            col = colors.get(m, np.random.rand(3))
            for contour in contours:
                cv2.polylines(outline_rgba, [contour], isClosed=True, color=(*col, 1.0), thickness=1)
                if m in selected_masks:
                    cv2.fillPoly(outline_rgba, [contour], color=(*col, 0.4))
        mask_alpha = outline_rgba[..., 3:4]
        rgba[..., :3] = (1 - mask_alpha) * rgba[..., :3] + mask_alpha * outline_rgba[..., :3]
        rgba[..., 3] = np.clip(rgba[..., 3] + outline_rgba[..., 3], 0, 1)

    texture_cache = rgba.flatten().tolist()
    last_show_masks = show_masks
    last_selected = selected_masks.copy()
    dpg.set_value("dynamic_texture", texture_cache)

def mask_click_callback(sender, app_data, user_data):
    if not dpg.get_value("show_masks_checkbox"):
        return
    if mask_array is None or mask_array.max() == 0:
        return
    mx, my = dpg.get_mouse_pos(local=False)
    x0, y0 = dpg.get_item_rect_min("drawlist")
    ix, iy = int(mx - x0), int(my - y0)
    if ix < 0 or iy < 0 or ix >= gray_img.shape[1] or iy >= gray_img.shape[0]:
        return
    m = int(mask_array[iy, ix])
    if m > 0:
        if m in selected_masks:
            selected_masks.remove(m)
        else:
            selected_masks.append(m)
        current_img = gray_img if not dpg.get_value("wga_checkbox") else channel2_stack[dpg.get_value("wga_slider")]
        update_texture(current_img, force=True)
        dpg.set_value("status_text", f"Selected mask: {m}")


def add_z_range_widget(parent, depth):
    for tag in ["z_range_group","z_min_slider","z_max_slider","set_boundaries_button"]:
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)
    mid = depth // 2
    with dpg.group(parent=parent, horizontal=True, tag="z_range_group"):
        dpg.add_text("Z Range:")
        dpg.add_button(label="Set Boundaries", tag="set_boundaries_button", callback=set_boundaries_callback)
    dpg.add_slider_int(label="Min Z", tag="z_min_slider", parent=parent, min_value=0, max_value=mid, default_value=0, callback=z_slider_callback)
    dpg.add_slider_int(label="Max Z", tag="z_max_slider", parent=parent, min_value=mid, max_value=depth-1, default_value=depth-1, callback=z_slider_callback)