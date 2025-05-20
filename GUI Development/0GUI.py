import os
import dearpygui.dearpygui as dpg
from GUI_helpers import (
    refresh_contents_list,
    open_folder_dialog,
    contents_list_callback,
    open_nd2_callback,
    z_slider_callback,
    set_boundaries_callback,
    run_rip_detector_callback,
    rip_checkbox_callback,
    wga_view_callback,
    update_texture,
    mask_click_callback,
    confirm_mask_selection_callback,
    save_metadata_callback
)

dpg.create_context()
with dpg.texture_registry(show=False):
    dpg.add_dynamic_texture(1024, 1024, [1.0] * (1024 * 1024 * 4), tag="dynamic_texture")
dpg.create_viewport(title='Masked Image GUI', width=1400, height=1100)

with dpg.window(tag="left_window", label="Controls", pos=(10, 10), width=300, height=300, no_move=True):
    dpg.add_button(label="Open Folder", callback=open_folder_dialog)
    dpg.add_text("None", tag="dir_path_repeat")
    dpg.add_checkbox(label="eGFP cells", tag="opt_egfp", default_value=True)
    dpg.add_checkbox(label="Save MP details", tag="opt_save", default_value=True)
    dpg.add_text("Status: Ready", tag="status_text")

with dpg.window(tag="contents_window", label="Folder Contents", pos=(10, 320), width=300, height=490, no_move=True):
    dpg.add_listbox(items=[], tag="contents_list", num_items=10, width=280, callback=contents_list_callback)
    dpg.add_button(label="Open .nd2", width = 280, callback=open_nd2_callback)

    dpg.add_spacer(height=10)

    with dpg.group(tag="identifiers_group", show=False):
        with dpg.group(horizontal=True):
            dpg.add_text("Sex:")
            dpg.add_radio_button(items=["M", "F", "Unknown"], tag="sex_radio", label="Sex", horizontal=True)
        with dpg.group(horizontal=True):
            dpg.add_text("Eye:")
            dpg.add_radio_button(items=["L", "R", "Unknown"], tag="eye_radio", label="Eye", horizontal=True)
        with dpg.group(horizontal=True):
            dpg.add_text("Time Condition (min):")
            dpg.add_input_text(tag="time_input", hint="e.g. 0, 15, 30", width=120)

    dpg.add_spacer(height=10)

    with dpg.group():
        with dpg.group(tag="wga_group", show=False, horizontal=True):
            dpg.add_text("WGA View")
            dpg.add_checkbox(tag="wga_checkbox", callback=wga_view_callback)
            dpg.add_slider_int(tag="wga_slider", min_value=0, max_value=0, width = 185, callback=wga_view_callback)

        dpg.add_spacer(height=10)

with dpg.window(tag="rip_panel", label="Rip Panel", pos=(10, 820), width=300, height=180, no_move=True):
    with dpg.group(tag="rip_group", show=False):
        dpg.add_checkbox(label="Rip?", tag="rip_checkbox", callback=rip_checkbox_callback)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Rip Detector Mode", tag="run_rip_button", show=False, callback=run_rip_detector_callback)
            dpg.add_checkbox(label="Show Masks", tag="show_masks_checkbox", default_value=True, show=False, callback=lambda s, a, u: update_texture())
        dpg.add_text("Selected: 0", tag="selected_mask_count")
        dpg.add_button(label="Confirm Masks", tag="confirm_masks_button", show=False, callback=confirm_mask_selection_callback)

with dpg.window(tag="right_window", label="Image Panel", pos=(355, 10), width=1024, height=1024, no_move=True):
    with dpg.drawlist(tag="drawlist", width=1024, height=1024):
        dpg.draw_image("dynamic_texture", (0, 0), (1024, 1024))
    handler = dpg.add_item_handler_registry()
    dpg.add_item_clicked_handler(callback=mask_click_callback, parent=handler)
    dpg.bind_item_handler_registry("drawlist", handler)

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()