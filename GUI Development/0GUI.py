import os
import dearpygui.dearpygui as dpg
from GUI_helpers import (
    refresh_contents_list,
    open_folder_dialog,
    contents_list_callback,
    open_nd2_callback,
    z_slider_callback,
    run_rip_detector_callback,
    rip_checkbox_callback,
    wga_view_callback,
    update_texture,
    mask_click_callback,
    confirm_mask_selection_callback,
    save_metadata_callback
)
from analysis_helpers import extract_traces

dpg.create_context()
with dpg.texture_registry(show=False):
    dpg.add_dynamic_texture(1024, 1024, [1.0] * (1024 * 1024 * 4), tag="dynamic_texture")
dpg.create_viewport(title='GUI', width=1430, height=1120)

with dpg.window(tag="left_window", label="Controls", pos=(10, 10), width=330, height=200, no_move=True):
    dpg.add_button(label="Open Folder", callback=open_folder_dialog)
    dpg.add_text("None", tag="dir_path_repeat")
    dpg.add_checkbox(label="eGFP cells", tag="opt_egfp", default_value=True)
    dpg.add_checkbox(label="Save Meta Data", tag="opt_save_metadata", default_value=True)
    dpg.add_checkbox(label="Save Extracted Traces", tag="opt_save_traces", default_value=True)
    dpg.add_checkbox(label="Save Analyzed Data", tag="opt_save_analyzed", default_value=True)    
    dpg.add_text("Status: Ready", tag="status_text")

with dpg.window(tag="contents_window", label="Folder Contents", pos=(10, 220), width=330, height=505, no_move=True):
    dpg.add_listbox(items=[], tag="contents_list", num_items=10, width=315, callback=contents_list_callback)
    dpg.add_button(label="Open .nd2", width = 315, callback=open_nd2_callback)

    dpg.add_spacer(height=10)

    with dpg.group(tag="identifiers_group", show=False):
        with dpg.group(horizontal=True):
            dpg.add_text("DJID:")
            dpg.add_input_text(tag="djid_input", readonly=False, width=37)
            dpg.add_text("Age (mo):")
            dpg.add_input_text(tag="age_input", readonly=False, width=25)
            dpg.add_text("Genotype:")
            dpg.add_combo(items=["Homo", "Het", "WT", "Unknown"], tag="gen_combo", label="", width=50)
        with dpg.group(horizontal=True):
            dpg.add_text("Sex:")
            dpg.add_combo(items=["M", "F", "Unknown"], tag="sex_combo", label="", width=30)
            dpg.add_text("Eye:")
            dpg.add_combo(items=["L", "R", "Unknown"], tag="eye_combo", label="", width=30)
            dpg.add_text("Treatment:")
            dpg.add_combo(items=["Experimental", "Control"], tag="treatment_combo", label="", width=87)
        with dpg.group(horizontal=True):
            dpg.add_text("Duration (min):")
            dpg.add_input_text(tag="time_input", hint="e.g. 0, 15, 30, 60, 90", width=200)

    dpg.add_spacer(height=10)

    with dpg.group():
        with dpg.group(tag="wga_group", show=False, horizontal=True):
            dpg.add_text("WGA View")
            dpg.add_checkbox(tag="wga_checkbox", callback=wga_view_callback)
            dpg.add_slider_int(tag="wga_slider", min_value=0, max_value=0, width = 223, callback=wga_view_callback)

        dpg.add_spacer(height=10)

with dpg.window(tag="rip_panel", label="Rip Panel", pos=(10, 735), width=330, height=120, no_move=True):
    with dpg.group(tag="rip_group", show=False):
        dpg.add_checkbox(label="Rip?", tag="rip_checkbox", callback=rip_checkbox_callback)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Rip Detector Mode", tag="run_rip_button", show=False, callback=run_rip_detector_callback)
            dpg.add_checkbox(label="Show Masks", tag="show_masks_checkbox", default_value=True, show=False, callback=lambda s, a, u: update_texture())
        dpg.add_text("Selected: 0", tag="selected_mask_count")
        dpg.add_button(label="Confirm Masks", tag="confirm_masks_button", show=False, callback=confirm_mask_selection_callback)

with dpg.window(tag="analysis_panel", label="Analysis Panel", pos=(10, 865), width=330, height=150, no_move=True):
    dpg.add_button(label="Extract Traces", tag="extract_traces_button", width=315)
    dpg.add_text("File: None", tag="trace_file_status", wrap=280)
    dpg.add_text("Status: Waiting", tag="trace_status_text", wrap=280)
dpg.set_item_callback("extract_traces_button", extract_traces)


with dpg.window(tag="right_window", label="Image Panel", pos=(350, 10), width=1040, height=1060, no_move=True):
    with dpg.drawlist(tag="drawlist", width=1024, height=1024):
        dpg.draw_image("dynamic_texture", (0, 0), (1024, 1024))
    handler = dpg.add_item_handler_registry()
    dpg.add_item_clicked_handler(callback=mask_click_callback, parent=handler)
    dpg.bind_item_handler_registry("drawlist", handler)

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()