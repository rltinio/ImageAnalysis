import os
import numpy as np
import pandas as pd
import dearpygui.dearpygui as dpg
import nd2
import cv2
from skimage import exposure, measure
from scipy.ndimage import center_of_mass
from skimage.measure import label, regionprops
from cellpose import models, denoise
import GUI_helpers

trace_data_df = None

def auto_brightness_contrast(image):
    normalized_image = image.astype(np.float32) / 255.0
    equalized_image = exposure.equalize_adapthist(normalized_image)
    equalized_image = (equalized_image * 255).astype(np.uint8)
    return equalized_image

def to_8bit(stack):
    stack = stack.astype(np.float32)
    stack -= np.min(stack)
    if np.max(stack) != 0:
        stack /= np.max(stack)
    return (255 * stack).astype(np.uint8)

def extract_masks(total_masks, points, reset_mask_ids=True):
    mod_masks = total_masks.copy()

    if isinstance(points, int):
        points = np.array([points])
    else:
        points = np.array(points)

    points = points + 1 

    mod_masks[~np.isin(mod_masks, points)] = 0

    if reset_mask_ids:
        unique_values = np.unique(mod_masks)
        unique_values.sort()
        for new_id, old_id in enumerate(unique_values):
            mod_masks[mod_masks == old_id] = new_id

    return mod_masks

def get_mask_diameter(mask):
    coords = np.column_stack(np.where(mask > 0))
    if coords.shape[0] == 0:
        return 15
    d = np.max(np.ptp(coords, axis=0))
    return max(5, d)

def square_mask(mask, perc_increase: int = 40):
    labeled_mask = measure.label(mask)
    regions = measure.regionprops(labeled_mask)
    largest_region = max(regions, key=lambda r: r.area)

    min_row, min_col, max_row, max_col = largest_region.bbox
    centroid = largest_region.centroid

    height = max_row - min_row
    width = max_col - min_col
    diameter = max(height, width)
    new_diameter = diameter * (1 + perc_increase / 100)
    half_side = new_diameter / 2

    top_left = (int(centroid[0] - half_side), int(centroid[1] - half_side))
    bottom_right = (int(centroid[0] + half_side), int(centroid[1] + half_side))

    top_left = (max(top_left[0], 0), max(top_left[1], 0))
    bottom_right = (min(bottom_right[0], mask.shape[0]), min(bottom_right[1], mask.shape[1]))

    new_mask = np.zeros_like(mask)
    new_mask[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]] = 1

    return new_mask

def get_sq_stacks(image, single_mask):
    sq_maski = square_mask(single_mask)

    min_row, min_col, max_row, max_col = regionprops(sq_maski.astype(int))[0].bbox

    sq_DAPI_stack = image[:,0, min_row:max_row, min_col:max_col]
    sq_eGFP_stack = image[:,1, min_row:max_row, min_col:max_col]
    sq_WGA_stack = image[:,2, min_row:max_row, min_col:max_col]
    sq_GLUT1_stack = image[:,3, min_row:max_row, min_col:max_col]

    sq_stacks = np.stack((sq_DAPI_stack, sq_eGFP_stack, sq_WGA_stack, sq_GLUT1_stack))

    return sq_stacks

def nucleus_com(single_channel, mask):
    masked_channel = single_channel * mask

    z_prof = np.sum(masked_channel, axis=(1, 2))
    z_max_idx = np.argmax(z_prof)

    com = center_of_mass(mask)

    com_3d = (int(com[0]), int(com[1]), z_max_idx)
    return com_3d


def extract_square_proj_expand(image, single_mask, extra_pixels = 50):
    DAPI_stack, WGA_stack = image[:, 0, :, :], image[:, 2, :, :]

    _, _, comzi = nucleus_com(DAPI_stack, single_mask)  # Gets the nucleus stack of the middle of the cell

    sq_maski = square_mask(single_mask)

    # Calculate the bounding box of the square mask
    min_row, min_col, max_row, max_col = regionprops(sq_maski.astype(int))[0].bbox

    # Dimensions of the region of interest
    roi_height = max_row - min_row
    roi_width = max_col - min_col

    # Dimensions of the new canvas with extra space
    new_height = roi_height + 2 * extra_pixels 
    new_width = roi_width + 2 * extra_pixels 
 
    # Create new black canvas (filled with zeros)
    new_WGA_slice = np.zeros((new_height, new_width), dtype=WGA_stack.dtype)
    new_DAPI_slice = np.zeros((new_height, new_width), dtype=DAPI_stack.dtype)

    # Calculate the placement of the ROI in the new canvas
    new_min_row = extra_pixels 
    new_min_col = extra_pixels 
 
    # Extract the region of interest and place it in the center of the new canvas
    sq_WGA_slice = WGA_stack[comzi, min_row:max_row, min_col:max_col]
    new_WGA_slice[new_min_row:new_min_row + roi_height, new_min_col:new_min_col + roi_width] = sq_WGA_slice

    return new_WGA_slice, comzi

def remove_boundary(mask, buffer=50):
    return mask[buffer:-buffer, buffer:-buffer] if buffer > 0 else mask

def closest_mask_2d(reference_mask, mask_array):
    ref_coords = np.argwhere(reference_mask)
    if len(ref_coords) == 0:
        return np.zeros_like(reference_mask)
    ref_center = np.mean(ref_coords, axis=0)
    labels = np.unique(mask_array)
    labels = labels[labels != 0]
    min_dist = float('inf')
    best_mask = np.zeros_like(reference_mask)
    for label in labels:
        candidate_mask = (mask_array == label)
        coords = np.argwhere(candidate_mask)
        center = np.mean(coords, axis=0)
        dist = np.linalg.norm(center - ref_center)
        if dist < min_dist:
            min_dist = dist
            best_mask = candidate_mask
    return best_mask.astype(np.uint8)

def get_traces(stacks, mask):
    traces = []
    for ch in stacks:
        ch_traces = []
        for z in ch:
            ch_traces.append(np.mean(z[mask > 0]))
        traces.append(ch_traces)
    return np.array(traces[0])  # assuming first channel is WGA

def nuclei_centers_of_mass(stack, masks):
    ids = np.unique(masks)
    ids = ids[ids != 0]
    return np.array([center_of_mass(stack, masks, idx) for idx in ids])

def remove_outliers_local(centers_of_mass, num_closest_points=20, z_threshold=2):
    if num_closest_points >= len(centers_of_mass):
        raise ValueError("num_closest_points must be less than the number of total points")
    filtered_data = []
    filtered_indices = []
    xs = np.array([coord[0] for coord in centers_of_mass])
    ys = np.array([coord[1] for coord in centers_of_mass])
    zs = np.array([coord[2] for coord in centers_of_mass])
    for i, (x, y, z) in enumerate(centers_of_mass):
        distances = np.sqrt((xs - x)**2 + (ys - y)**2 + (zs - z)**2)
        closest_indices = distances.argsort()[1:num_closest_points+1]
        z_closest = zs[closest_indices]
        mean_z = np.mean(z_closest)
        std_dev_z = np.std(z_closest)
        if abs(z - mean_z) <= z_threshold * std_dev_z:
            filtered_data.append((x, y, z))
            filtered_indices.append(i)
    return filtered_data, filtered_indices

def organize_data(trace_results, mask_id):
    df = pd.DataFrame({'trace': trace_results})
    df['mask_id'] = mask_id
    return df

def normalize(array):
    array = np.array(array)
    return (array - array.min()) / (array.max() - array.min())

def extract_traces():
    global trace_data_df
    print('extract traces')

    dpg.configure_item("extract_traces_button", enabled=False)
    dpg.set_value("trace_file_status", "File: Starting...")
    dpg.set_value("trace_status_text", "Status: Preparing...")

    results = []

    dapi_model_path = 'CP_models/T5_DAPI_V4'
    dapi_model = denoise.CellposeDenoiseModel(gpu=True, model_type=dapi_model_path, restore_type="deblur_cyto3")

    model_path_wga = 'CP_models/T5_WGA_V2'
    wga_model = models.CellposeModel(gpu=True, pretrained_model=model_path_wga)
    print('done loading models')
    
    for idx, row in GUI_helpers.metadata_df.iterrows():
        filename = row.get("filename")
        if not isinstance(filename, str) or not filename.strip():
            dpg.set_value("trace_status_text", "Error: Invalid filename in metadata.")
            dpg.configure_item("extract_traces_button", enabled=True)
            return

        file_path = os.path.join(GUI_helpers.current_folder, filename)
        if not os.path.exists(file_path):
            dpg.set_value("trace_status_text", f"Error: File not found - {file_path}")
            dpg.configure_item("extract_traces_button", enabled=True)
            return

        z_min, z_max = int(row["z_min"]), int(row["z_max"])
        dpg.set_value("trace_file_status", f"File: {filename}")

        with nd2.ND2File(file_path) as f:
            z_sep = f.voxel_size().z
            stack = to_8bit(f.asarray())
            dapi_stack = stack[:, 0, :, :]
            wga_stack = stack[:, 2, :, :]
            cropped_dapi = dapi_stack[z_min:z_max+1]
            cropped_wga = wga_stack[z_min:z_max+1]
        print('Found file', z_sep)

        proj = np.max(cropped_dapi, axis=0)
        enhanced = auto_brightness_contrast(proj)


        print('Now running dapi model')
        dapi_masks, _, _, _ = dapi_model.eval(enhanced, diameter=None, channels=[0, 0])
        print('Done running dapi model')

        print('Starting coords')
        coords_3d = nuclei_centers_of_mass(cropped_dapi, dapi_masks)
        print(len(coords_3d))
        filtered_coords, filtered_idxs = remove_outliers_local(coords_3d, num_closest_points=15, z_threshold=2)
        filtered_dapi_masks = extract_masks(dapi_masks, filtered_idxs)

        mask_ids = np.delete(np.unique(filtered_dapi_masks), 0) - 1

        print('Found masks', mask_ids)

        for i in mask_ids:
            dpg.set_value("trace_status_text", f"Extracting mask {i} of {len(mask_ids)}")

            single_mask = extract_masks(filtered_dapi_masks, i)
            print('single mask', np.unique(single_mask))
            diam = get_mask_diameter(single_mask)
            expansion = 50

            sq_stacks = get_sq_stacks(stack, single_mask)
            print('Passed sq_stacks', i, single_mask.shape)

            expanded_sq, z_level = extract_square_proj_expand(stack, single_mask, expansion)

            expanded_mask, _, _ = wga_model.eval(expanded_sq, diameter=diam, channels=[0, 0])
            cleaned_mask = remove_boundary(expanded_mask, expansion)

            if len(np.unique(cleaned_mask)) == 1:
                continue
            elif len(np.unique(cleaned_mask)) > 2:
                cleaned_mask = closest_mask_2d(single_mask, cleaned_mask)

            trace_results = get_traces(sq_stacks, cleaned_mask)

            eGFP_sum = np.sum(sq_stacks[1][z_level][cleaned_mask.astype(bool)])
            eGFP_sum_per_area = eGFP_sum / np.sum(cleaned_mask)

            cell_data = organize_data(trace_results, i)
            djid = row["djid"] if pd.notnull(row["djid"]) else ""
            eye = row["eye"] if pd.notnull(row["eye"]) else ""
            time_min = row["time_min"] if pd.notnull(row["time_min"]) else ""
            file_base = row["filename"] if pd.notnull(row["filename"]) else ""

            nested_array = np.array(range(stack.shape[0])) * z_sep
            cell_data['X_vals'] = [nested_array for _ in range(len(cell_data))]
            cell_data['file_name'] = file_base
            cell_data['DJID'] = djid
            cell_data['Eye'] = eye
            cell_data['Time_Min'] = time_min
            cell_data['eGFP_Value'] = False
            cell_data['eGFP_Raw_Intensity'] = eGFP_sum_per_area
            cell_data['in_rip'] = False

            results.append(cell_data)

        dpg.set_value("trace_status_text", f"Finished {filename}")

    if results:
        trace_data_df = pd.concat(results, ignore_index=True)
        egfp_vals = trace_data_df["eGFP_Raw_Intensity"].values
        normalized_vals = normalize(egfp_vals)
        trace_data_df["eGFP_Value"] = normalized_vals > 0.2

    dpg.set_value("trace_file_status", "File: Done")
    dpg.set_value("trace_status_text", "Status: Complete")
    dpg.configure_item("extract_traces_button", enabled=True)