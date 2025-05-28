import os
import numpy as np
import pandas as pd
import dearpygui.dearpygui as dpg
import nd2
import cv2
from skimage import exposure
from scipy.ndimage import center_of_mass
from sklearn.neighbors import NearestNeighbors
from cellpose import models, denoise
import GUI_helpers

trace_data_df = None

def to_8bit(img):
    img = img.astype(np.float32)
    img -= img.min()
    img /= img.max()
    return (img * 255).astype(np.uint8)

def extract_masks(masks, mask_ids):
    return np.array([(masks == mask_id).astype(np.uint8) for mask_id in mask_ids])

def get_mask_diameter(mask):
    coords = np.column_stack(np.where(mask > 0))
    if coords.shape[0] == 0:
        return 15
    d = np.max(np.ptp(coords, axis=0))
    return max(5, d)

def get_sq_stacks(channels, mask):
    mask_coords = np.where(mask > 0)
    z, y, x = np.meshgrid(np.arange(len(channels[0])), np.arange(channels[0].shape[1]), np.arange(channels[0].shape[2]), indexing='ij')
    x0, x1 = np.min(mask_coords[2]), np.max(mask_coords[2])
    y0, y1 = np.min(mask_coords[1]), np.max(mask_coords[1])
    x0 = max(x0 - 20, 0)
    x1 = min(x1 + 20, channels[0].shape[2])
    y0 = max(y0 - 20, 0)
    y1 = min(y1 + 20, channels[0].shape[1])
    return np.array([c[:, y0:y1, x0:x1] for c in channels])

def extract_square_proj_expand(channels, mask, expansion):
    mask_coords = np.where(mask > 0)
    z_min, z_max = np.min(mask_coords[0]), np.max(mask_coords[0])
    z_center = (z_min + z_max) // 2
    z_start = max(z_center - 1, 0)
    z_end = min(z_center + 2, channels[0].shape[0])
    cropped = [c[z_start:z_end] for c in channels]
    proj = np.max(np.stack(cropped, axis=1), axis=2)
    return proj, z_center

def remove_boundary(mask, border):
    return mask[border:-border, border:-border]

def closest_mask_2d(reference_mask, masks):
    ref_center = np.array(np.argwhere(reference_mask > 0).mean(axis=0))
    ids = np.unique(masks)[1:]
    min_dist = float('inf')
    best_mask = None
    for i in ids:
        temp_mask = (masks == i)
        center = np.array(np.argwhere(temp_mask).mean(axis=0))
        dist = np.linalg.norm(center - ref_center)
        if dist < min_dist:
            min_dist = dist
            best_mask = temp_mask
    return best_mask.astype(np.uint8)

def get_traces(stacks, mask):
    return np.array([np.mean(stack[:, :][mask > 0]) for stack in stacks])

def nuclei_centers_of_mass(stack, mask):
    from scipy.ndimage import center_of_mass
    ids = np.unique(mask)
    ids = ids[ids != 0]
    return np.array([center_of_mass(stack, mask, i) for i in ids])

def remove_outliers_local(coords, num_closest_points=15, z_threshold=2):
    from sklearn.neighbors import NearestNeighbors
    z_values = coords[:, 0].reshape(-1, 1)
    nbrs = NearestNeighbors(n_neighbors=min(num_closest_points, len(z_values))).fit(z_values)
    distances, indices = nbrs.kneighbors(z_values)
    mean_z = np.mean(z_values[indices], axis=1)
    diff = np.abs(z_values.flatten() - mean_z.flatten())
    inliers = diff < z_threshold
    return coords[inliers], np.where(inliers)[0]

def auto_brightness_contrast(image):
    normalized_image = image.astype(np.float32) / 255.0
    equalized_image = exposure.equalize_adapthist(normalized_image)
    equalized_image = (equalized_image * 255).astype(np.uint8)
    return equalized_image

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

    for idx, row in GUI_helpers.metadata_df.iterrows():
        filename = row.get("filename")
        if not isinstance(filename, str) or not filename.strip():
            continue

        file_path = os.path.join(GUI_helpers.current_folder, filename)
        if not os.path.exists(file_path):
            continue

        z_min, z_max = int(row["z_min"]), int(row["z_max"])
        dpg.set_value("trace_file_status", f"File: {filename}")

        with nd2.ND2File(file_path) as f:
            z_sep = f.voxel_size().z
            stack = to_8bit(f.asarray())
            dapi_stack = stack[:, 0, :, :]
            egfp_stack = stack[:, 1, :, :]
            cropped_dapi = dapi_stack[z_min:z_max+1]
            cropped_egfp = egfp_stack[z_min:z_max+1]

        proj = np.max(cropped_dapi, axis=0)
        enhanced = auto_brightness_contrast(proj)

        dapi_masks, _, _, _ = dapi_model.eval(enhanced, diameter=None, channels=[0, 0])
        dapi_masks = dapi_masks.astype(np.int32)

        coords_3d = nuclei_centers_of_mass(cropped_dapi, dapi_masks)
        filtered_coords, filtered_idxs = remove_outliers_local(coords_3d, num_closest_points=15, z_threshold=2)
        filtered_masks = extract_masks(dapi_masks, filtered_idxs)
        dapi_masks = filtered_masks.copy()

        mask_ids = [m for m in np.unique(dapi_masks) if m > 0]
        all_masks = extract_masks(dapi_masks, mask_ids)

        for i, mask_id in enumerate(mask_ids):
            dpg.set_value("trace_status_text", f"Extracting mask {i+1} of {len(mask_ids)}")

            single_mask = all_masks[i]
            diam = get_mask_diameter(single_mask)
            expansion = 50

            sq_stacks = get_sq_stacks([cropped_dapi, cropped_egfp], single_mask)
            expanded_sq, z_level = extract_square_proj_expand([cropped_dapi, cropped_egfp], single_mask, expansion)

            expanded_mask, _, _ = wga_model.eval(expanded_sq, diameter=diam, channels=[0, 0])
            cleaned_mask = remove_boundary(expanded_mask, expansion)

            if len(np.unique(cleaned_mask)) == 1:
                continue
            elif len(np.unique(cleaned_mask)) > 2:
                cleaned_mask = closest_mask_2d(single_mask, cleaned_mask)

            trace = get_traces(sq_stacks, cleaned_mask)

            egfp_sum = np.sum(sq_stacks[1][z_level][cleaned_mask.astype(bool)])
            egfp_per_area = egfp_sum / np.sum(cleaned_mask)

            trace_length = len(trace)
            x_vals = np.arange(trace_length) * z_sep

            base = os.path.splitext(filename)[0]
            djid = base[:4]
            eye = base[4].upper() if len(base) > 4 else ""

            results.append({
                "filename": filename,
                "cell_id": int(mask_id),
                "trace": trace.tolist(),
                "x_vals": x_vals.tolist(),
                "egfp": float(egfp_per_area),
                "DJID": djid,
                "Eye": eye
            })

        dpg.set_value("trace_status_text", f"Finished {filename}")

    trace_data_df = pd.DataFrame(results)

    if not trace_data_df.empty:
        egfp_vals = trace_data_df["egfp"].values
        normalized = (egfp_vals - egfp_vals.min()) / (egfp_vals.max() - egfp_vals.min())
        trace_data_df["eGFP_Value"] = normalized > 0.2

    dpg.set_value("trace_file_status", "File: Done")
    dpg.set_value("trace_status_text", "Status: Complete")
    dpg.configure_item("extract_traces_button", enabled=True)
