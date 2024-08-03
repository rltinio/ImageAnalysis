import numpy as np
from skimage import exposure, measure
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import center_of_mass
from scipy.spatial import ConvexHull, distance_matrix
from scipy.signal import find_peaks
from skimage.measure import label, regionprops
import pandas as pd
#from cellpose import utils, io, plot, models, denoise
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


### FREQUENTLY USED FUNCTIONS ###

'''
Compresses the brightest pixels from the entire stack into one slice
'''

def max_proj(channel_zstack):
    channel_max = np.max(channel_zstack, axis = 0)
    return channel_max

'''
Applies "auto" from bright and contrast plug-in from FIJI
Does not follow the same algorithmn
'''
def auto_brightness_contrast(image):
    # Convert to float and normalize to range [0, 1]
    normalized_image = image.astype(np.float32) / 255.0
    equalized_image = exposure.equalize_adapthist(normalized_image)
    equalized_image = (equalized_image * 255).astype(np.uint8)
    
    return equalized_image

'''
Returning z indicies where the nuclei of each roi has the greatest intensity
'''

def nuclei_slices(single_channel, masks):

    total_masks = np.delete(np.unique(masks,0),0)
    number_slices = single_channel.shape[0]
    
    max_z_slices = []

    for mask in total_masks:
        single_mask = masks == mask
        
        intensity_max = 0
        intensity_max_slice = 0

        for slice in range(number_slices):
            layer_intensity = np.sum(single_channel[slice][single_mask])

            if layer_intensity > intensity_max:
                intensity_max = layer_intensity
                intensity_max_slice = slice

        max_z_slices.append(intensity_max_slice)

    return max_z_slices

def extract_masks(total_masks, points, reset_mask_ids=True):
    mod_masks = total_masks.copy()

    # Ensure points is a numpy array
    if isinstance(points, int):
        points = np.array([points])
    else:
        points = np.array(points)

    points = points + 1  # Increment to maintain zero-based indexing

    # Mask to keep only the desired points
    mod_masks[~np.isin(mod_masks, points)] = 0

    if reset_mask_ids:
        unique_values = np.unique(mod_masks)
        unique_values.sort()  # Sort unique values to maintain order
        for new_id, old_id in enumerate(unique_values):
            mod_masks[mod_masks == old_id] = new_id

    return mod_masks

def get_mask_diameter(mask):
    """
    Calculate the diameter of a binary mask, defined as the maximum distance
    between any two points on the boundary of the mask.

    Parameters:
    - mask: 2D numpy array (binary mask).

    Returns:
    - diameter: The maximum distance between any two boundary points in the mask.
    """
    # Label connected regions in the binary mask
    labeled_mask = label(mask)
    
    # Ensure the mask contains only one region (consider the largest if there are multiple)
    largest_region = max(regionprops(labeled_mask), key=lambda r: r.area)
    
    # Get the coordinates of the boundary of the largest region
    boundary_coords = largest_region.coords
    
    # Calculate all pairwise distances between boundary points
    distances = np.linalg.norm(boundary_coords[:, np.newaxis] - boundary_coords, axis=2)
    
    # Get the maximum distance found
    diameter = np.max(distances)
    
    return diameter

'''
Somehow this code below is slower than the one above. They do the same thing
'''

def nuclei_slices_exp(single_channel, masks):

    total_masks = np.delete(np.unique(masks,0),0)
    
    max_z_slices = []

    for mask in total_masks:
        single_mask = masks == mask

        masked_channel = single_channel * single_mask

        z_prof = np.sum(masked_channel, axis = (1,2))
        
        z_max_idx = np.argmax(z_prof)

        max_z_slices.append(z_max_idx)

    return max_z_slices

def cell_projector(single_channel, masks, nuclei_z_indicies):
    
    max_average = int(np.sum(nuclei_z_indicies)/len(nuclei_z_indicies))
    edited_image = single_channel[max_average].copy()
    base_image_display = single_channel[max_average].copy()
    total_masks = np.delete(np.unique(masks,0),0) - 1

    for mask in total_masks:
        single_mask = masks == mask
        single_slice = nuclei_z_indicies[mask]
        edited_image[single_mask] = single_channel[single_slice][single_mask]

    return base_image_display, edited_image

def to_8bit(image, do_scaling=True):
    # Ensure the input image is a NumPy array
    image = np.asarray(image)
    
    # Initialize the output image with zeros, of dtype np.uint8
    output_image = np.zeros(image.shape, dtype=np.uint8)
    
    if do_scaling:
        # Determine the min and max values from the image
        min_val = image.min()
        max_val = image.max()
        
        # Avoid division by zero if min_val == max_val
        if max_val > min_val:
            scale = 255.0 / (max_val - min_val)
            output_image = ((image - min_val) * scale).clip(0, 255).astype(np.uint8)
        else:
            # If the image has a single value, map it to 0 or 255
            output_image.fill(255 if min_val > 0 else 0)
    else:
        # If not scaling, ensure the values are within the 0-255 range
        output_image = image.clip(0, 255).astype(np.uint8)
    
    return output_image

### PLOTTING FUNCTIONS ###

def plot_2images(before_image, after_image, auto_bc=False, titles:list = ['Before', 'After']):
    
    if auto_bc:
        before_image = auto_brightness_contrast(before_image.copy())
        after_image = auto_brightness_contrast(after_image.copy())

    fig = plt.figure(figsize=(6, 6))
    gs = GridSpec(1, 2, figure=fig, wspace=0.05)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    ax0.imshow(before_image)
    ax0.axis('off')
    ax0.set_title(titles[0])

    ax1.imshow(after_image)
    ax1.axis('off')
    ax1.set_title(titles[1])

    plt.show()

def plot_4images(before_images, after_images, auto_bc=False):

    c1_before, c2_before = before_images
    c1_after, c2_after = after_images
    
    if auto_bc:
        c1_before = auto_brightness_contrast(c1_before.copy())
        c2_before = auto_brightness_contrast(c2_before.copy())
        c1_after = auto_brightness_contrast(c1_after.copy())
        c2_after = auto_brightness_contrast(c2_after.copy())

    fig = plt.figure(figsize=(6, 6))
    gs = GridSpec(2, 2, figure=fig, wspace=0.05, hspace=0.05)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    ax0.imshow(c1_before)
    ax0.axis('off')

    ax1.imshow(c1_after)
    ax1.axis('off')

    ax2.imshow(c2_before)
    ax2.axis('off')

    ax3.imshow(c2_after)
    ax3.axis('off')

    plt.show()

def plot_maskids(center_of_masses, text_color:str = 'red', font_size:int = 5):

    if len(center_of_masses[0]) != 2:
        raise Exception('Function expects 2D or xy points')
    
    for idx, coords in enumerate(center_of_masses):
        plt.annotate(text=str(idx), xy=(coords[1], coords[0]), xytext=(coords[1], coords[0]), color=text_color, fontsize=font_size, ha='center', va='center')

### Projection ###


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

def WGA_projector(DAPI_stack, WGA_stack, single_mask, z_sep):
    sq_maski = square_mask(single_mask, perc_increase=10)
    comzi = nucleus_com(DAPI_stack, single_mask)

    above_and_below = int(0.5 / z_sep)
    min_slice = max(0, comzi[2] - above_and_below)
    max_slice = min(WGA_stack.shape[0], comzi[2] + above_and_below + 1)

    sq_stack = WGA_stack[min_slice:max_slice] * sq_maski

    sq_stack = ~(max_proj(~sq_stack))

    return sq_stack, sq_maski

def WGA_stitcher(DAPI_stack, WGA_stack, masks, z_sep):
    z_avg = int(np.mean(nuclei_slices(DAPI_stack, masks)))
    base_stack = (WGA_stack[z_avg])

    total_masks = np.delete(np.unique(masks), 0)

    for mask_num in total_masks:
        maski = masks == mask_num
        sq_stacki, sq_maski = WGA_projector(DAPI_stack, WGA_stack, maski, z_sep)
        if sq_stacki is not None:
            base_stack[sq_maski] = sq_stacki[sq_maski]

    return base_stack

def nucleus_com(single_channel, mask):
    masked_channel = single_channel * mask

    z_prof = np.sum(masked_channel, axis=(1, 2))
    z_max_idx = np.argmax(z_prof)

    com = center_of_mass(mask)

    com_3d = (int(com[0]), int(com[1]), z_max_idx)
    return com_3d

def extract_square_proj_expand(image, single_mask, extra_pixels = 50):
    '''
    Modification of upload_training_cells

    Usually modularity is important, however it's more practical that these functions feed into one another here
    '''

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

    return new_WGA_slice

def remove_pixels_from_edges(array, pixels_to_remove):
    """
    Removes a specified number of pixels from each edge of the array.

    Parameters:
    - array (np.ndarray): The input array (2D or 3D).
    - pixels_to_remove (int): The number of pixels to remove from each edge.

    Returns:
    - np.ndarray: The resulting array with reduced size.
    """
    if len(array.shape) == 2:
        # For 2D arrays
        rows, cols = array.shape
        return array[pixels_to_remove:rows-pixels_to_remove, pixels_to_remove:cols-pixels_to_remove]
    elif len(array.shape) == 3:
        # For 3D arrays
        depth, rows, cols = array.shape
        return array[:, pixels_to_remove:rows-pixels_to_remove, pixels_to_remove:cols-pixels_to_remove]
    else:
        raise ValueError("Array must be 2D or 3D")
    
def get_sq_stacks(image, single_mask):
    sq_maski = square_mask(single_mask)

    # Extract bounding box around the mask
    min_row, min_col, max_row, max_col = regionprops(sq_maski.astype(int))[0].bbox

    sq_DAPI_stack = image[:,0, min_row:max_row, min_col:max_col]
    sq_eGFP_stack = image[:,1, min_row:max_row, min_col:max_col]
    sq_WGA_stack = image[:,2, min_row:max_row, min_col:max_col]
    sq_GLUT1_stack = image[:,3, min_row:max_row, min_col:max_col]

    sq_stacks = np.stack((sq_WGA_stack, sq_DAPI_stack, sq_eGFP_stack, sq_GLUT1_stack))

    return sq_stacks

# sq_stack and mask
def get_traces(sq_stacks, single_mask):
    sq_WGA_stack, sq_DAPI_stack, sq_eGFP_stack, sq_GLUT1_stack = sq_stacks

    # Retrieving values in a vectorized manner
    stacks = np.array([sq_WGA_stack, sq_DAPI_stack, sq_eGFP_stack, sq_GLUT1_stack])
    masked_stacks = stacks * single_mask
    summed_values = np.sum(masked_stacks, axis=(-1, -2))  # Sum along the spatial dimensions

    # Store the results with corresponding channel names
    channel_names = ['WGA', 'DAPI', 'eGFP', 'GluT1']
    result = {name: value for name, value in zip(channel_names, summed_values)}

    return result

import numpy as np

def remove_boundary(array, pixels_to_remove):
    """
    Removes a specified number of pixels from each edge of the array.

    Parameters:
    - array (np.ndarray): The input array (2D or 3D).
    - pixels_to_remove (int): The number of pixels to remove from each edge.

    Returns:
    - np.ndarray: The resulting array with reduced size.
    """
    if len(array.shape) == 2:
        # For 2D arrays
        rows, cols = array.shape
        return array[pixels_to_remove:rows-pixels_to_remove, pixels_to_remove:cols-pixels_to_remove]
    elif len(array.shape) == 3:
        # For 3D arrays
        depth, rows, cols = array.shape
        return array[:, pixels_to_remove:rows-pixels_to_remove, pixels_to_remove:cols-pixels_to_remove]
    else:
        raise ValueError("Array must be 2D or 3D")
    
def organize_data(trace_results, mask_id):
    
    summed_values_list = []
    mask_id_list = []
    channel_list = []

    for channel_name, summed_values in trace_results.items():
            summed_values_list.append(summed_values.tolist())
            mask_id_list.append(mask_id)
            channel_list.append(channel_name)

    # Create a DataFrame from the collected data
    data = {
        'Y_vals': summed_values_list,
        'mask_id': mask_id_list,
        'Stain': channel_list
    }
    df = pd.DataFrame(data)

    return df


### COORDINATE FUNCTIONS ###

def nuclei_centers_of_mass(single_channel, masks):
    """
    Returns the (x, y, z) coordinates of the center of mass for each mask.

    :param single_channel: 3D numpy array (slices, height, width) of a single channel.
    :param masks: 3D numpy array (same shape as single_channel) with mask labels.
    :return: List of tuples containing (x, y, z) coordinates of the center of mass.
    """
    total_masks = np.delete(np.unique(masks,0),0)
    
    centers_of_mass = []

    for mask in total_masks:
        single_mask = masks == mask

        masked_channel = single_channel * single_mask

        z_prof = np.sum(masked_channel, axis = (1,2))
        
        z_max_idx = np.argmax(z_prof)

        # Calculate the center of mass for the mask
        com = center_of_mass(single_mask)
        
        # The z coordinate of the center of mass is determined by the slice with the max intensity
        center_of_mass_3d = (int(com[0]), int(com[1]), z_max_idx)

        centers_of_mass.append(center_of_mass_3d)

    return centers_of_mass

def get_nuclei_position(masks):

    cell_coords = []

    for maski in np.unique(masks):
        
        single_mask = extract_masks(masks, maski)

        cell_coords.append(center_of_mass(single_mask))

    return cell_coords

def remove_outliers(data):
    """
    Remove numbers from the list that have a standard deviation greater than 2 from the mean.

    :param data: List of numbers.
    :return: List of numbers with outliers removed.
    """
    mean = np.mean(data)
    std_dev = np.std(data) * 2

    # Define the lower and upper bounds
    lower_bound = mean - std_dev
    upper_bound = mean + std_dev

    # Filter the data to remove outliers
    filtered_data = [x for x in data if lower_bound <= x <= upper_bound]
    
    return filtered_data

def remove_outliers_3D(data):
    """
    Expects data is a in a tuple (x,y,z)

    """
    Z_idxs = [i[2] for i in data]

    mean = np.mean(Z_idxs)
    std_dev = np.std(Z_idxs) * 2

    # Define the lower and upper bounds
    lower_bound = mean - std_dev
    upper_bound = mean + std_dev

    # Filter the data to remove outliers
    filtered_data = [x for x in data if lower_bound <= x[2] <= upper_bound]
    
    return filtered_data

def remove_outliers_local(centers_of_mass, num_closest_points=20, z_threshold=2):
    """
    Removes points that are outliers based on the z-values of their closest neighbors.

    :param centers_of_mass: List of tuples containing (x, y, z) coordinates of the centers of mass.
    :param num_closest_points: Number of closest points to consider for outlier detection.
    :param z_threshold: Number of standard deviations from the mean z-value to consider as an outlier.

    :return: Tuple containing filtered list of (x, y, z) coordinates and the indices of the filtered points.
    """
    if num_closest_points >= len(centers_of_mass):
        raise ValueError("num_closest_points must be less than the number of total points")

    filtered_data = []
    filtered_indices = []
    xs = np.array([coord[0] for coord in centers_of_mass])
    ys = np.array([coord[1] for coord in centers_of_mass])
    zs = np.array([coord[2] for coord in centers_of_mass])

    for i, (x, y, z) in enumerate(centers_of_mass):
        # Calculate distances from the current point to all other points
        distances = np.sqrt((xs - x)**2 + (ys - y)**2 + (zs - z)**2)

        # Get indices of the closest points, excluding the point itself
        closest_indices = distances.argsort()[1:num_closest_points+1]

        # Calculate mean and standard deviation of the z-values of these closest points
        z_closest = zs[closest_indices]
        mean_z = np.mean(z_closest)
        std_dev_z = np.std(z_closest)

        # Check if the current point is an outlier in z-value
        if abs(z - mean_z) <= z_threshold * std_dev_z:
            filtered_data.append((x, y, z))
            filtered_indices.append(i)

    return filtered_data, filtered_indices

def remove_associated_masks(masks, filtered_indices):
    """
    Removes masks associated with the filtered coordinates.

    :param masks: List of masks.
    :param filtered_indices: Indices of the coordinates that are not outliers.

    :return: Filtered list of masks.
    """
    return [masks[i] for i in filtered_indices]

def plot_3d_points(centers_of_mass, view_angle = (45, 45)):
    """
    Plots the (x, y, z) coordinates of the centers of mass.

    :param centers_of_mass: List of tuples containing (x, y, z) coordinates of the centers of mass.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs = [coord[0] for coord in centers_of_mass]
    ys = [coord[1] for coord in centers_of_mass]
    zs = [coord[2] for coord in centers_of_mass]

    ax.scatter(xs, ys, zs, c='r', marker='o')

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    # Set ticks for x, y, and z axes
    ax.set_zticks(np.arange(0, 81, 10))

    ax.set_zlim(0, 80)
    ax.view_init(view_angle[0], view_angle[1])

    # Move the z-axis to the opposite side
    ax.zaxis._axinfo['juggled'] = (1, 2, 0)

    plt.show()
def plot_3d_surface(comp, view_angle=(25, 45), cmap=cm.jet):
    """
    Plots a 3D surface using trisurf.

    Parameters:
    - comp: List of tuples containing (x, y, z) coordinates.
    - view_angle: Tuple specifying the (elevation, azimuth) for the view angle.
    - cmap: Colormap for the surface.

    Returns:
    - None: Displays the 3D plot.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs = [coord[0] for coord in comp]
    ys = [coord[1] for coord in comp]
    zs = [coord[2] for coord in comp]

    # Get the min and max of the z-values for consistent colormap limits
    z_min, z_max = min(zs), max(zs)

    surf = ax.plot_trisurf(xs, ys, zs, cmap=cmap, linewidth=0, vmin=z_min, vmax=z_max)
    fig.colorbar(surf)

    # Set axis labels
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    ax.set_zlim(0, 80)

    # Set ticks for x, y, and z axes
    ax.set_zticks(np.arange(0, 81, 10))

    # Move the z-axis to the opposite side
    ax.zaxis._axinfo['juggled'] = (1, 2, 0)

    # Set view angle
    ax.view_init(*view_angle)

    # Add grid lines to make axes more visible
    ax.grid(True)

    # Apply tight layout and show plot
    fig.tight_layout()
    plt.show()

def to_rgb(mask, rgb_value:list = [255,255,255], background_value:list = [0,0,0]):
    height, width = mask.shape
    array_3d = np.zeros((height, width, 3), dtype=np.uint8)
    array_3d[mask != 0] = rgb_value
    array_3d[mask == 0] = background_value

    return array_3d


def extract_information(filename):

    '''
    Extracts info from the .nd2 filename 
    '''

    file_base = filename.rsplit('.', 1)[0]
    base_name = file_base.split('_')[0]
    
    DJID = base_name[:4]
    eye = base_name[4]

    return DJID, eye, file_base


def plot_single_cell(single_cell):

    '''
    Takes dataframe containing traces of a single cell

    Example:
    dataframe.query('cell_unid == 90')

    '''
    single_cell['Stain'] = single_cell['Stain'].apply(lambda x: 'GluT1' if x == 'GluT1' else x)
    
    stain_colors = {'WGA': 'grey', 'eGFP': 'green', 'GluT1': 'magenta', 'DAPI': 'blue'}

    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.bottom"] = True

    plt.figure(figsize=(22, 15))

    for index, row in single_cell.iterrows():
        stain_color = stain_colors.get(row['Stain'], 'gray')
        plt.plot(row['X_vals'], row['Y_vals'], linestyle='-', label=row['Stain'], color=stain_color, linewidth=10)
        
        # Peak picking for each line
        peaks, _ = find_peaks(row['Y_vals'], prominence=15, distance=20)
        x_peak = np.array(row['X_vals'])[peaks]
        y_peak = np.array(row['Y_vals'])[peaks]
        plt.plot(x_peak, y_peak, 'x', color=stain_color, linewidth=2)

    plt.xlabel('Depth (micron)', fontsize=60)
    plt.ylabel('Y Axis', fontsize=60)

    plt.yticks(fontsize=60)
    plt.xticks(fontsize=60)
    plt.xlabel('Depth (µm)', fontsize=60)
    plt.ylabel('Fluorescence (a.u)', fontsize=60)
    plt.title('All Stains of a Single Cell', fontsize = 65)
    plt.legend(fontsize = 45)

    ax = plt.gca()
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)

    plt.tight_layout()
    plt.show()
    