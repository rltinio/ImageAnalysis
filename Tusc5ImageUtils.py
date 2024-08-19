import numpy as np
from skimage import exposure, measure
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import center_of_mass
from scipy.spatial import ConvexHull, distance_matrix
from scipy.signal import find_peaks
from skimage.measure import label, regionprops
import pandas as pd
from cellpose import utils, io, plot, models, denoise
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import re


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

    return new_WGA_slice, comzi

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

    sq_stacks = np.stack((sq_DAPI_stack, sq_eGFP_stack, sq_WGA_stack, sq_GLUT1_stack))

    return sq_stacks

def find_center_2d(mask_array):
    """
    This function finds the center of the mask in a given 2D array.
    The center is calculated as the mean of the x and y coordinates where the mask value is greater than 0.
    """
    indices = np.argwhere(mask_array > 0)
    if len(indices) == 0:
        return None
    center_x = np.mean(indices[:, 0])
    center_y = np.mean(indices[:, 1])
    return (center_x, center_y)

def closest_mask_2d(array_one, array_two):
    """
    This function finds the binary mask in array_two whose center is closest to the center of the mask in array_one.
    """
    # Calculate the center of the mask in array_one
    center_one = find_center_2d(array_one)
    if center_one is None:
        return None
    
    min_distance = float('inf')
    closest_mask_value = None

    # Get unique masks in array_two
    unique_masks = np.unique(array_two[array_two > 0])

    # Calculate the center of each mask in array_two
    for mask_value in unique_masks:
        mask_array = np.where(array_two == mask_value, 1, 0)
        center_two = find_center_2d(mask_array)
        if center_two is None:
            continue
        # Calculate the Euclidean distance between centers
        distance = np.sqrt((center_one[0] - center_two[0])**2 + (center_one[1] - center_two[1])**2)
        if distance < min_distance:
            min_distance = distance
            closest_mask_value = mask_value

    # Create a binary mask for the closest mask value
    binary_mask = np.where(array_two == closest_mask_value, 1, 0)

    return binary_mask

def get_traces(sq_stacks, single_mask):
    
    masked_stacks = sq_stacks * single_mask
    summed_values = np.sum(masked_stacks, axis=(-1, -2))  # Sum along the spatial dimensions

    # Store the results with corresponding channel names
    channel_names = ['DAPI', 'eGFP', 'WGA', 'GluT1']
    result = {name: value for name, value in zip(channel_names, summed_values)}

    return result

def list_squares(image, masks):

    array_list = []

    for mask_id in (np.delete(np.unique(masks), 0) - 1):
        extracted_mask = extract_masks(masks, mask_id)
        sq_WGA_slice = extract_square_proj_expand(image, extracted_mask, extra_pixels=0)

        array_list.append(sq_WGA_slice)

    return array_list

def fit_images_in_square(images, pad_value=5):
    """
    Arrange a list of 2D arrays (images) into a grid to fit approximately in a square shape.

    Parameters:
    images (list of np.ndarray): List of 2D arrays to arrange.
    pad_value (int, optional): Value to use for padding. Default is 0.

    Returns:
    np.ndarray: A single 2D array with all input arrays arranged in a grid.
    """
    # Calculate number of images and the size of the grid
    num_images = len(images)
    grid_size = int(np.ceil(np.sqrt(num_images)))

    # Determine the maximum height and width of the images
    max_height = max(image.shape[0] for image in images)
    max_width = max(image.shape[1] for image in images)

    # Create a padded image grid
    grid = []
    for i in range(grid_size):
        row_images = []
        for j in range(grid_size):
            index = i * grid_size + j
            if index < num_images:
                image = images[index]
                # Pad image to the maximum dimensions
                padded_image = np.pad(image, ((0, max_height - image.shape[0]), (0, max_width - image.shape[1])), mode='constant', constant_values=pad_value)
                row_images.append(padded_image)
            else:
                # If there are no more images, fill with padding
                row_images.append(np.full((max_height, max_width), pad_value))
        # Concatenate images in the row
        grid.append(np.hstack(row_images))
    
    # Concatenate all rows to form the final grid
    stitched_image = np.vstack(grid)

    return stitched_image

def add_scale_bar(image, microns_per_pixel, scale_bar_length_microns, x_position, y_position, bar_height=5, bar_color=(255, 255, 255)):

    # Calculate the length of the scale bar in pixels
    scale_bar_pixel_length = int(scale_bar_length_microns / microns_per_pixel)

    # Copy the image to avoid modifying the original
    image_with_bar = image.copy()

    # Draw the scale bar
    cv2.rectangle(image_with_bar, (x_position, y_position), 
                  (x_position + scale_bar_pixel_length, y_position + bar_height), 
                  bar_color, -1)

    return image_with_bar

def grey_to_color(image_slice, color:str = 'white'):

    # Create an empty image with the same dimensions but with 3 channels
    slice_rgb = np.zeros((image_slice.shape[0], image_slice.shape[1], 3), dtype=np.uint8)

    if color == 'magenta':
        slice_rgb[:, :, 0] = image_slice
        slice_rgb[:, :, 2] = image_slice
    
    if color == 'white':
        slice_rgb[:, :, 1] = image_slice
        slice_rgb[:, :, 0] = image_slice
        slice_rgb[:, :, 2] = image_slice

    if color == 'blue':
        slice_rgb[:, :, 2] = image_slice

    if color == 'green':
        slice_rgb[:, :, 1] = image_slice

    if color == 'red':
        slice_rgb[:, :, 0] = image_slice

    return slice_rgb

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

def normalize(numbers):
    min_val = min(numbers)
    max_val = max(numbers)
    if max_val == min_val:
        # If all numbers are the same, return a list of 0.5s
        return [0.5] * len(numbers)
    return np.array([(x - min_val) / (max_val - min_val) for x in numbers])

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
    file_base = filename.rsplit('.', 1)[0]  # Remove the file extension
    base_name = file_base.split('_')[0]     # Get the first part of the filename before the first underscore

    DJID = base_name[:4]                    # Extract the DJID (first 4 characters)
    
    # Extract 'R', 'R0', 'R1', 'L', 'L0', 'L1', 'RA', 'RB', etc. before the first underscore
    match = re.search(r'([RL][0-1A-D]?)', base_name[4:])
    if match:
        eye = match.group(1)
    else:
        eye = base_name[4]  # Fallback if the pattern is not present

    return DJID, eye, file_base


def plot_single_cell(single_cell, prominence=15, distance=20, threshold = None):
    stain_colors = {'WGA': 'grey', 'eGFP': 'green', 'GluT1': 'magenta', 'DAPI': 'blue'}

    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.bottom"] = True

    plt.figure(figsize=(22, 15))

    for index, row in single_cell.iterrows():
        stain_color = stain_colors.get(row['Stain'], 'gray')
        plt.plot(row['X_vals'], row['Y_vals'], linestyle='-', label=row['Stain'], color=stain_color, linewidth=10)
        
        # Peak picking for each line using the provided prominence and distance
        peaks, _ = find_peaks(row['Y_vals'], prominence=prominence, distance=distance, threshold = None)
        print(f"Stain: {row['Stain']}, Detected peaks at indices: {peaks}")
        
        if len(peaks) > 0:
            x_peak = np.array(row['X_vals'])[peaks]
            y_peak = np.array(row['Y_vals'])[peaks]
            plt.plot(x_peak, y_peak, 'o', color='red', markersize=10)

    plt.xlabel('Depth (µm)', fontsize=60)
    plt.ylabel('Fluorescence (a.u)', fontsize=60)
    plt.title('All Stains of a Single Cell', fontsize=65)
    plt.legend(fontsize=45)

    ax = plt.gca()
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)

    plt.tight_layout()
    plt.show()

###
import cv2

def annotate_image(image, centers_of_masses, text_color=(255, 255, 255), font_scale=.75, thickness=1):
    # Make a copy of the image to annotate
    annotated_image = image.copy()
    
    # Convert text_color from RGB to BGR since OpenCV uses BGR
    text_color = (text_color[2], text_color[1], text_color[0])
    
    for idx, coords in enumerate(centers_of_masses):
        # Draw the text on the image
        cv2.putText(annotated_image, str(idx), (int(coords[1]), int(coords[0])), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)
    
    return annotated_image

#ttkinter gui
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

class NumberInputApp:
    def __init__(self, root, overlay_image):
        self.root = root
        self.root.title("Fast Number Input")
        
        self.input_var = tk.StringVar()
        self.input_list = []

        self.overlay_image = overlay_image
        self.zoom_factor = 1.0  # Initialize zoom factor
        self.offset_x = 0
        self.offset_y = 0
        self.drag_start_x = 0
        self.drag_start_y = 0

        self.create_widgets()
        self.bind_keys()
        self.bind_zoom()
        self.bind_drag()

    def create_widgets(self):
        # Create an Entry widget to display the input
        self.entry = tk.Entry(self.root, textvariable=self.input_var, font=('Helvetica', 24), justify='center')
        self.entry.grid(row=0, column=0, columnspan=3)

        # Create number buttons
        self.buttons = []
        for i in range(1, 9):
            button = tk.Button(self.root, text=str(i), font=('Helvetica', 24), command=lambda i=i: self.on_button_click(i))
            self.buttons.append(button)

        # Arrange buttons in a grid
        for i in range(1, 9):
            self.buttons[i-1].grid(row=(i-1)//3 + 1, column=(i-1)%3)

        # Create Clear and Submit buttons
        self.clear_button = tk.Button(self.root, text="Clear", font=('Helvetica', 24), command=self.clear_input)
        self.clear_button.grid(row=4, column=0)

        self.submit_button = tk.Button(self.root, text="Submit", font=('Helvetica', 24), command=self.submit_input)
        self.submit_button.grid(row=4, column=2)

        # Create a Label to display the image
        self.image_label = tk.Label(self.root)
        self.image_label.grid(row=5, column=0, columnspan=3)
        self.display_image()

    def bind_keys(self):
        for i in range(1, 9):
            self.root.bind(str(i), self.on_key_press)
        self.root.bind('<Return>', self.submit_input)
        self.root.bind('<BackSpace>', self.on_backspace_press)

    def bind_zoom(self):
        self.root.bind("<MouseWheel>", self.on_mouse_wheel)

    def bind_drag(self):
        self.image_label.bind("<ButtonPress-1>", self.on_button_press)
        self.image_label.bind("<B1-Motion>", self.on_mouse_drag)
        self.image_label.bind("<ButtonRelease-1>", self.on_button_release)

    def on_button_click(self, number):
        current_value = self.input_var.get()
        self.input_var.set(current_value + str(number))

    def on_key_press(self, event):
        current_value = self.input_var.get()
        self.input_var.set(current_value + event.char)

    def on_backspace_press(self, event):
        current_value = self.input_var.get()
        self.input_var.set(current_value[:-1])

    def clear_input(self):
        self.input_var.set("")

    def submit_input(self, event=None):
        input_value = self.input_var.get()
        if input_value:
            self.input_list.append(int(input_value))
            #print("User input:", input_value)
            self.clear_input()

    def display_image(self):
        image = Image.fromarray(self.overlay_image)
        new_size = (int(500 * self.zoom_factor), int(500 * self.zoom_factor))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)

        # Adjust the image position based on the offset
        self.image_label.config(image=photo)
        self.image_label.image = photo
        self.image_label.place(x=self.offset_x, y=self.offset_y)

    def on_mouse_wheel(self, event):
        if event.delta > 0:
            self.zoom_factor *= 1.1  # Zoom in
        elif event.delta < 0:
            self.zoom_factor *= 0.9  # Zoom out
        self.display_image()

    def on_button_press(self, event):
        self.drag_start_x = event.x
        self.drag_start_y = event.y

    def on_mouse_drag(self, event):
        delta_x = event.x - self.drag_start_x
        delta_y = event.y - self.drag_start_y
        self.offset_x += delta_x
        self.offset_y += delta_y
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        self.display_image()

    def on_button_release(self, event):
        pass

def rip_identifier(nd2_file, image, dapi_masks, coords_2d):

    file_base = nd2_file.rsplit('.', 1)[0]
    def run_app(overlay_image):
        root = tk.Tk()
        app = NumberInputApp(root, overlay_image)
        root.mainloop()
        return app.input_list
    
    WGA_stack = to_8bit(image[:,2,:,:].copy())

    # Plots ripped cells
    z_intensities = np.sum(WGA_stack, axis=(1, 2))
    ILM_layer = int(np.argmax(z_intensities))

    

    overlay = plot.mask_overlay(WGA_stack[ILM_layer, :, :], dapi_masks)
    overlay_image = annotate_image(overlay, coords_2d, text_color=(255, 255, 255), font_scale=.5, thickness=1)

    # Run the Tkinter input window 
    cells_in_rip = {file_base: run_app(overlay_image)}

    return cells_in_rip

class NumberInputApp:
    def __init__(self, root, overlay_image):
        self.root = root
        self.root.title("Fast Number Input")
        
        self.input_var = tk.StringVar()
        self.input_list = []

        self.overlay_image = overlay_image
        self.zoom_factor = 1.0  # Initialize zoom factor
        self.offset_x = 0
        self.offset_y = 0
        self.drag_start_x = 0
        self.drag_start_y = 0

        self.create_widgets()
        self.bind_keys()
        self.bind_zoom()
        self.bind_drag()

    def create_widgets(self):
        # Create an Entry widget to display the input
        self.entry = tk.Entry(self.root, textvariable=self.input_var, font=('Helvetica', 24), justify='center')
        self.entry.grid(row=0, column=0, columnspan=3)

        # Create number buttons
        self.buttons = []
        for i in range(1, 10):
            button = tk.Button(self.root, text=str(i), font=('Helvetica', 24), command=lambda i=i: self.on_button_click(i))
            self.buttons.append(button)
        
        # Add button for 0
        zero_button = tk.Button(self.root, text="0", font=('Helvetica', 24), command=lambda: self.on_button_click(0))
        self.buttons.append(zero_button)

        # Arrange buttons in a grid (3x4 grid for 0-9)
        for i in range(1, 10):
            self.buttons[i-1].grid(row=(i-1)//3 + 1, column=(i-1)%3)
        self.buttons[9].grid(row=4, column=1)  # Place 0 in the center of the last row

        # Create Clear and Submit buttons
        self.clear_button = tk.Button(self.root, text="Clear", font=('Helvetica', 24), command=self.clear_input)
        self.clear_button.grid(row=5, column=0)

        self.submit_button = tk.Button(self.root, text="Submit", font=('Helvetica', 24), command=self.submit_input)
        self.submit_button.grid(row=5, column=2)

        # Create a Label to display the image
        self.image_label = tk.Label(self.root)
        self.image_label.grid(row=6, column=0, columnspan=3)
        self.display_image()

    def bind_keys(self):
        for i in range(0, 10):
            self.root.bind(str(i), self.on_key_press)
        self.root.bind('<Return>', self.submit_input)
        self.root.bind('<BackSpace>', self.on_backspace_press)

    def bind_zoom(self):
        self.root.bind("<MouseWheel>", self.on_mouse_wheel)

    def bind_drag(self):
        self.image_label.bind("<ButtonPress-1>", self.on_button_press)
        self.image_label.bind("<B1-Motion>", self.on_mouse_drag)
        self.image_label.bind("<ButtonRelease-1>", self.on_button_release)

    def on_button_click(self, number):
        current_value = self.input_var.get()
        self.input_var.set(current_value + str(number))

    def on_key_press(self, event):
        current_value = self.input_var.get()
        self.input_var.set(current_value + event.char)

    def on_backspace_press(self, event):
        current_value = self.input_var.get()
        self.input_var.set(current_value[:-1])

    def clear_input(self):
        self.input_var.set("")

    def submit_input(self, event=None):
        input_value = self.input_var.get()
        if input_value:
            self.input_list.append(int(input_value))
            #print("User input:", input_value)
            self.clear_input()

    def display_image(self):
        image = Image.fromarray(self.overlay_image)
        new_size = (int(500 * self.zoom_factor), int(500 * self.zoom_factor))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)

        # Adjust the image position based on the offset
        self.image_label.config(image=photo)
        self.image_label.image = photo
        self.image_label.place(x=self.offset_x, y=self.offset_y)

    def on_mouse_wheel(self, event):
        if event.delta > 0:
            self.zoom_factor *= 1.1  # Zoom in
        elif event.delta < 0:
            self.zoom_factor *= 0.9  # Zoom out
        self.display_image()

    def on_button_press(self, event):
        self.drag_start_x = event.x
        self.drag_start_y = event.y

    def on_mouse_drag(self, event):
        delta_x = event.x - self.drag_start_x
        delta_y = event.y - self.drag_start_y
        self.offset_x += delta_x
        self.offset_y += delta_y
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        self.display_image()

    def on_button_release(self, event):
        pass

def rip_identifier(nd2_file, image, dapi_masks, coords_2d):

    file_base = nd2_file.rsplit('.', 1)[0]
    def run_app(overlay_image):
        root = tk.Tk()
        app = NumberInputApp(root, overlay_image)
        root.mainloop()
        return app.input_list
    
    WGA_stack = to_8bit(image[:,2,:,:].copy())

    # Plots ripped cells
    z_intensities = np.sum(WGA_stack, axis=(1, 2))
    ILM_layer = int(np.argmax(z_intensities))

    

    overlay = plot.mask_overlay(WGA_stack[ILM_layer, :, :], dapi_masks)
    overlay_image = annotate_image(overlay, coords_2d, text_color=(255, 255, 255), font_scale=.5, thickness=1)

    # Run the Tkinter input window 
    cells_in_rip = {file_base: run_app(overlay_image)}

    return cells_in_rip

import tkinter as tk
from tkinter.ttk import *
from PIL import Image, ImageTk, ImageEnhance
import numpy as np

def run_max_projector_app(dapi_stack):
    class RangeSliderH(Frame):
        def __init__(self, parent, vars, max_slice, padX=12, **kwargs):
            super().__init__(parent, **kwargs)
            self.vars = vars
            self.max_slice = max_slice
            self.padX = padX
            self.canvas = tk.Canvas(self, width=300, height=50)
            self.canvas.pack()

            self.update_slider()

            self.canvas.bind('<B1-Motion>', self.move_slider)
            self.canvas.bind('<ButtonRelease-1>', self.update_image)

        def update_slider(self):
            self.canvas.delete("all")
            width = self.canvas.winfo_width()
            handle_radius = 15  # Increase handle size for easier dragging

            x1 = self.padX + (width - 2 * self.padX) * self.vars[0].get()
            x2 = self.padX + (width - 2 * self.padX) * self.vars[1].get()

            self.canvas.create_line(x1, 25, x2, 25, fill="gray", width=6)  # Thicker line for better visibility
            self.canvas.create_oval(x1 - handle_radius, 25 - handle_radius, x1 + handle_radius, 25 + handle_radius, fill="blue", outline="black")
            self.canvas.create_oval(x2 - handle_radius, 25 - handle_radius, x2 + handle_radius, 25 + handle_radius, fill="blue", outline="black")

        def move_slider(self, event):
            width = self.canvas.winfo_width()
            x = event.x

            closest_handle = None
            min_distance = float('inf')

            # Determine which handle is closer to the mouse click
            for i, var in enumerate(self.vars):
                handle_x = self.padX + (width - 2 * self.padX) * var.get()
                distance = abs(x - handle_x)
                if distance < min_distance:
                    min_distance = distance
                    closest_handle = i

            # Move the closest handle
            if closest_handle is not None:
                new_val = (x - self.padX) / (width - 2 * self.padX)
                if closest_handle == 0:
                    self.vars[0].set(max(0, min(new_val, self.vars[1].get())))
                else:
                    self.vars[1].set(max(self.vars[0].get(), min(new_val, 1)))

            self.update_slider()
            self.update_image()

        def get_values(self):
            return int(self.vars[0].get() * self.max_slice), int(self.vars[1].get() * self.max_slice)

        def update_image(self, event=None):
            z1, z2 = self.get_values()
            app.update_image(z1, z2)
            app.update_entry_values(z1, z2)

    class MaxProjectorApp:
        def __init__(self, root, stack):
            self.root = root
            self.root.title("Z-Slice Image Viewer")

            self.stack = stack
            self.min_slice = 0
            self.max_slice = stack.shape[0] - 1

            self.image_width = 700
            self.image_height = 700

            self.z0 = None
            self.z1 = None

            self.create_widgets()
            self.update_image(self.min_slice, self.max_slice)

        def create_widgets(self):
            slider_frame = Frame(self.root)
            slider_frame.pack(pady=10)

            self.hLeft = tk.DoubleVar(value=self.min_slice / self.max_slice)
            self.hRight = tk.DoubleVar(value=self.max_slice / self.max_slice)
            self.range_slider = RangeSliderH(slider_frame, [self.hLeft, self.hRight], max_slice=self.max_slice)
            self.range_slider.pack(fill='x', expand=True)

            self.slice_text = Label(slider_frame, text=f"Slice range: {self.min_slice} to {self.max_slice}")
            self.slice_text.pack(pady=5)

            self.brighten_var = tk.IntVar()
            self.brighten_check = Checkbutton(slider_frame, text="Brighten", variable=self.brighten_var, command=self.on_brighten_toggle)
            self.brighten_check.pack(side='left', pady=5)

            self.z1_entry = Entry(slider_frame, width=5)
            self.z1_entry.pack(side='left', padx=5)
            self.z1_entry.bind("<Return>", self.on_entry_update)

            self.z2_entry = Entry(slider_frame, width=5)
            self.z2_entry.pack(side='left', padx=5)
            self.z2_entry.bind("<Return>", self.on_entry_update)

            self.image_label = Label(self.root)
            self.image_label.pack(pady=10)

            exit_button = Button(self.root, text="Exit", command=self.on_exit)
            exit_button.pack(pady=10)

        def on_brighten_toggle(self):
            z1, z2 = self.range_slider.get_values()
            self.update_image(z1, z2)

        def on_entry_update(self, event=None):
            try:
                z1 = int(self.z1_entry.get())
                z2 = int(self.z2_entry.get())
                if z1 >= self.min_slice and z2 <= self.max_slice and z1 < z2:
                    self.hLeft.set(z1 / self.max_slice)
                    self.hRight.set(z2 / self.max_slice)
                    self.update_image(z1, z2)
                else:
                    raise ValueError
            except ValueError:
                print("Invalid input for slices")

        def update_entry_values(self, z1, z2):
            self.z1_entry.delete(0, tk.END)
            self.z1_entry.insert(0, str(z1))
            self.z2_entry.delete(0, tk.END)
            self.z2_entry.insert(0, str(z2))

        def update_image(self, z1, z2):
            self.slice_text.config(text=f"Slice range: {z1} to {z2}")

            slice_mp = max_proj(self.stack[z1:z2])

            if self.brighten_var.get() == 1:
                slice_mp = self.apply_brightness_contrast(slice_mp)

            image = Image.fromarray(slice_mp)

            image_resized = image.resize((self.image_width, self.image_height))
            photo = ImageTk.PhotoImage(image_resized)

            self.image_label.config(image=photo)
            self.image_label.image = photo

            self.z0 = z1
            self.z1 = z2

        def apply_brightness_contrast(self, image_array):
            image = Image.fromarray(image_array)
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.2)
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)
            return np.array(image)

        def export_slices(self):
            z0 = self.z0
            z1 = self.z1

            return z0, z1

        def on_exit(self):
            self.export_slices()
            self.root.destroy()

    root = tk.Tk()
    app = MaxProjectorApp(root, dapi_stack)
    root.mainloop()

    z0, z1 = app.export_slices()
    return z0, z1