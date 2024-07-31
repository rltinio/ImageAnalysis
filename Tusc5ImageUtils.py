import numpy as np
from skimage import exposure, measure
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import center_of_mass
from scipy.spatial import ConvexHull, distance_matrix

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
    total_masks = np.delete(np.unique(masks,0),0)

    for mask in total_masks:
        single_mask = masks == mask
        single_slice = nuclei_z_indicies[mask-1]
        edited_image[single_mask] = single_channel[single_slice][single_mask]

    return base_image_display, edited_image

def plot_before_after(before_image, after_image, auto_bc = False):

    if auto_bc == True:
        before_image = auto_brightness_contrast(before_image)
        after_image = auto_brightness_contrast(after_image)

    fig, axs = plt.subplots(1,2)

    axs[0].imshow(before_image)
    axs[0].axis('off')
    axs[0].set_title('Base Image')

    axs[1].imshow(after_image)
    axs[1].axis('off')
    axs[1].set_title('Edited Image')

# def to_8bit(image, do_scaling=True):
#     if image.dtype != np.uint16:
#         raise ValueError("Input image must be of dtype np.uint16")
    
#     output_image = np.zeros(image.shape, dtype=np.uint8)
    
#     if do_scaling:
#         min_val = image.min()
#         max_val = image.max()
#         scale = 256.0 / (max_val - min_val + 1)
#         output_image = ((image - min_val) * scale).clip(0, 255).astype(np.uint8)
#     else:
#         output_image = image.clip(0, 255).astype(np.uint8)
    
#     return output_image

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

# def cell_projector(single_channel, masks, nuclei_z_indicies):
    
#     max_average = int(np.sum(total_nuclei_slices)/len(total_nuclei_slices))
#     edited_image = single_channel[max_average].copy()
#     base_image_display = single_channel[max_average].copy()
#     total_masks = np.delete(np.unique(masks,0),0)

#     for mask in total_masks:
#         single_mask = masks == mask
#         single_slice = nuclei_z_indicies[mask-1]
#         edited_image[single_mask] = single_channel[single_slice][single_mask]

#     return base_image_display, edited_image


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

    sq_stack = ~auto_brightness_contrast(max_proj(~sq_stack))

    return sq_stack, sq_maski

def WGA_stitcher(DAPI_stack, WGA_stack, masks, z_sep):
    z_avg = int(np.mean(nuclei_slices(DAPI_stack, masks)))
    base_stack = auto_brightness_contrast(WGA_stack[z_avg])

    total_masks = np.delete(np.unique(masks), 0)

    for mask_num in total_masks:
        maski = masks == mask_num
        sq_stacki, sq_maski = WGA_projector(DAPI_stack, WGA_stack, maski, z_sep)
        if sq_stacki is not None:
            base_stack[sq_maski] = sq_stacki[sq_maski]

    plt.imshow(base_stack)
    plt.axis('off')
    return base_stack

def nucleus_com(single_channel, masks):
    total_masks = np.delete(np.unique(masks), 0)

    mask = total_masks[0]
    single_mask = masks == mask
    masked_channel = single_channel * single_mask

    z_prof = np.sum(masked_channel, axis=(1, 2))
    z_max_idx = np.argmax(z_prof)

    com = center_of_mass(single_mask)

    com_3d = (int(com[0]), int(com[1]), z_max_idx)
    return com_3d

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
        center_of_mass = center_of_mass(single_mask)
        
        # The z coordinate of the center of mass is determined by the slice with the max intensity
        center_of_mass_3d = (int(center_of_mass[0]), int(center_of_mass[1]), z_max_idx)

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
    Remove numbers from the list that have a standard deviation greater than 1 from the mean.

    :param data: List of numbers.
    :return: List of numbers with outliers removed.
    """
    mean = np.mean(data)
    std_dev = np.std(data)

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
    std_dev = np.std(Z_idxs)

    # Define the lower and upper bounds
    lower_bound = mean - std_dev
    upper_bound = mean + std_dev

    # Filter the data to remove outliers
    filtered_data = [x for x in data if lower_bound <= x[2] <= upper_bound]
    
    return filtered_data

def plot_centers_of_mass(centers_of_mass):
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

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.set_zlim(0,108)
    ax.view_init(25, 45)

    plt.show()
