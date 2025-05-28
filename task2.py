import os
import pydicom
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, zoom, find_objects, center_of_mass, binary_erosion, binary_dilation, binary_fill_holes
from skimage.segmentation import flood_fill # Using skimage for a more robust flood fill
import imageio

# Ensure output directories exist at the beginning
Path("results/task2").mkdir(parents=True, exist_ok=True) # Changed to task2
Path("frames_task2").mkdir(parents=True, exist_ok=True) # Changed to task2


def normalize(input_array):
    amin = np.amin(input_array)
    amax = np.amax(input_array)
    if amax == amin:
        return np.zeros_like(input_array)
    return (input_array - amin) / (amax - amin)


def load_image_data(image_path):
    img_dcmset = []
    for root, _, filenames in os.walk(image_path):
        for filename in sorted(filenames): # Sort filenames for consistent order
            dcm_path = Path(root) / filename
            try:
                dicom = pydicom.dcmread(dcm_path, force=True)
                if hasattr(dicom, 'PixelData'):
                    img_dcmset.append(dicom)
            except Exception as e:
                print(f"Warning: Could not read or process DICOM file {dcm_path}: {e}")
    return img_dcmset


def process_image_data(img_dcmset):
    if not img_dcmset:
        raise ValueError("Input DICOM set is empty. Check image_path or DICOM contents.")

    img_dcmset_filtered = []
    for dcm in img_dcmset:
        required_attrs = ['SliceThickness', 'PixelSpacing', 'ImagePositionPatient', 'pixel_array']
        # AcquisitionNumber is not always present or consistently used
        if all(hasattr(dcm, attr) for attr in required_attrs):
            img_dcmset_filtered.append(dcm)
        else:
            missing = [attr for attr in required_attrs if not hasattr(dcm, attr)]
            # print(f"Warning: DICOM {dcm.filename if hasattr(dcm, 'filename') else 'N/A'} missing attributes: {missing}")
            pass


    if not img_dcmset_filtered:
        raise ValueError("No valid DICOMs with required attributes found in the set.")

    # Sort by ImagePositionPatient[2] (Z-coordinate)
    img_dcmset_filtered.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    slice_thickness = float(img_dcmset_filtered[0].SliceThickness)
    full_pixel_spacing = [float(ps) for ps in img_dcmset_filtered[0].PixelSpacing]

    # Apply RescaleSlope and RescaleIntercept if available for HU conversion
    pixel_arrays = []
    for dcm in img_dcmset_filtered:
        arr = dcm.pixel_array.astype(np.float32)
        if hasattr(dcm, 'RescaleSlope') and hasattr(dcm, 'RescaleIntercept'):
            slope = float(dcm.RescaleSlope)
            intercept = float(dcm.RescaleIntercept)
            arr = arr * slope + intercept
        pixel_arrays.append(arr)
    
    img_pixelarray = np.stack(pixel_arrays, axis=0)

    return img_dcmset_filtered, img_pixelarray, slice_thickness, full_pixel_spacing


def load_segmentation_data(seg_path):
    mask_dcm = pydicom.dcmread(seg_path, force=True)
    if not hasattr(mask_dcm, 'pixel_array'):
        raise AttributeError(f"Segmentation file {seg_path} does not have pixel_array attribute.")
    seg_data = mask_dcm.pixel_array
    return seg_data, mask_dcm


def create_segmentation_masks(seg_dcm_obj, ct_dcm_list, target_shape_zyx, ignore_sop_mapping=False):
    seg_data_raw = seg_dcm_obj.pixel_array
    seg_3d_aligned = np.zeros(target_shape_zyx, dtype=np.int32)

    if ignore_sop_mapping or not hasattr(seg_dcm_obj, 'PerFrameFunctionalGroupsSequence'):
        # For binary masks, check if seg_data_raw has values > 0
        # Sometimes the mask values are not 1, but other positive integers
        segment_map = {}
        if hasattr(seg_dcm_obj, 'SegmentSequence'):
            for item in seg_dcm_obj.SegmentSequence:
                segment_map[item.SegmentNumber] = item.SegmentNumber

        num_seg_frames = seg_data_raw.shape[0]
        num_ct_slices = target_shape_zyx[0]
        
        if num_seg_frames == num_ct_slices:
            seg_label_to_assign = 1
            if segment_map and len(segment_map) == 1:
                seg_label_to_assign = list(segment_map.keys())[0]

            for z in range(num_ct_slices):
                # Fix: Use > 0 instead of == 1 to catch all positive values
                mask_slice = seg_data_raw[z] > 0
                seg_3d_aligned[z][mask_slice] = seg_label_to_assign
                
        else:
            print(f"Warning: Frame/slice mismatch. Trying alternative mapping...")
            # Try to map available frames proportionally
            for i in range(min(num_seg_frames, num_ct_slices)):
                mask_slice = seg_data_raw[i] > 0
                if np.any(mask_slice):
                    seg_3d_aligned[i][mask_slice] = 1
    else:
        # Original SOP mapping logic remains the same
        ref_segs = seg_dcm_obj.PerFrameFunctionalGroupsSequence
        valid_masks = {}
        for i, frame_group in enumerate(ref_segs):
            try:
                ref_ct_uid = frame_group.DerivationImageSequence[0].SourceImageSequence[0].ReferencedSOPInstanceUID
                seg_number = frame_group.SegmentIdentificationSequence[0].ReferencedSegmentNumber
                valid_masks.setdefault(ref_ct_uid, {})[seg_number] = i
            except (AttributeError, IndexError):
                continue

        for z_idx, ct_slice_dcm in enumerate(ct_dcm_list):
            if ct_slice_dcm.SOPInstanceUID in valid_masks:
                for seg_number, raw_frame_idx in valid_masks[ct_slice_dcm.SOPInstanceUID].items():
                    if raw_frame_idx < seg_data_raw.shape[0]:
                        seg_slice_pixels = seg_data_raw[raw_frame_idx]
                        # Fix: Use > 0 instead of == 1
                        seg_3d_aligned[z_idx][seg_slice_pixels > 0] = int(seg_number)
    
    return seg_3d_aligned

# --- New functions for Task 2 ---
def get_mask_bbox_centroid(mask_3d):
    """
    Extracts the bounding box and centroid from a 3D binary mask.
    Assumes mask_3d is a binary array (True/1 where mask is present).
    """
    if not np.any(mask_3d): # Check if the mask contains any True values
        return None, None

    # Find objects returns a list of slice objects
    loc = find_objects(mask_3d)
    if not loc:
        return None, None
    
    # Get the slice for the first (and presumably only) object
    bbox_slices = loc[0] # (slice_z, slice_y, slice_x)
    
    # Convert slice objects to (min, max) coordinates for each dimension
    # bbox = [(s.start, s.stop) for s in bbox_slices] # z_min, z_max, y_min, y_max, x_min, x_max
    bbox_coords = {
        "z_min": bbox_slices[0].start, "z_max": bbox_slices[0].stop,
        "y_min": bbox_slices[1].start, "y_max": bbox_slices[1].stop,
        "x_min": bbox_slices[2].start, "x_max": bbox_slices[2].stop,
    }

    # Calculate centroid
    # Centroid is returned in (z, y, x) order
    centroid_zyx = center_of_mass(mask_3d)
    
    return bbox_coords, centroid_zyx

def segment_tumor_centroid_region_growing(ct_image_3d, centroid_zyx, tolerance):
    if centroid_zyx is None:
        return np.zeros_like(ct_image_3d, dtype=bool)

    z, y, x = int(round(centroid_zyx[0])), int(round(centroid_zyx[1])), int(round(centroid_zyx[2]))

    if not (0 <= z < ct_image_3d.shape[0] and 
            0 <= y < ct_image_3d.shape[1] and 
            0 <= x < ct_image_3d.shape[2]):
        return np.zeros_like(ct_image_3d, dtype=bool)

    try:
        # Use a more conservative approach with multiple seed points around centroid
        seed_points = [
            (z, y, x),
            (max(0, z-1), y, x),
            (min(ct_image_3d.shape[0]-1, z+1), y, x),
            (z, max(0, y-1), x),
            (z, min(ct_image_3d.shape[1]-1, y+1), x),
            (z, y, max(0, x-1)),
            (z, y, min(ct_image_3d.shape[2]-1, x+1))
        ]
        
        combined_mask = np.zeros_like(ct_image_3d, dtype=bool)
        
        for seed_z, seed_y, seed_x in seed_points:
            try:
                temp_mask = flood_fill(ct_image_3d, (seed_z, seed_y, seed_x), 
                                     new_value=-9999, tolerance=tolerance, connectivity=1)
                temp_binary = (temp_mask == -9999)
                combined_mask = np.logical_or(combined_mask, temp_binary)
            except:
                continue
        
        # Post-processing: morphological operations
        from scipy.ndimage import binary_opening, binary_closing, label
        structure = np.ones((3, 3, 3))
        
        # Clean up the mask
        combined_mask = binary_opening(combined_mask, structure=structure, iterations=1)
        combined_mask = binary_closing(combined_mask, structure=structure, iterations=1)
        
        # Keep largest connected component
        labeled_array, num_features = label(combined_mask)
        if num_features > 1:
            component_sizes = np.bincount(labeled_array.ravel())[1:]
            largest_component = np.argmax(component_sizes) + 1
            combined_mask = (labeled_array == largest_component)
            
        return combined_mask

    except Exception as e:
        print(f"Error during region growing: {e}")
        return np.zeros_like(ct_image_3d, dtype=bool)

def dice_coefficient(true_mask, pred_mask):
    """Computes the Dice coefficient."""
    true_mask = np.asarray(true_mask).astype(bool)
    pred_mask = np.asarray(pred_mask).astype(bool)
    intersection = np.logical_and(true_mask, pred_mask)
    if true_mask.sum() + pred_mask.sum() == 0: # Both masks are empty
        return 1.0 # Or 0.0, depending on convention for empty sets
    return (2. * intersection.sum()) / (true_mask.sum() + pred_mask.sum() + 1e-6) # Add epsilon for stability

def jaccard_index(true_mask, pred_mask):
    """Computes the Jaccard index (Intersection over Union)."""
    true_mask = np.asarray(true_mask).astype(bool)
    pred_mask = np.asarray(pred_mask).astype(bool)
    intersection = np.logical_and(true_mask, pred_mask)
    union = np.logical_or(true_mask, pred_mask)
    if union.sum() == 0: # Both masks are empty
        return 1.0
    return intersection.sum() / (union.sum() + 1e-6) # Add epsilon for stability

# --- Visualization (modified from provided) ---
def _create_overlay_for_plane(ct_2d_slice, seg_2d_slice_dict, segment_colors, overlay_alpha=0.3):
    ct_slice_normalized = normalize(ct_2d_slice.astype(np.float32))
    ct_colored_rgba = plt.cm.bone(ct_slice_normalized)
    final_blended_rgb = ct_colored_rgba[..., :3].copy()
    
    for seg_name, seg_2d_slice in seg_2d_slice_dict.items():
        if seg_name in segment_colors:
            segment_color_rgb = segment_colors[seg_name]
            seg_mask_binary = (seg_2d_slice > 0) # Works for binary or labeled masks
            mask_pixels = seg_mask_binary > 0
            for c in range(3):
                final_blended_rgb[mask_pixels, c] = \
                    (final_blended_rgb[mask_pixels, c] * (1 - overlay_alpha)) + \
                    (segment_color_rgb[c] * overlay_alpha)
    return np.clip(final_blended_rgb, 0, 1)

def visualize_ortho_views_with_all_masks(
    img_volume_zyx,
    seg_volume_dict, # e.g., {'ground_truth_tumor': gt_mask, 'predicted_tumor': pred_mask}
    slice_coords_zyx,
    pixel_spacing_row_col,
    slice_thickness_z,
    segment_colors=None,
    overlay_alpha=0.3,
    main_title="Orthogonal Views with Segment Overlays"
    ):

    if segment_colors is None: # Define default colors if none provided
        segment_colors = {
            'ground_truth_tumor': [0, 1, 0],  # Green for ground truth
            'predicted_tumor': [1, 0, 0],  # Red for prediction
            'liver': [0, 0, 1] # Blue for liver (if present)
        }

    if not isinstance(img_volume_zyx, np.ndarray) or img_volume_zyx.ndim != 3:
        print("Error: Image volume is not a 3D NumPy array.")
        return
    
    for seg_name, seg_volume in seg_volume_dict.items():
        if not isinstance(seg_volume, np.ndarray) or seg_volume.ndim != 3:
            print(f"Error: Segmentation volume '{seg_name}' is not a 3D NumPy array.")
            return
        if img_volume_zyx.shape != seg_volume.shape:
            print(f"Error: Image shape {img_volume_zyx.shape} and {seg_name} seg shape {seg_volume.shape} mismatch.")
            return
    
    if not (pixel_spacing_row_col and len(pixel_spacing_row_col) == 2 and 
            all(s > 0 for s in pixel_spacing_row_col) and slice_thickness_z > 0):
        print("Error: Invalid spacing parameters (must be positive).")
        return

    z_idx, y_idx, x_idx = map(int,slice_coords_zyx)
    num_slices_z, height_y, width_x = img_volume_zyx.shape

    z_idx = np.clip(z_idx, 0, num_slices_z - 1)
    y_idx = np.clip(y_idx, 0, height_y - 1)
    x_idx = np.clip(x_idx, 0, width_x - 1)

    axial_ct_slice = img_volume_zyx[z_idx, :, :]
    sagittal_ct_slice = img_volume_zyx[:, :, x_idx]
    coronal_ct_slice = img_volume_zyx[:, y_idx, :]

    axial_seg_dict = {name: (seg_vol[z_idx, :, :] > 0) for name, seg_vol in seg_volume_dict.items()}
    sagittal_seg_dict = {name: (seg_vol[:, :, x_idx] > 0) for name, seg_vol in seg_volume_dict.items()}
    coronal_seg_dict = {name: (seg_vol[:, y_idx, :] > 0) for name, seg_vol in seg_volume_dict.items()}

    axial_overlay = _create_overlay_for_plane(axial_ct_slice, axial_seg_dict, segment_colors, overlay_alpha)
    sagittal_overlay = _create_overlay_for_plane(sagittal_ct_slice, sagittal_seg_dict, segment_colors, overlay_alpha)
    coronal_overlay = _create_overlay_for_plane(coronal_ct_slice, coronal_seg_dict, segment_colors, overlay_alpha)

    ps_row_y, ps_col_x = pixel_spacing_row_col[0], pixel_spacing_row_col[1]
    aspect_axial = ps_row_y / ps_col_x
    aspect_sagittal = slice_thickness_z / ps_row_y
    aspect_coronal = slice_thickness_z / ps_col_x
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    axes[0].imshow(axial_overlay, aspect=aspect_axial, origin='upper')
    axes[0].set_title(f'Axial (Z = {z_idx})')
    axes[0].set_xlabel(f'X-axis (cols, spacing {ps_col_x:.2f}mm)')
    axes[0].set_ylabel(f'Y-axis (rows, spacing {ps_row_y:.2f}mm)')
    axes[0].axhline(y_idx, color='cyan', linestyle='--', alpha=0.7)
    axes[0].axvline(x_idx, color='cyan', linestyle='--', alpha=0.7)

    axes[1].imshow(sagittal_overlay, aspect=aspect_sagittal, origin='lower')
    axes[1].set_title(f'Sagittal (X = {x_idx})')
    axes[1].set_xlabel(f'Y-axis (rows, spacing {ps_row_y:.2f}mm)')
    axes[1].set_ylabel(f'Z-axis (slices, spacing {slice_thickness_z:.2f}mm)')
    axes[1].axhline(z_idx, color='cyan', linestyle='--', alpha=0.7)
    axes[1].axvline(y_idx, color='cyan', linestyle='--', alpha=0.7)

    axes[2].imshow(coronal_overlay, aspect=aspect_coronal, origin='lower')
    axes[2].set_title(f'Coronal (Y = {y_idx})')
    axes[2].set_xlabel(f'X-axis (cols, spacing {ps_col_x:.2f}mm)')
    axes[2].set_ylabel(f'Z-axis (slices, spacing {slice_thickness_z:.2f}mm)')
    axes[2].axhline(z_idx, color='cyan', linestyle='--', alpha=0.7)
    axes[2].axvline(x_idx, color='cyan', linestyle='--', alpha=0.7)
    
    for ax in axes:
        ax.set_facecolor('black')

    legend_elements = [plt.Line2D([0], [0], color=color, lw=4, label=seg_name.replace("_", " ").title())
                       for seg_name, color in segment_colors.items() if seg_name in seg_volume_dict]
    
    if legend_elements:
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))

    fig.suptitle(f'{main_title} at (Z:{z_idx}, Y:{y_idx}, X:{x_idx})', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = Path(f'results/task2/ortho_views_assessment_Z{z_idx}_Y{y_idx}_X{x_idx}.png')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, facecolor='black')
    plt.show()
    print(f"Orthogonal views saved to {save_path}")


def create_gif_with_all_masks(img_pixelarray, seg_3d_dict, slice_thickness, col_pixel_spacing, segment_colors, n_frames=30, alpha=0.3, gif_filename_suffix="assessment"):
    if img_pixelarray.shape[0] == 0:
        print("Cannot create GIF: Image data is empty.")
        return
    if not seg_3d_dict or all(seg.shape[0] == 0 for seg in seg_3d_dict.values()):
        print("Cannot create GIF: All segmentation data is empty.")
        return

    frames_list = []
    img_min_val, img_max_val = np.min(img_pixelarray), np.max(img_pixelarray)
    denominator = img_max_val - img_min_val if img_max_val - img_min_val != 0 else 1.0

    for i, angle in enumerate(np.linspace(0, 360, n_frames, endpoint=False)):
        rotated_img = rotate(img_pixelarray, angle, axes=(1, 2), reshape=False, order=1, cval=img_min_val)
        rotated_segs = {name: rotate(seg_3d, angle, axes=(1, 2), reshape=False, order=0, mode='nearest')
                        for name, seg_3d in seg_3d_dict.items()}

        mip_img = maximum_intensity_projection(rotated_img, axis=1)
        mip_segs = {name: maximum_intensity_projection(rot_seg, axis=1)
                    for name, rot_seg in rotated_segs.items()}

        mip_img_scaled = (mip_img - img_min_val) / denominator
        mip_img_norm = np.clip(mip_img_scaled, 0, 1)
        
        cmap_bone = plt.get_cmap('bone')
        mip_img_colored_rgba = cmap_bone(mip_img_norm)
        combined_image_rgb = mip_img_colored_rgba[..., :3].copy()

        for seg_name, mip_seg in mip_segs.items():
            if seg_name in segment_colors:
                mask = (mip_seg > 0).astype(float) # MIP of binary/labeled mask will still be >0 where segment is
                mask_rgb = np.array(segment_colors[seg_name])
                alpha_mask_expanded = mask[..., np.newaxis] * alpha
                combined_image_rgb = combined_image_rgb * (1 - alpha_mask_expanded) + mask_rgb * alpha_mask_expanded
        
        combined_image_rgb = np.clip(combined_image_rgb, 0, 1)
        combined_image_uint8 = (combined_image_rgb * 255).astype(np.uint8)

        rescaled_frame_uint8 = combined_image_uint8
        if col_pixel_spacing > 0 and slice_thickness > 0 and slice_thickness != col_pixel_spacing:
            height_scale_factor = slice_thickness / col_pixel_spacing
            zoom_factors_for_display = [height_scale_factor, 1, 1]
            temp_rescaled = zoom(combined_image_uint8, zoom_factors_for_display, order=1, mode='nearest')
            rescaled_frame_uint8 = np.clip(temp_rescaled, 0, 255).astype(np.uint8)
        
        flipped_frame_uint8 = np.flipud(rescaled_frame_uint8)
        frames_list.append(flipped_frame_uint8)

    if frames_list:
        gif_path = f'results/task2/rotating_mip_{gif_filename_suffix}.gif'
        imageio.mimsave(gif_path, frames_list, fps=10)
        print(f"Animation saved as '{gif_path}'")
    else:
        print("No frames generated for GIF.")

def maximum_intensity_projection(image, axis=0): # Already provided, but good to have it self-contained
    return np.max(image, axis=axis)


def main():
    image_path = "0697/31_EQP_Ax5.00mm"
    # Assuming tumor_seg_path provides the ground truth tumor mask
    ground_truth_tumor_seg_path = "0697/31_EQP_Ax5.00mm_ManualROI_Tumor.dcm"
    # Optional: liver mask for context in visualization
    liver_seg_path = "0697/31_EQP_Ax5.00mm_ManualROI_Liver.dcm" 

    # Check paths
    if not Path(image_path).exists():
        print(f"Error: CT Image path does not exist: {image_path}"); return
    if not Path(ground_truth_tumor_seg_path).exists():
        print(f"Error: Ground Truth Tumor segmentation path does not exist: {ground_truth_tumor_seg_path}"); return
    
    liver_mask_available = Path(liver_seg_path).exists()
    if not liver_mask_available:
        print(f"Warning: Liver segmentation path does not exist: {liver_seg_path}. Proceeding without it.")

    try:
        print("Loading CT image data...")
        img_dcmset_loaded = load_image_data(image_path)
        if not img_dcmset_loaded: print("No DICOM files loaded. Exiting."); return
        
        img_dcmset_processed, img_pixelarray, slice_thickness, full_pixel_spacing = process_image_data(img_dcmset_loaded)
        print(f"CT Image volume shape: {img_pixelarray.shape}")
        print(f"Slice thickness: {slice_thickness:.2f}mm, Pixel spacing: {full_pixel_spacing[0]:.2f}mm (row), {full_pixel_spacing[1]:.2f}mm (col)")
  
        # Create 3D aligned ground truth tumor mask
        # Let's check the segment number from the dcm_obj if possible
        print("\nLoading ground truth tumor segmentation data...")
        gt_tumor_seg_raw, gt_tumor_dcm_obj = load_segmentation_data(ground_truth_tumor_seg_path)
        
        # Try SOP mapping first, then fallback
        gt_tumor_3d_labeled = create_segmentation_masks(gt_tumor_dcm_obj, img_dcmset_processed, 
                                                       img_pixelarray.shape, ignore_sop_mapping=False)
        
        # Check all possible segment numbers in the result
        unique_segments = np.unique(gt_tumor_3d_labeled)
        print(f"Unique segment values in ground truth: {unique_segments}")
        
        # Try to find the tumor segment (usually the non-zero value)
        tumor_segments = unique_segments[unique_segments > 0]
        if len(tumor_segments) > 0:
            gt_tumor_segment_number = tumor_segments[0]  # Take first non-zero
        else:
            print("No segments found, trying fallback mapping...")
            gt_tumor_3d_labeled = create_segmentation_masks(gt_tumor_dcm_obj, img_dcmset_processed, 
                                                           img_pixelarray.shape, ignore_sop_mapping=True)
            unique_segments = np.unique(gt_tumor_3d_labeled)
            tumor_segments = unique_segments[unique_segments > 0]
            gt_tumor_segment_number = tumor_segments[0] if len(tumor_segments) > 0 else 1
        
        ground_truth_tumor_mask_binary = (gt_tumor_3d_labeled == gt_tumor_segment_number)
        
        if not np.any(ground_truth_tumor_mask_binary):
            print("ERROR: Ground truth tumor mask is still empty after all attempts!")
            return
            
        print(f"Ground truth tumor mask: {np.sum(ground_truth_tumor_mask_binary)} voxels")

        # --- a) Extract bounding box and centroid from the ground truth tumor mask ---
        print("\n--- Task a: Extracting Bounding Box and Centroid ---")
        tumor_bbox, tumor_centroid_zyx = get_mask_bbox_centroid(ground_truth_tumor_mask_binary)
        
        if tumor_bbox and tumor_centroid_zyx:
            print(f"Tumor Bounding Box (z_min, z_max, y_min, y_max, x_min, x_max):")
            print(f"  Z: {tumor_bbox['z_min']} - {tumor_bbox['z_max']}")
            print(f"  Y: {tumor_bbox['y_min']} - {tumor_bbox['y_max']}")
            print(f"  X: {tumor_bbox['x_min']} - {tumor_bbox['x_max']}")
            print(f"Tumor Centroid (Z, Y, X): ({tumor_centroid_zyx[0]:.2f}, {tumor_centroid_zyx[1]:.2f}, {tumor_centroid_zyx[2]:.2f})")
        else:
            print("Could not extract bounding box or centroid. Ground truth tumor mask might be empty.")
            return # Cannot proceed if these are not found

        # --- b) Create a semi-automatic tumor segmentation algorithm ---
        print("\n--- Task b: Semi-automatic Tumor Segmentation ---")
        
        # Analyze intensities within ground truth for better thresholds
        tumor_intensities = img_pixelarray[ground_truth_tumor_mask_binary]
        intensity_mean = np.mean(tumor_intensities)
        intensity_std = np.std(tumor_intensities)
        
        print(f"Tumor intensity stats - Mean: {intensity_mean:.1f}, Std: {intensity_std:.1f}")
        
        # Use more conservative tolerance for region growing
        adaptive_tolerance = min(30, intensity_std * 1.5)
        predicted_tumor_mask = segment_tumor_centroid_region_growing(img_pixelarray, tumor_centroid_zyx, 
                                                                            tolerance=adaptive_tolerance)
        
        # # If you want to assess the centroid method instead, uncomment the next line:
        final_predicted_tumor_mask = predicted_tumor_mask

        # --- c) Visualize and Assess Correctness ---
        print("\n--- Task c: Visualization and Assessment ---")
        
        # Numerical Assessment
        dice = dice_coefficient(ground_truth_tumor_mask_binary, final_predicted_tumor_mask)
        jaccard = jaccard_index(ground_truth_tumor_mask_binary, final_predicted_tumor_mask)
        print(f"Numerical Assessment (comparing to ground truth):")
        print(f"  Dice Coefficient: {dice}")
        print(f"  Jaccard Index (IoU): {jaccard}")

        # Prepare for visualization
        seg_volumes_for_viz = {
            'ground_truth_tumor': ground_truth_tumor_mask_binary,
            'predicted_tumor': final_predicted_tumor_mask
        }
        segment_colors_for_viz = {
            'ground_truth_tumor': [0, 1, 0],  # Green
            'predicted_tumor': [1, 0, 0],   # Red
        }

        # Load liver mask for context if available
        if liver_mask_available:
            print("\nLoading liver segmentation data for context...")
            liver_seg_raw, liver_dcm_obj = load_segmentation_data(liver_seg_path)
            liver_segment_number = 1 # Default
            if hasattr(liver_dcm_obj, 'SegmentSequence') and len(liver_dcm_obj.SegmentSequence) > 0:
                liver_segment_number = liver_dcm_obj.SegmentSequence[0].SegmentNumber
            liver_3d_labeled = create_segmentation_masks(liver_dcm_obj, img_dcmset_processed, img_pixelarray.shape, ignore_sop_mapping=False)
            liver_mask_binary = (liver_3d_labeled == liver_segment_number)
            if np.any(liver_mask_binary):
                seg_volumes_for_viz['liver'] = liver_mask_binary
                segment_colors_for_viz['liver'] = [0, 0, 1] # Blue
                print(f"Liver mask added to visualization.")
            else:
                print("Warning: Processed liver mask is empty.")

        # Visual Assessment: Orthogonal Views
        # Use the tumor centroid (or center of image if centroid is problematic) for slice coordinates
        if tumor_centroid_zyx:
            z_viz, y_viz, x_viz = map(int, tumor_centroid_zyx)
        else: # Fallback to center of image
            z_viz = img_pixelarray.shape[0] // 2
            y_viz = img_pixelarray.shape[1] // 2
            x_viz = img_pixelarray.shape[2] // 2
        
        # Ensure visualization coordinates are within bounds
        z_viz = np.clip(z_viz, 0, img_pixelarray.shape[0] - 1)
        y_viz = np.clip(y_viz, 0, img_pixelarray.shape[1] - 1)
        x_viz = np.clip(x_viz, 0, img_pixelarray.shape[2] - 1)
        
        coords_for_viz = (z_viz, y_viz, x_viz)
        
        print(f"\nVisualizing orthogonal views at Z={coords_for_viz[0]}, Y={coords_for_viz[1]}, X={coords_for_viz[2]}...")
        visualize_ortho_views_with_all_masks(
            img_pixelarray,
            seg_volumes_for_viz,
            coords_for_viz,
            full_pixel_spacing, # [row_spacing_y, col_spacing_x]
            slice_thickness,
            segment_colors_for_viz
        )

        # Visual Assessment: GIF
        print("\nCreating GIF animation with all relevant masks...")
        create_gif_with_all_masks(
            img_pixelarray, 
            seg_volumes_for_viz, 
            slice_thickness, 
            full_pixel_spacing[1], # col_pixel_spacing (for X)
            segment_colors_for_viz,
            gif_filename_suffix="tumor_assessment"
        )
        
        print("\n--- Task 2 Completed ---")

    except ValueError as ve:
        print(f"ValueError in main execution: {ve}")
    except AttributeError as ae:
        print(f"AttributeError in main execution: {ae}")
    except Exception as e:
        print(f"An unexpected error occurred in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()