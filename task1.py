import os
import pydicom
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, zoom
import imageio

# Ensure output directories exist at the beginning
Path("results/task1").mkdir(parents=True, exist_ok=True)
Path("frames").mkdir(parents=True, exist_ok=True)

def normalize(input_array):
    amin = np.amin(input_array)
    amax = np.amax(input_array)
    # Avoid division by zero if amin == amax (e.g., for a black image)
    if amax == amin:
        return np.zeros_like(input_array)
    return (input_array - amin) / (amax - amin)


def load_image_data(image_path):
    img_dcmset = []
    for root, _, filenames in os.walk(image_path):
        for filename in filenames:
            dcm_path = Path(root) / filename
            try:
                dicom = pydicom.dcmread(dcm_path, force=True)
                # Basic check for pixel data
                if hasattr(dicom, 'PixelData'):
                    img_dcmset.append(dicom)
            except Exception as e:
                print(f"Warning: Could not read or process DICOM file {dcm_path}: {e}")
    return img_dcmset


def process_image_data(img_dcmset):
    if not img_dcmset:
        raise ValueError("Input DICOM set is empty. Check image_path or DICOM contents.")

    # Filter out DICOMs without essential attributes for safety
    img_dcmset = [
        dcm for dcm in img_dcmset
        if hasattr(dcm, 'SliceThickness') and
           hasattr(dcm, 'PixelSpacing') and
           hasattr(dcm, 'AcquisitionNumber') and
           hasattr(dcm, 'ImagePositionPatient') and
           hasattr(dcm, 'pixel_array')
    ]

    if not img_dcmset:
        raise ValueError("No valid DICOMs with required attributes found in the set.")
        
    slice_thickness = img_dcmset[0].SliceThickness
    full_pixel_spacing = img_dcmset[0].PixelSpacing # This is a list [row_spacing, col_spacing]

    # Making sure that there is only one acquisition if AcquisitionNumber is present
    # Some CT series might not have AcquisitionNumber, or it might not be relevant for filtering
    if all(hasattr(dcm, 'AcquisitionNumber') for dcm in img_dcmset):
        try:
            acq_number = min(dcm.AcquisitionNumber for dcm in img_dcmset if dcm.AcquisitionNumber is not None) # Handle None
            img_dcmset = [dcm for dcm in img_dcmset if dcm.AcquisitionNumber == acq_number]
        except ValueError: # All AcquisitionNumbers are None or list is empty
             pass # proceed without filtering by acquisition number

    img_dcmset.sort(key=lambda x: x.ImagePositionPatient[2])
    img_pixelarray = np.stack([dcm.pixel_array for dcm in img_dcmset], axis=0)

    return img_dcmset, img_pixelarray, slice_thickness, full_pixel_spacing 


def load_segmentation_data(seg_path):
    mask_dcm = pydicom.dcmread(seg_path, force=True)
    if not hasattr(mask_dcm, 'pixel_array'):
        raise AttributeError(f"Segmentation file {seg_path} does not have pixel_array attribute.")
    seg_data = mask_dcm.pixel_array
    return seg_data, mask_dcm # Return mask_dcm as well for create_segmentation_masks

def _create_overlay_for_plane(ct_2d_slice, seg_2d_slice_dict, segment_colors, overlay_alpha=0.3):
    """Helper to create an overlay for a single 2D plane with multiple segments."""
    # Normalize CT slice (ensure it's float for division)
    ct_slice_normalized = normalize(ct_2d_slice.astype(np.float32))

    # Get base CT colors (e.g., bone colormap)
    ct_colored_rgba = plt.cm.bone(ct_slice_normalized) # This is RGBA
    
    # Start with CT RGB
    final_blended_rgb = ct_colored_rgba[..., :3].copy()
    
    # Apply each segmentation overlay
    for seg_name, seg_2d_slice in seg_2d_slice_dict.items():
        if seg_name in segment_colors:
            segment_color_rgb = segment_colors[seg_name]
            
            # Ensure segmentation is a binary mask (True where segment is present)
            seg_mask_binary = (seg_2d_slice > 0)
            
            # Find pixels where the segmentation mask is true
            mask_pixels = seg_mask_binary > 0
            
            # For these pixels, blend the current color with the segment color
            for c in range(3): # Iterate over R, G, B channels
                final_blended_rgb[mask_pixels, c] = \
                    (final_blended_rgb[mask_pixels, c] * (1 - overlay_alpha)) + \
                    (segment_color_rgb[c] * overlay_alpha)
            
    return np.clip(final_blended_rgb, 0, 1)

def visualize_ortho_views_with_overlay(
    img_volume_zyx,           # 3D CT data (e.g., Z, Y, X)
    seg_volume_dict,          # Dict of 3D Seg data: {'liver': seg_3d_liver, 'tumor': seg_3d_tumor}
    slice_coords_zyx,         # tuple (z_idx, y_idx, x_idx) for plane intersection
    pixel_spacing_row_col,    # list/tuple [row_spacing_y, col_spacing_x]
    slice_thickness_z,        # float for Z spacing
    segment_colors=None,      # Dict of colors: {'liver': [1,0,0], 'tumor': [0,1,0]}
    overlay_alpha=0.3,
    main_title="Orthogonal Views with Multi-Segment Overlay"
    ):

    if segment_colors is None:
        segment_colors = {
            'liver': [1, 0, 0],   # Red
            'tumor': [0, 1, 0]    # Green
        }

    if not isinstance(img_volume_zyx, np.ndarray) or img_volume_zyx.ndim != 3:
        print("Error: Image volume is not a 3D NumPy array.")
        return
    
    # Validate all segmentation volumes
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

    z_idx, y_idx, x_idx = map(int,slice_coords_zyx) # Ensure integer indices
    num_slices_z, height_y, width_x = img_volume_zyx.shape

    # Clip indices to be within bounds
    z_idx = np.clip(z_idx, 0, num_slices_z - 1)
    y_idx = np.clip(y_idx, 0, height_y - 1)
    x_idx = np.clip(x_idx, 0, width_x - 1)

    # Extract 2D planes from CT volume
    axial_ct_slice = img_volume_zyx[z_idx, :, :]
    sagittal_ct_slice = img_volume_zyx[:, :, x_idx]
    coronal_ct_slice = img_volume_zyx[:, y_idx, :]

    # Extract 2D planes from all segmentation volumes
    axial_seg_dict = {}
    sagittal_seg_dict = {}
    coronal_seg_dict = {}
    
    for seg_name, seg_volume in seg_volume_dict.items():
        axial_seg_dict[seg_name] = (seg_volume[z_idx, :, :] > 0)
        sagittal_seg_dict[seg_name] = (seg_volume[:, :, x_idx] > 0)
        coronal_seg_dict[seg_name] = (seg_volume[:, y_idx, :] > 0)

    # Create overlay images using the helper
    axial_overlay = _create_overlay_for_plane(axial_ct_slice, axial_seg_dict, segment_colors, overlay_alpha)
    sagittal_overlay = _create_overlay_for_plane(sagittal_ct_slice, sagittal_seg_dict, segment_colors, overlay_alpha)
    coronal_overlay = _create_overlay_for_plane(coronal_ct_slice, coronal_seg_dict, segment_colors, overlay_alpha)

    # Calculate aspect ratios for imshow
    # aspect = data_pixel_height_spacing / data_pixel_width_spacing
    ps_row_y, ps_col_x = pixel_spacing_row_col[0], pixel_spacing_row_col[1]

    aspect_axial = ps_row_y / ps_col_x          # Y-spacing / X-spacing
    aspect_sagittal = slice_thickness_z / ps_row_y # Z-spacing / Y-spacing
    aspect_coronal = slice_thickness_z / ps_col_x  # Z-spacing / X-spacing
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 7)) # Adjusted figsize

    # Axial View (Y rows, X columns)
    axes[0].imshow(axial_overlay, aspect=aspect_axial, origin='upper')
    axes[0].set_title(f'Axial (Z = {z_idx})')
    axes[0].set_xlabel(f'X-axis (cols, spacing {ps_col_x:.2f}mm)')
    axes[0].set_ylabel(f'Y-axis (rows, spacing {ps_row_y:.2f}mm)')
    axes[0].axhline(y_idx, color='lime', linestyle='--', alpha=0.7) # Line for Y-coordinate of intersection
    axes[0].axvline(x_idx, color='lime', linestyle='--', alpha=0.7) # Line for X-coordinate of intersection

    # Sagittal View (Z slices, Y rows)
    axes[1].imshow(sagittal_overlay, aspect=aspect_sagittal, origin='upper')
    axes[1].set_title(f'Sagittal (X = {x_idx})')
    axes[1].set_xlabel(f'Y-axis (rows, spacing {ps_row_y:.2f}mm)')
    axes[1].set_ylabel(f'Z-axis (slices, spacing {slice_thickness_z:.2f}mm)')
    axes[1].axhline(z_idx, color='lime', linestyle='--', alpha=0.7) # Line for Z-coordinate
    axes[1].axvline(y_idx, color='lime', linestyle='--', alpha=0.7) # Line for Y-coordinate

    # Coronal View (Z slices, X columns)
    axes[2].imshow(coronal_overlay, aspect=aspect_coronal, origin='upper')
    axes[2].set_title(f'Coronal (Y = {y_idx})')
    axes[2].set_xlabel(f'X-axis (cols, spacing {ps_col_x:.2f}mm)')
    axes[2].set_ylabel(f'Z-axis (slices, spacing {slice_thickness_z:.2f}mm)')
    axes[2].axhline(z_idx, color='lime', linestyle='--', alpha=0.7) # Line for Z-coordinate
    axes[2].axvline(x_idx, color='lime', linestyle='--', alpha=0.7) # Line for X-coordinate
    
    for ax in axes:
        ax.set_facecolor('black') # Black background for unplotted areas

    # Create legend for segments
    legend_elements = []
    for seg_name, color in segment_colors.items():
        if seg_name in seg_volume_dict:
            legend_elements.append(plt.Line2D([0], [0], color=color, lw=4, label=seg_name.capitalize()))
    
    if legend_elements:
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))

    fig.suptitle(f'{main_title} at (Z:{z_idx}, Y:{y_idx}, X:{x_idx})', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle
    save_path = Path(f'results/task1/ortho_views_multi_seg_Z{z_idx}_Y{y_idx}_X{x_idx}.png')
    save_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, facecolor='black')
    plt.show()

def create_segmentation_masks(seg_dcm_obj, ct_dcm_list, ignore=False):
    seg_data_raw = seg_dcm_obj.pixel_array
    seg_raw_shape = seg_data_raw.shape # (num_seg_frames, H, W)
    
    num_ct_slices = len(ct_dcm_list)
    if num_ct_slices == 0:
        raise ValueError("CT DCM list is empty, cannot determine target segmentation shape.")
    
    # Output aligned segmentation mask should have same H, W as seg_data_raw, but depth of num_ct_slices
    seg_3d_aligned = np.zeros((num_ct_slices, seg_raw_shape[1], seg_raw_shape[2]), dtype=np.int32)

    if ignore:
        # This block assumes seg_data_raw has a specific layered structure:
        # frames 0 to (NFSB-1) for Liver, NFSB to (2*NFSB-1) for Tumor, etc.
        # NFSB = num_frames_per_segment_block (hardcoded as 109 in original logic)
        num_frames_per_segment_block = 109 

        segment_definitions = {
            1: (0, num_frames_per_segment_block),  # Liver
            2: (num_frames_per_segment_block, 2 * num_frames_per_segment_block),  # Tumor
            3: (2 * num_frames_per_segment_block, 3 * num_frames_per_segment_block),  # Vein
            4: (3 * num_frames_per_segment_block, 4 * num_frames_per_segment_block)   # Aorta
        }

        for seg_value, (start_idx, end_idx) in segment_definitions.items():
            # Extract the assumed block of frames for this segment from seg_data_raw
            segment_frames_source = seg_data_raw[start_idx:end_idx]
            
            # Map these frames to the seg_3d_aligned slices
            # Iterate up to the shortest of: num_ct_slices, available frames in source block
            num_slices_to_map = min(num_ct_slices, segment_frames_source.shape[0])
            
            for z in range(num_slices_to_map):
                # Pixels with value 1 in the source frame are assigned the seg_value
                seg_3d_aligned[z][segment_frames_source[z] == 1] = seg_value
    
    else: # ignore == False (use DICOM metadata for mapping)
        if not hasattr(seg_dcm_obj, 'PerFrameFunctionalGroupsSequence'):
            print("Warning: PerFrameFunctionalGroupsSequence not found in segmentation DICOM. Cannot map segments accurately without it when ignore=False.")
            return seg_3d_aligned # Return empty or partially filled seg_3d_aligned

        ref_segs = seg_dcm_obj.PerFrameFunctionalGroupsSequence
        valid_masks = {}
        for i, frame_group in enumerate(ref_segs):
            try:
                ref_ct_uid = frame_group.DerivationImageSequence[0].SourceImageSequence[0].ReferencedSOPInstanceUID
                seg_number = frame_group.SegmentIdentificationSequence[0].ReferencedSegmentNumber
                # ImagePositionPatient could be used for additional validation if needed
                # img_pos = frame_group.PlanePositionSequence[0].ImagePositionPatient
                valid_masks.setdefault(ref_ct_uid, {})[seg_number] = {"frame_idx_in_raw": i}
            except (AttributeError, IndexError) as e:
                # print(f"Warning: Could not extract mapping info for a frame in SEG: {e}")
                continue # Skip this frame group if essential tags are missing

        for z_idx, ct_slice_dcm in enumerate(ct_dcm_list):
            if ct_slice_dcm.SOPInstanceUID in valid_masks:
                for seg_number, info in valid_masks[ct_slice_dcm.SOPInstanceUID].items():
                    raw_frame_idx = info["frame_idx_in_raw"]
                    if raw_frame_idx < seg_raw_shape[0]: # Check bounds for seg_data_raw
                        seg_slice_pixels = seg_data_raw[raw_frame_idx]
                        # Assuming pixel value 1 in the raw segment frame marks the segment area
                        seg_3d_aligned[z_idx][seg_slice_pixels == 1] = int(seg_number)
    return seg_3d_aligned


def maximum_intensity_projection(image, axis=0):
    return np.max(image, axis=axis)


def create_gif(img_pixelarray, seg_3d_dict, slice_thickness, pixel_spacing, n_frames=30, alpha=0.3):
    if img_pixelarray.shape[0] == 0:
        print("Cannot create GIF: Image data is empty.")
        return
    
    # Check if any segmentation data exists
    if not seg_3d_dict or all(seg.shape[0] == 0 for seg in seg_3d_dict.values()):
        print("Cannot create GIF: All segmentation data is empty.")
        return
    
    # Basic check for compatibility
    for seg_name, seg_3d in seg_3d_dict.items():
        if img_pixelarray.shape != seg_3d.shape:
            print(f"Warning: Image shape {img_pixelarray.shape} and {seg_name} segmentation shape {seg_3d.shape} mismatch.")

    frames_list = []
    # Define colors for different segments
    segment_colors = {
        'liver': [1.0, 0.0, 0.0],  # Red
        'tumor': [0.0, 1.0, 0.0],  # Green
    }

    # --- Flicker Correction: Use global min/max for CT normalization ---
    img_min_val = np.min(img_pixelarray)
    img_max_val = np.max(img_pixelarray)
    denominator = img_max_val - img_min_val
    if denominator == 0: # Avoid division by zero for constant volume
        denominator = 1.0 
    # --- End Flicker Correction ---

    for i, angle in enumerate(np.linspace(0, 360, n_frames, endpoint=False)):
        # Rotate CT image
        rotated_img = rotate(img_pixelarray, angle, axes=(1, 2), reshape=False, order=1, cval=img_min_val)
        
        # Rotate all segmentations
        rotated_segs = {}
        for seg_name, seg_3d in seg_3d_dict.items():
            rotated_segs[seg_name] = rotate(seg_3d, angle, axes=(1, 2), reshape=False, order=0, mode='nearest')

        # Create MIP for CT
        mip_img = maximum_intensity_projection(rotated_img, axis=1) 
        
        # Create MIP for all segmentations
        mip_segs = {}
        for seg_name, rotated_seg in rotated_segs.items():
            mip_segs[seg_name] = maximum_intensity_projection(rotated_seg, axis=1)

        # --- Flicker Correction: Apply global normalization ---
        mip_img_scaled = (mip_img - img_min_val) / denominator
        mip_img_norm = np.clip(mip_img_scaled, 0, 1)
        # --- End Flicker Correction ---

        cmap_bone = plt.get_cmap('bone')
        mip_img_colored_rgba = cmap_bone(mip_img_norm) 

        combined_image_rgb = mip_img_colored_rgba[..., :3].copy()

        # Apply all segmentation overlays
        for seg_name, mip_seg in mip_segs.items():
            if seg_name in segment_colors:
                mask = (mip_seg > 0).astype(float)
                mask_rgb = np.array(segment_colors[seg_name])
                alpha_mask_expanded = mask[..., np.newaxis] * alpha
                combined_image_rgb = combined_image_rgb * (1 - alpha_mask_expanded) + mask_rgb * alpha_mask_expanded
        
        combined_image_rgb = np.clip(combined_image_rgb, 0, 1)
        combined_image_uint8 = (combined_image_rgb * 255).astype(np.uint8)

        # --- Aspect Ratio Correction ---
        rescaled_frame_uint8 = combined_image_uint8 # Default if no rescaling
        if pixel_spacing > 0 and slice_thickness > 0 and slice_thickness != pixel_spacing:
            height_scale_factor = slice_thickness / pixel_spacing
            
            # zoom_factors for (height, width, channels)
            zoom_factors_for_display = [height_scale_factor, 1, 1] 
            
            temp_rescaled = zoom(combined_image_uint8, zoom_factors_for_display, order=1, mode='nearest')
            rescaled_frame_uint8 = np.clip(temp_rescaled, 0, 255).astype(np.uint8)
        # --- End Aspect Ratio Correction ---

        # --- Flip the frame upside down ---
        flipped_frame_uint8 = np.flipud(rescaled_frame_uint8)
        # --- End Flip ---

        frames_list.append(flipped_frame_uint8)

    if frames_list:
        imageio.mimsave('results/task1/rotating_mip_multi_seg_animation.gif', frames_list, fps=10)
        print("Multi-segment animation saved as 'results/task1/rotating_mip_multi_seg_animation.gif'")
    else:
        print("No frames generated for GIF.")


def main():
    # Paths for CT images and segmentation masks
    image_path = "0697/31_EQP_Ax5.00mm" 
    liver_seg_path = "0697/31_EQP_Ax5.00mm_ManualROI_Liver.dcm"
    tumor_seg_path = "0697/31_EQP_Ax5.00mm_ManualROI_Tumor.dcm"

    # Check if paths exist
    if not Path(image_path).exists():
        print(f"Error: Image path does not exist: {image_path}")
        return
    if not Path(liver_seg_path).exists():
        print(f"Error: Liver segmentation path does not exist: {liver_seg_path}")
        return
    if not Path(tumor_seg_path).exists():
        print(f"Error: Tumor segmentation path does not exist: {tumor_seg_path}")
        return

    try:
        img_dcmset_loaded = load_image_data(image_path)
        if not img_dcmset_loaded:
            print("No DICOM files loaded from image path. Exiting.")
            return

        img_dcmset_processed, img_pixelarray, slice_thickness, full_pixel_spacing = process_image_data(img_dcmset_loaded)
        
        # Load both segmentation masks
        liver_seg_data_raw, liver_seg_dcm_obj = load_segmentation_data(liver_seg_path)
        tumor_seg_data_raw, tumor_seg_dcm_obj = load_segmentation_data(tumor_seg_path)

        # Create 3D aligned segmentation masks for both liver and tumor
        use_ignore_logic = True # Or False, depending on your SEG file's metadata quality
        liver_seg_3d_aligned = create_segmentation_masks(liver_seg_dcm_obj, img_dcmset_processed, ignore=use_ignore_logic)
        tumor_seg_3d_aligned = create_segmentation_masks(tumor_seg_dcm_obj, img_dcmset_processed, ignore=use_ignore_logic)

        # Create dictionary for multiple segmentations
        seg_3d_dict = {
            'liver': liver_seg_3d_aligned,
            'tumor': tumor_seg_3d_aligned
        }

        # Define colors for visualization
        segment_colors = {
            'liver': [1, 0, 0],    # Red
            'tumor': [0, 1, 0]     # Green
        }

        # --- Orthogonal Views Visualization ---
        if all(seg is not None and seg.size > 0 for seg in seg_3d_dict.values()):
            # Define coordinates for the intersection of the three planes.
            idx_z_ortho = 22 
            if not (0 <= idx_z_ortho < img_pixelarray.shape[0]):
                print(f"Chosen Z-index {idx_z_ortho} is out of bounds. Defaulting to center slice.")
                idx_z_ortho = img_pixelarray.shape[0] // 2
            
            idx_y_ortho = img_pixelarray.shape[1] // 2
            idx_x_ortho = img_pixelarray.shape[2] // 2
            coords_for_ortho = (idx_z_ortho, idx_y_ortho, idx_x_ortho)

            visualize_ortho_views_with_overlay(
                img_pixelarray, 
                seg_3d_dict, 
                coords_for_ortho, 
                full_pixel_spacing,
                slice_thickness,
                segment_colors
            )
        else:
            print("Skipping orthogonal views: One or more segmentation masks were not generated or are empty.")

        # Multi-segment GIF creation
        create_gif(img_pixelarray, seg_3d_dict, slice_thickness, full_pixel_spacing[1], n_frames=30)

    except Exception as e:
        print(f"An error occurred in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()