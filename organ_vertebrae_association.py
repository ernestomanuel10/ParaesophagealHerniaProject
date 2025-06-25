# --- import ---

import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
from glob import glob
from scipy.interpolate import make_interp_spline
from scipy.stats import mannwhitneyu
import shutil

# --- anatomical labels ---
LABELS = {
    'Spleen': 1, 'Stomach': 6, 'Esophagus': 15, 'Liver': 5,
    'Vertebrae_S1': 26, 'Vertebrae_L5': 27, 'Vertebrae_L4': 28, 'Vertebrae_L3': 29, 'Vertebrae_L2': 30, 'Vertebrae_L1': 31,
    'Vertebrae_T12': 32, 'Vertebrae_T11': 33, 'Vertebrae_T10': 34, 'Vertebrae_T9': 35, 'Vertebrae_T8': 36, 'Vertebrae_T7': 37,
    'Vertebrae_T6': 38, 'Vertebrae_T5': 39, 'Vertebrae_T4': 40, 'Vertebrae_T3': 41, 'Vertebrae_T2': 42, 'Vertebrae_T1': 43,
    'Vertebrae_C7': 44, 'Vertebrae_C6': 45, 'Vertebrae_C5': 46, 'Vertebrae_C4': 47, 'Vertebrae_C3': 48, 'Vertebrae_C2': 49, 'Vertebrae_C1': 50,
}
vertebrae_labels = [name for name in LABELS if name.startswith("Vertebrae")]

def compute_physical_z_com(image, label):
    """
    Compute the physical z-coordinate of the center of mass for a given label.
    """
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(image)
    return stats.GetCentroid(label)[2] if stats.HasLabel(label) else None

def get_slice_index_from_physical_z(physical_z, image):
    """
    Convert physical z-coordinate to slice index using image metadata.
    """
    origin = image.GetOrigin()
    spacing = image.GetSpacing()
    direction = image.GetDirection()
    return int(round((physical_z - origin[2]) / (spacing[2] * direction[8])))

def find_first_last_slice(seg_array, label_value):
    """
    Return the first and last slice indices where the label appears in the segmentation array.
    """
    z_indices = np.where(np.any(seg_array == label_value, axis=(1, 2)))[0]
    return (int(z_indices[0]), int(z_indices[-1])) if len(z_indices) else (None, None)

def compute_summary(ct_dir, seg_dir):
    """
    Process a set of CT and segmentation files from two directories.
    Returns two DataFrames:
      - slice_df: contains first/last slice info for each organ
      - vertebrae_df: contains closest vertebrae to top/bottom of each organ
    """
    slice_records = []
    vertebrae_records = []

    # Process all CT files ending in .nii.gz
    all_files = sorted(f for f in os.listdir(ct_dir) if f.endswith('.nii.gz'))

    for fname in all_files:
        scan_name = os.path.splitext(os.path.basename(fname))[0]
        ct_path = os.path.join(ct_dir, fname)
        seg_path = os.path.join(seg_dir, f'{scan_name}.nii')

        if not os.path.exists(seg_path):
            print(f'Skipping {scan_name}: segmentation file not found.')
            continue

        # Load segmentation image and array
        seg_img = sitk.ReadImage(seg_path)
        seg_arr = sitk.GetArrayFromImage(seg_img)
        total_slices = seg_arr.shape[0]

        # Compute center of mass slice index for each vertebra
        vertebrae_slices = {}
        for name in vertebrae_labels:
            label = LABELS[name]
            binary_img = sitk.BinaryThreshold(seg_img, label, label, insideValue=1, outsideValue=0)
            z_physical = compute_physical_z_com(binary_img, 1)
            if z_physical is not None:
                vertebrae_slices[name] = get_slice_index_from_physical_z(z_physical, seg_img)

        # Initialize per-scan data containers
        organ_slices = {'scan name': scan_name}
        organ_vertebrae = {'scan name': scan_name}

        # For each organ, find bounding slices and map to closest vertebrae
        for organ in ['Stomach', 'Spleen', 'Liver', 'Esophagus']:
            label_val = LABELS[organ]
            first, last = find_first_last_slice(seg_arr, label_val)
            organ_slices[f'{organ.lower()}_first_slice'] = first
            organ_slices[f'{organ.lower()}_last_slice'] = last

            if first is not None and first > 1:
                closest_bottom = min(vertebrae_slices, key=lambda k: abs(vertebrae_slices[k] - first))
                organ_vertebrae[f'{organ.lower()}_bottom_vertebra'] = closest_bottom
            else:
                organ_vertebrae[f'{organ.lower()}_bottom_vertebra'] = None

            if last is not None and last < total_slices - 2:
                closest_top = min(vertebrae_slices, key=lambda k: abs(vertebrae_slices[k] - last))
                organ_vertebrae[f'{organ.lower()}_top_vertebra'] = closest_top
            else:
                organ_vertebrae[f'{organ.lower()}_top_vertebra'] = None

        # Append scan data
        slice_records.append(organ_slices)
        vertebrae_records.append(organ_vertebrae)

    # Return as two separate DataFrames
    slice_df = pd.DataFrame(slice_records)
    vertebrae_df = pd.DataFrame(vertebrae_records)
    
    return slice_df, vertebrae_df

# --- Run for each dataset ---
"""
Have a directory of CT scans and for segmentations
"""
slices_pe_df, vertebrae_pe_df  = compute_summary('CT/pe_hernia', 'seg/pe_hernia')
slices_any_df, vertebrae_any_df = compute_summary('CT/any_hernia', 'seg/any_hernia')
slices_no_df, vertebrae_no_df  = compute_summary('CT/no_hernia', 'seg/no_hernia')

# Add scan type column and concatenate all slice summaries
slices_pe_df['scan_type']  = 'pe_hernia'
slices_any_df['scan_type'] = 'any_hernia'
slices_no_df['scan_type']  = 'no_hernia'

final_slices_df = pd.concat([slices_pe_df, slices_any_df, slices_no_df], ignore_index=True)

# Add scan type column and concatenate all vertebrae summaries
vertebrae_pe_df['scan_type']  = 'pe_hernia'
vertebrae_any_df['scan_type'] = 'any_hernia'
vertebrae_no_df['scan_type']  = 'no_hernia'

final_vertebrae_df = pd.concat([vertebrae_pe_df, vertebrae_any_df, vertebrae_no_df], ignore_index=True)

final_slices_df.to_csv('csv_summary/all_slices_summary.csv', index=False)
final_vertebrae_df.to_csv('csv_summary/all_vertebrae_summary.csv', index=False)

# --- Plotting Function
def plot_vertebrae_distribution(df_pe, df_no, organ, ordered_vertebrae, mode='encapsulated'):
    """
    Plots the relative frequency distribution of vertebrae associated with a given organ
    for PE Hernia and No Hernia groups. Allows plotting by:
      - 'encapsulated': all vertebrae between bottom and top of the organ
      - 'top': only the top vertebra associated with the organ
    Saves a smoothed bar plot as a JPG.
    """

    # --- Extract vertebra indices per scan row ---
    def get_vertebrae_indices(row):
        top = row.get(f'{organ}_top_vertebra')
        bottom = row.get(f'{organ}_bottom_vertebra')

        if pd.isna(top) or pd.isna(bottom):
            return []

        try:
            top_idx = ordered_vertebrae.index(top)
            bottom_idx = ordered_vertebrae.index(bottom)
        except ValueError:
            return []

        if mode == 'top':
            # Include top vertebra only if within L4–T4 range
            idx = top_idx
            idx_L4 = ordered_vertebrae.index('Vertebrae_L4')
            idx_T4 = ordered_vertebrae.index('Vertebrae_T4')
            return [idx] if idx_L4 <= idx <= idx_T4 else []
        else:
            # Include all vertebrae from bottom to top (inclusive)
            return list(range(min(top_idx, bottom_idx), max(top_idx, bottom_idx) + 1))

    # --- Count vertebrae occurrences across all scans ---
    def compute_distribution(df):
        counts = {v: 0 for v in ordered_vertebrae}
        valid_rows = 0

        for _, row in df.iterrows():
            indices = get_vertebrae_indices(row)
            if not indices:
                continue
            valid_rows += 1
            for i in indices:
                counts[ordered_vertebrae[i]] += 1

        return counts, valid_rows

    # --- Draw smoothed line ---
    def smooth_line(x, y, color, label):
        mask = y > 0
        if np.sum(mask) >= 4:
            x_masked = x[mask]
            y_masked = y[mask]
            x_new = np.linspace(x_masked.min(), x_masked.max(), 300)
            y_smooth = make_interp_spline(x_masked, y_masked, k=3)(x_new)
            plt.plot(x_new, y_smooth, color=color, lw=2, label=label)
        else:
            plt.plot(x, y, color=color, lw=2, label=label)

    # --- Compute and normalize counts ---
    counts_pe, n_pe = compute_distribution(df_pe)
    counts_no, n_no = compute_distribution(df_no)
    x = np.arange(len(ordered_vertebrae))
    heights_pe = np.array([counts_pe[v] / n_pe if n_pe else 0 for v in ordered_vertebrae])
    heights_no = np.array([counts_no[v] / n_no if n_no else 0 for v in ordered_vertebrae])

    # --- Create plot ---
    plt.figure(figsize=(14, 6))
    width = 0.4
    plt.bar(x - width/2, heights_pe, width, label=f'PE Hernia (N={n_pe})', alpha=0.7)
    plt.bar(x + width/2, heights_no, width, label=f'No Hernia (N={n_no})', alpha=0.7)
    smooth_line(x, heights_pe, 'blue', 'PE Hernia Smoothed')
    smooth_line(x, heights_no, 'orange', 'No Hernia Smoothed')

    plt.xticks(x, ordered_vertebrae, rotation=90)
    plt.xlabel('Vertebra')
    plt.ylabel('Relative Frequency')
    plt.title(f'{"Top" if mode=="top" else "Encapsulated"} Vertebrae Distribution ({organ.capitalize()})')
    plt.legend()
    plt.tight_layout()

    # --- Save plot as JPG ---
    histogram_dir = 'histogram'
    filename = f"{organ}_{'top' if mode=='top' else 'encapsulated'}_vertebrae_comparison_pe_vs_no_hernia.jpg"
    plt.savefig(os.path.join(histogram_dir, filename), dpi=300)
    plt.close()

plot_vertebrae_distribution(vertebrae_pe_df, vertebrae_no_df, organ='stomach', ordered_vertebrae=vertebrae_labels, mode='encapsulated')
plot_vertebrae_distribution(vertebrae_pe_df, vertebrae_no_df, organ='stomach', ordered_vertebrae=vertebrae_labels, mode='top')