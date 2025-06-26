# === Vertebrae Summary OOP Pipeline ===
"""
This script defines a modular, object-oriented pipeline for processing CT and segmentation data to:
1. Compute slice indices and vertebrae associations for abdominal organs.
2. Save anatomical summaries into structured CSV files.
3. Generate smoothed histograms of vertebrae involvement by organ and pathology group.
"""

import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# --- Global Constants ---
LABELS = {
    'Spleen': 1, 'Stomach': 6, 'Esophagus': 15, 'Liver': 5,
    'Vertebrae_S1': 26, 'Vertebrae_L5': 27, 'Vertebrae_L4': 28, 'Vertebrae_L3': 29, 'Vertebrae_L2': 30, 'Vertebrae_L1': 31,
    'Vertebrae_T12': 32, 'Vertebrae_T11': 33, 'Vertebrae_T10': 34, 'Vertebrae_T9': 35, 'Vertebrae_T8': 36, 'Vertebrae_T7': 37,
    'Vertebrae_T6': 38, 'Vertebrae_T5': 39, 'Vertebrae_T4': 40, 'Vertebrae_T3': 41, 'Vertebrae_T2': 42, 'Vertebrae_T1': 43,
    'Vertebrae_C7': 44, 'Vertebrae_C6': 45, 'Vertebrae_C5': 46, 'Vertebrae_C4': 47, 'Vertebrae_C3': 48, 'Vertebrae_C2': 49, 'Vertebrae_C1': 50,
}

VERTEBRAE_LABELS = [label for label in LABELS if label.startswith("Vertebrae")]


# --- Helper Functions ---
def compute_physical_z_com(image, label):
    """
    Compute the physical Z-coordinate of the center of mass for a label in an image.

    Args:
        image (sitk.Image): Binary label image.
        label (int): Label value to compute center of mass for.

    Returns:
        float or None: Z-coordinate of the centroid if label is present.
    """
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(image)
    return stats.GetCentroid(label)[2] if stats.HasLabel(label) else None

def get_slice_index_from_physical_z(physical_z, image):
    """
    Convert a physical Z-coordinate into a slice index using image metadata.

    Args:
        physical_z (float): Z-coordinate.
        image (sitk.Image): Reference image.

    Returns:
        int: Corresponding slice index in the 3D array.
    """
    origin, spacing, direction = image.GetOrigin(), image.GetSpacing(), image.GetDirection()
    return int(round((physical_z - origin[2]) / (spacing[2] * direction[8])))

def find_first_last_slice(seg_array, label_value):
    """
    Find the first and last axial slices where a label appears.

    Args:
        seg_array (np.ndarray): 3D segmentation array.
        label_value (int): Label to search for.

    Returns:
        tuple: (first_slice, last_slice) or (None, None) if not found.
    """
    z_indices = np.where(np.any(seg_array == label_value, axis=(1, 2)))[0]
    return (int(z_indices[0]), int(z_indices[-1])) if len(z_indices) else (None, None)


# --- Data Extraction Pipeline ---
class VertebraePipeline:
    """
    Main class for extracting organ-vertebrae anatomical relationships from CT and segmentation files.
    """

    def __init__(self, ct_root, seg_root, output_dir):
        """
        Initialize the pipeline with paths for data input and output.

        Args:
            ct_root (str): Root directory for CT scan groups.
            seg_root (str): Root directory for corresponding segmentation masks.
            output_dir (str): Output directory for summary CSVs.
        """
        self.ct_dir = ct_root
        self.seg_root = seg_root
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def compute_summary(self, group):
        """
        Compute per-organ slice bounds and closest vertebrae for all scans in a group.

        Args:
            group (str): Group folder name (e.g. 'pe_hernia').

        Returns:
            tuple: (slice_df, vertebrae_df) DataFrames with organ slice and vertebrae associations.
        """
        ct_dir = os.path.join(self.ct_dir, group)
        seg_dir = os.path.join(self.seg_root, group)
        slice_records, vertebrae_records = [], []

        for fname in sorted(f for f in os.listdir(ct_dir) if f.endswith('.nii.gz')):
            scan_name = os.path.splitext(fname)[0]
            ct_path = os.path.join(ct_dir, fname)
            seg_path = os.path.join(seg_dir, f'{scan_name}.nii')

            if not os.path.exists(seg_path):
                print(f"⚠️ Missing segmentation for {scan_name}, skipping.")
                continue

            seg_img = sitk.ReadImage(seg_path)
            seg_arr = sitk.GetArrayFromImage(seg_img)
            vertebrae_slices = {}

            for name in VERTEBRAE_LABELS:
                label = LABELS[name]
                binary_img = sitk.BinaryThreshold(seg_img, label, label, 1, 0)
                z = compute_physical_z_com(binary_img, 1)
                if z:
                    vertebrae_slices[name] = get_slice_index_from_physical_z(z, seg_img)

            organ_slices = {'scan name': scan_name, 'scan_type': group}
            organ_vertebrae = {'scan name': scan_name, 'scan_type': group}

            for organ in ['Stomach', 'Spleen', 'Liver', 'Esophagus']:
                label_val = LABELS[organ]
                first, last = find_first_last_slice(seg_arr, label_val)
                organ_slices[f'{organ.lower()}_first_slice'] = first
                organ_slices[f'{organ.lower()}_last_slice'] = last

                organ_vertebrae[f'{organ.lower()}_bottom_vertebra'] = (
                    min(vertebrae_slices, key=lambda k: abs(vertebrae_slices[k] - first))
                    if first is not None and vertebrae_slices else None
                )
                organ_vertebrae[f'{organ.lower()}_top_vertebra'] = (
                    min(vertebrae_slices, key=lambda k: abs(vertebrae_slices[k] - last))
                    if last is not None and vertebrae_slices else None
                )

            slice_records.append(organ_slices)
            vertebrae_records.append(organ_vertebrae)

        return pd.DataFrame(slice_records), pd.DataFrame(vertebrae_records)

    def run_all(self):
        """
        Run `compute_summary` on all expected scan groups and export results.

        Saves:
            all_slices_summary.csv
            all_vertebrae_summary.csv
        """
        groups = ['pe_hernia', 'any_hernia', 'no_hernia']
        all_slices, all_vertebrae = [], []

        for group in groups:
            slices_df, vertebrae_df = self.compute_summary(group)
            all_slices.append(slices_df)
            all_vertebrae.append(vertebrae_df)

        final_slices_df = pd.concat(all_slices, ignore_index=True)
        final_vertebrae_df = pd.concat(all_vertebrae, ignore_index=True)

        final_slices_df.to_csv(os.path.join(self.output_dir, 'all_slices_summary.csv'), index=False)
        final_vertebrae_df.to_csv(os.path.join(self.output_dir, 'all_vertebrae_summary.csv'), index=False)
        print(f"✅ CSVs saved to {self.output_dir}")


# --- Histogram Plotting Class ---
class VertebraeHistogram:
    """
    Class for plotting relative frequency distributions of vertebrae involved with each organ.
    Allows comparison between PE Hernia and No Hernia scan groups.
    """

    def __init__(self, ordered_vertebrae, output_dir='histogram'):
        """
        Initialize the histogram plotter.

        Args:
            ordered_vertebrae (list): List of vertebra label strings in anatomical order.
            output_dir (str): Path to save plots.
        """
        self.ordered_vertebrae = ordered_vertebrae
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_distribution(self, df_pe, df_no, organ, mode='encapsulated'):
        """
        Generate and save histogram comparing vertebrae distributions between PE Hernia and No Hernia.

        Args:
            df_pe (pd.DataFrame): Vertebrae summary for PE Hernia scans.
            df_no (pd.DataFrame): Vertebrae summary for No Hernia scans.
            organ (str): Name of the organ (lowercase).
            mode (str): 'encapsulated' for full range or 'top' for top vertebra only.
        """

        def get_vertebrae_indices(row):
            """Return vertebrae index/indices from row depending on mode."""
            top = row.get(f'{organ}_top_vertebra')
            bottom = row.get(f'{organ}_bottom_vertebra')
            if pd.isna(top) or pd.isna(bottom): return []
            try:
                top_idx = self.ordered_vertebrae.index(top)
                bottom_idx = self.ordered_vertebrae.index(bottom)
            except ValueError:
                return []
            if mode == 'top':
                idx = top_idx
                idx_L4 = self.ordered_vertebrae.index('Vertebrae_L4')
                idx_T4 = self.ordered_vertebrae.index('Vertebrae_T4')
                return [idx] if idx_L4 <= idx <= idx_T4 else []
            return list(range(min(top_idx, bottom_idx), max(top_idx, bottom_idx) + 1))

        def compute_distribution(df):
            """Compute vertebra frequency across patients."""
            counts = {v: 0 for v in self.ordered_vertebrae}
            valid_rows = 0
            for _, row in df.iterrows():
                indices = get_vertebrae_indices(row)
                if not indices: continue
                valid_rows += 1
                for i in indices:
                    counts[self.ordered_vertebrae[i]] += 1
            return counts, valid_rows

        def smooth_line(x, y, color, label):
            """Plot smoothed spline for visual clarity."""
            mask = y > 0
            if np.sum(mask) >= 4:
                x_new = np.linspace(x[mask].min(), x[mask].max(), 300)
                y_smooth = make_interp_spline(x[mask], y[mask], k=3)(x_new)
                plt.plot(x_new, y_smooth, color=color, lw=2, label=label)
            else:
                plt.plot(x, y, color=color, lw=2, label=label)

        # --- Compute data ---
        counts_pe, n_pe = compute_distribution(df_pe)
        counts_no, n_no = compute_distribution(df_no)
        x = np.arange(len(self.ordered_vertebrae))
        y_pe = np.array([counts_pe[v] / n_pe if n_pe else 0 for v in self.ordered_vertebrae])
        y_no = np.array([counts_no[v] / n_no if n_no else 0 for v in self.ordered_vertebrae])

        # --- Plot ---
        plt.figure(figsize=(14, 6))
        plt.bar(x - 0.2, y_pe, width=0.4, label=f'PE Hernia (N={n_pe})', alpha=0.7)
        plt.bar(x + 0.2, y_no, width=0.4, label=f'No Hernia (N={n_no})', alpha=0.7)
        smooth_line(x, y_pe, 'blue', 'PE Hernia Smoothed')
        smooth_line(x, y_no, 'orange', 'No Hernia Smoothed')

        plt.xticks(x, self.ordered_vertebrae, rotation=90)
        plt.xlabel('Vertebra')
        plt.ylabel('Relative Frequency')
        plt.title(f'{organ.capitalize()} Vertebrae Distribution ({mode.capitalize()})')
        plt.legend()
        plt.tight_layout()

        filename = f"{organ}_{mode}_vertebrae_comparison_pe_vs_no_hernia.jpg"
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()


# --- Execution Entry Point ---
if __name__ == "__main__":

    pipeline = VertebraePipeline(
        ct_dir = '/content/drive/MyDrive/CT/', 
        seg_dir = '/content/drive/MyDrive/seg/', 
        output_dir = '/content/drive/MyDrive/csv_summary'
        )
    pipeline.run_all()


    df_pe = pd.read_csv('csv_summary/all_vertebrae_summary.csv')
    df_no = df_pe[df_pe['scan_type'] == 'no_hernia']
    df_pe = df_pe[df_pe['scan_type'] == 'pe_hernia']

    histogram = VertebraeHistogram(ordered_vertebrae=VERTEBRAE_LABELS)
    

    histogram.plot_distribution(df_pe, df_no, organ='stomach', mode='encapsulated')
    histogram.plot_distribution(df_pe, df_no, organ='stomach', mode='top')
