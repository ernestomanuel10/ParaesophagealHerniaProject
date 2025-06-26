import os
import logging
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor

# Suppress warnings from PyRadiomics
logging.getLogger("radiomics").setLevel(logging.ERROR)


class RadiomicsFeatureExtractor:
    def __init__(self, ct_dir, seg_dir, output_excel, selected_labels=None, selected_features=None):
        """
        Initialize the feature extractor object.

        Args:
            ct_dir (str): Directory containing CT images.
            seg_dir (str): Directory containing corresponding segmentation masks.
            output_excel (str): Path to save the output Excel file.
            selected_labels (list or None): Labels to extract features for (or None for all).
            selected_features (dict or None): Dict of features to extract, or None for all shape + first-order.
        """
        self.ct_dir = ct_dir
        self.seg_dir = seg_dir
        self.output_excel = output_excel
        self.selected_labels = selected_labels
        self.selected_features = selected_features
        self.extractor = self._initialize_extractor()

    def _initialize_extractor(self):
        """
        Configure the radiomics extractor with specified feature classes.
        """
        extractor = featureextractor.RadiomicsFeatureExtractor()
        extractor.disableAllFeatures()

        if self.selected_features is None:
            extractor.enableFeatureClassByName('shape')
            extractor.enableFeatureClassByName('firstorder')
        else:
            extractor.enableFeaturesByName(self.selected_features)

        return extractor

    def _get_labels_to_process(self, mask):
        """
        Determine which labels in the mask to extract features from.

        Args:
            mask (SimpleITK.Image): The segmentation mask image.

        Returns:
            list of int: List of label values to process.
        """
        label_array = sitk.GetArrayFromImage(mask)
        all_labels = set(label_array.flatten()) - {0}  # exclude background

        if self.selected_labels is None:
            return list(all_labels)
        else:
            return [label for label in self.selected_labels if label in all_labels]

    def extract_features(self, image_path, mask_path):
        """
        Extract features from one CT/mask pair.

        Args:
            image_path (str): Path to CT image.
            mask_path (str): Path to segmentation mask.

        Returns:
            pd.DataFrame: Extracted features for each label.
        """
        image = sitk.ReadImage(image_path)
        mask = sitk.ReadImage(mask_path)
        mask.CopyInformation(image)

        records = []

        labels_to_process = self._get_labels_to_process(mask)
        for label in labels_to_process:
            try:
                features = self.extractor.execute(image, mask, label=label)
            except Exception as e:
                print(f"❌ Failed to extract label {label} from {os.path.basename(image_path)}: {e}")
                continue

            for key, value in features.items():
                if key.startswith('original_shape_'):
                    feature_type = 'shape'
                    feature_name = key.replace('original_shape_', '')
                elif key.startswith('original_firstorder_'):
                    feature_type = 'intensity'
                    feature_name = key.replace('original_firstorder_', '')
                else:
                    continue  # skip other feature types

                records.append({
                    'label': label,
                    'feature_type': feature_type,
                    'feature_name': feature_name,
                    'value': value
                })

        return pd.DataFrame(records)

    def run(self):
        """
        Process all CT and segmentation pairs and write results to an Excel file.
        """
        with pd.ExcelWriter(self.output_excel, engine='xlsxwriter') as writer:
            for file in os.listdir(self.ct_dir):
                if not file.endswith('.nii.gz'):
                    continue

                base_name = file.replace('.nii.gz', '')
                ct_path = os.path.join(self.ct_dir, file)
                seg_path = os.path.join(self.seg_dir, f'{base_name}.nii')

                if not os.path.exists(seg_path):
                    print(f"⚠️ Segmentation not found for {base_name}, skipping.")
                    continue

                print(f"🔍 Processing {base_name}...")
                try:
                    df = self.extract_features(ct_path, seg_path)
                    sheet_name = base_name[:31]  # Excel sheet name limit
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                except Exception as e:
                    print(f"❌ Error processing {base_name}: {e}")

        print(f"✅ All features saved to: {self.output_excel}")



###         --- EXECUTE CODE ---        ##
if __name__ == "__main__":
    extractor = RadiomicsFeatureExtractor(
        ct_dir="/content/drive/MyDrive/CT/",
        seg_dir="/content/drive/MyDrive/seg/",
        output_excel="/content/drive/MyDrive/features.xlsx",
        selected_labels=None,  # or [5, 15] for liver and esophagus
        selected_features=None  # or {'shape': ['Elongation'], 'firstorder': ['Kurtosis']}
    )
    extractor.run()
