import os
import logging
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor

# Suppress warnings from PyRadiomics
logging.getLogger("radiomics").setLevel(logging.ERROR)


class RadiomicsFeatureExtractor:
    def __init__(self, ct_list, seg_list, output_excel, selected_labels=None, selected_features=None):
        """
        Initialize the feature extractor object.

        Args:
            ct_list (list): List of paths to CT images.
            seg_list (list): List of paths to corresponding segmentation masks.
            output_excel (str): Path to save the output Excel file.
            selected_labels (list or None): Labels to extract features for (or None for all).
            selected_features (dict or None): Dict of features to extract, or None for all shape + first-order.
        """
        self.ct_list = ct_list
        self.seg_list = seg_list
        self.output_excel = output_excel
        self.selected_labels = selected_labels
        self.selected_features = selected_features
        self.extractor = self._initialize_extractor()

    def _initialize_extractor(self):
        extractor = featureextractor.RadiomicsFeatureExtractor()
        extractor.disableAllFeatures()

        if self.selected_features is None:
            extractor.enableFeatureClassByName('shape')
            extractor.enableFeatureClassByName('firstorder')
        else:
            extractor.enableFeaturesByName(self.selected_features)

        return extractor

    def _get_labels_to_process(self, mask):
        label_array = sitk.GetArrayFromImage(mask)
        all_labels = set(label_array.flatten()) - {0}

        if self.selected_labels is None:
            return list(all_labels)
        else:
            return [label for label in self.selected_labels if label in all_labels]

    def extract_features(self, image_path, mask_path):
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
                    continue

                records.append({
                    'label': label,
                    'feature_type': feature_type,
                    'feature_name': feature_name,
                    'value': value
                })

        return pd.DataFrame(records)

    def run(self):
        with pd.ExcelWriter(self.output_excel, engine='xlsxwriter') as writer:
            for ct_path, seg_path in zip(self.ct_list, self.seg_list):
                base_name = os.path.basename(ct_path).replace('.nii.gz', '')
                if not os.path.exists(ct_path) or not os.path.exists(seg_path):
                    print(f"⚠️ Missing file for {base_name}, skipping.")
                    continue

                print(f"🔍 Processing {base_name}...")
                try:
                    df = self.extract_features(ct_path, seg_path)
                    sheet_name = base_name[:31]
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                except Exception as e:
                    print(f"❌ Error processing {base_name}: {e}")

        print(f"✅ All features saved to: {self.output_excel}")


### --- EXECUTE CODE --- ###
if __name__ == "__main__":
    ct_list = ["/content/drive/MyDrive/CT/CT1.nii.gz", "/content/drive/MyDrive/CT/CT2.nii.gz"]
    seg_list = ["/content/drive/MyDrive/seg/CT1.nii.gz", "/content/drive/MyDrive/seg/CT2.nii.gz"]

    extractor = RadiomicsFeatureExtractor(
        ct_list=ct_list,
        seg_list=seg_list,
        output_excel="/content/drive/MyDrive/features.xlsx",
        selected_labels=None,
        selected_features=None
    )
    extractor.run()
