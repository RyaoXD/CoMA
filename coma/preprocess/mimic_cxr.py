# from MGCA
import numpy as np
import pandas as pd
from coma.constants import *
from coma.preprocess.utils import extract_mimic_text


extract_text = False
generate_ccd = False

np.random.seed(42)


def generate_master_ccd():
    def get_ccd_file_path(image_path):
        path_parts = image_path.split('/')
        study_dir = path_parts[-2]
        ccd_filename = f"{study_dir}_ccd.txt"
        ccd_path_parts = path_parts[1:-1]
        ccd_path_parts[-1] = ccd_filename
        
        ccd_path = '/'.join(ccd_path_parts)
        return ccd_path
    
    def read_ccd_file_content(ccd_relative_path):
        if not ccd_relative_path:
            return ""
        
        ccd_full_path = os.path.join(MIMIC_CXR_DATA_DIR, ccd_relative_path)
        
        if not os.path.exists(ccd_full_path):
            print(f"file does not exist: {ccd_full_path}")
            return ""
        
        try:
            with open(ccd_full_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                return content
        except Exception as e:
            print(f"failed to read file {ccd_full_path}: {e}")
            return ""
    
    master_df = pd.read_csv(MIMIC_CXR_MASTER_CSV)
    result_data = []
    for index, row in master_df.iterrows():
        image_path = row['Path']
        ccd_path = get_ccd_file_path(image_path)
        ccd_content = read_ccd_file_content(ccd_path) if ccd_path else ""
        new_row = {
            'Path': row['Path'],
            'dicom_id': row['dicom_id'],
            'subject_id': row['subject_id'],
            'study_id': row['study_id'],
            'ViewPosition': row['ViewPosition'],
            'text': ccd_content,
            'split': row['split']
        }
        
        result_data.append(new_row)
    
    result_df = pd.DataFrame(result_data)
    result_df.to_csv(MIMIC_CXR_MASTER_CCD_CSV, index=False, encoding='utf-8')    
    return result_df

def main():
    if extract_text:
        extract_mimic_text()
    metadata_df = pd.read_csv(MIMIC_CXR_META_CSV)
    metadata_df = metadata_df[["dicom_id", "subject_id",
                               "study_id", "ViewPosition"]].astype(str)
    metadata_df["study_id"] = metadata_df["study_id"].apply(lambda x: "s"+x)
    # Only keep frontal images
    metadata_df = metadata_df[metadata_df["ViewPosition"].isin(["PA", "AP"])]

    text_df = pd.read_csv(MIMIC_CXR_TEXT_CSV)
    text_df.dropna(subset=["impression", "findings"], how="all", inplace=True)
    text_df = text_df[["study", "impression", "findings"]]
    text_df.rename(columns={"study": "study_id"}, inplace=True)

    split_df = pd.read_csv(MIMIC_CXR_SPLIT_CSV)
    split_df = split_df.astype(str)
    split_df["study_id"] = split_df["study_id"].apply(lambda x: "s"+x)
    # TODO: merge validate and test into test.
    split_df["split"] = split_df["split"].apply(
        lambda x: "valid" if x == "validate" or x == "test" else x)

    chexpert_df = pd.read_csv(MIMIC_CXR_CHEXPERT_CSV)
    chexpert_df[["subject_id", "study_id"]] = chexpert_df[[
        "subject_id", "study_id"]].astype(str)
    chexpert_df["study_id"] = chexpert_df["study_id"].apply(lambda x: "s"+x)

    master_df = pd.merge(metadata_df, text_df, on="study_id", how="left")
    master_df = pd.merge(master_df, split_df, on=["dicom_id", "subject_id", "study_id"], how="inner")
    master_df.dropna(subset=["impression", "findings"], how="all", inplace=True)
    
    n = len(master_df)
    master_data = master_df.values

    root_dir = str(MIMIC_CXR_DATA_DIR).split("/")[-1] + "/files"
    path_list = []
    for i in range(n):
        row = master_data[i]
        file_path = "%s/p%s/p%s/%s/%s.jpg" % (root_dir, str(
            row[1])[:2], str(row[1]), str(row[2]), str(row[0]))
        path_list.append(file_path)
        
    master_df.insert(loc=0, column="Path", value=path_list)

    # Create labeled data df
    labeled_data_df = pd.merge(master_df, chexpert_df, on=[
                               "subject_id", "study_id"], how="inner")
    labeled_data_df.drop(["dicom_id", "subject_id", "study_id",
                          "impression", "findings"], axis=1, inplace=True)

    train_df = labeled_data_df.loc[labeled_data_df["split"] == "train"]
    train_df.to_csv(MIMIC_CXR_TRAIN_CSV, index=False)
    valid_df = labeled_data_df.loc[labeled_data_df["split"] == "valid"]
    valid_df.to_csv(MIMIC_CXR_TEST_CSV, index=False)

    # Fill nan in text
    master_df[["impression"]] = master_df[["impression"]].fillna(" ")
    master_df[["findings"]] = master_df[["findings"]].fillna(" ")
    master_df.to_csv(MIMIC_CXR_MASTER_CSV, index=False)

    if generate_ccd:
        generate_master_ccd()

if __name__ == "__main__":
    main()