import os
import time
import numpy as np
import pydicom
from tqdm import tqdm
import pandas as pd
from scipy import ndimage
from skimage import morphology
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_dicom_series(dicom_dir):
    try:
        dicom_files = []
        for file in os.listdir(dicom_dir):
            if file.lower().endswith('.dcm'):
                dicom_files.append(os.path.join(dicom_dir, file))
        
        if not dicom_files:
            return None, "No DICOM files found in directory"
        
        slices = []
        for dicom_file in dicom_files:
            try:
                ds = pydicom.dcmread(dicom_file)
                if hasattr(ds, 'pixel_array'):
                    slices.append(ds)
            except Exception as e:
                print(f"Warning: Could not read {dicom_file}: {e}")
                continue
        
        if not slices:
            return None, "No valid DICOM slices found"
        
        try:
            slices.sort(key=lambda x: float(x.SliceLocation))
        except (AttributeError, ValueError):
            try:
                slices.sort(key=lambda x: int(x.InstanceNumber))
            except (AttributeError, ValueError):
                slices.sort(key=lambda x: x.filename)
        
        pixel_arrays = []
        for slice_ds in slices:
            pixel_array = slice_ds.pixel_array.astype(np.float64)
            
            if hasattr(slice_ds, 'RescaleSlope') and hasattr(slice_ds, 'RescaleIntercept'):
                pixel_array = pixel_array * float(slice_ds.RescaleSlope) + float(slice_ds.RescaleIntercept)
            
            pixel_arrays.append(pixel_array)
        
        volume = np.stack(pixel_arrays, axis=-1)  # Shape: (height, width, depth)
        
        volume = np.transpose(volume, (2, 0, 1))  # Change to (depth, height, width)
        
        return volume, None
        
    except Exception as e:
        return None, f"Error loading DICOM series: {str(e)}"

def bias_field_correction(img_data):
    """Simple bias field correction using N4 approximation"""
    smoothed = ndimage.gaussian_filter(img_data, sigma=10)
    smoothed[smoothed == 0] = 1
    corrected = img_data / smoothed * np.mean(smoothed)
    return corrected

def skull_strip(img_data, threshold_percentile=50):
    threshold = np.percentile(img_data[img_data > 0], threshold_percentile)
    brain_mask = img_data > threshold
    brain_mask = morphology.remove_small_objects(brain_mask, min_size=1000)
    brain_mask = ndimage.binary_fill_holes(brain_mask)
    skull_stripped = img_data * brain_mask
    return skull_stripped

def affine_to_mni(img_data, target_shape=(91, 109, 91)):
    zoom_factors = [target_shape[i] / img_data.shape[i] for i in range(3)]
    
    resampled = ndimage.zoom(img_data, zoom_factors, order=1)
    
    return resampled

def z_score_normalize(img_data):
    mask = img_data > 0
    if np.sum(mask) == 0:
        return img_data
    
    mean_val = np.mean(img_data[mask])
    std_val = np.std(img_data[mask])
    
    if std_val == 0:
        return img_data
    
    img_data[mask] = (img_data[mask] - mean_val) / std_val
    
    return img_data

def preprocess_dicom_directory(dicom_dir, output_dir, ptid):
    try:
        img_data, error = load_dicom_series(dicom_dir)
        if img_data is None:
            return False, error
        
        print(f"Loaded volume shape: {img_data.shape} for PTID: {ptid}")
        img_data = bias_field_correction(img_data)
        
        img_data = skull_strip(img_data)
        
        img_data = affine_to_mni(img_data)
        
        img_data = z_score_normalize(img_data)
        
        output_path = os.path.join(output_dir, f"{ptid}.npy")
        np.save(output_path, img_data.astype(np.float32))
        
        return True, None
    
    except Exception as e:
        return False, str(e)

def main():
    CSV_PATH = "MRI_Paths.csv"  
    OUTPUT_DIR = "mri_preproc"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading CSV file...")
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {CSV_PATH}")
        print("Please update CSV_PATH to point to your actual CSV file")
        return
    
    ptid_col = 'PTID'
    dir_col = 'dcm_directory'  
    
    if ptid_col not in df.columns or dir_col not in df.columns:
        print(f"Error: Required columns not found in CSV")
        print(f"Expected columns: {ptid_col}, {dir_col}")
        print(f"Available columns: {list(df.columns)}")
        return
    
    print(f"Found {len(df)} DICOM directories to process")
    print(f"Sample entries:")
    for i in range(min(3, len(df))):
        print(f"  PTID: {df.iloc[i][ptid_col]}, Directory: {df.iloc[i][dir_col]}")
    
    start_time = time.time()
    
    failed_scans = []
    successful_scans = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing DICOM directories"):
        ptid = row[ptid_col]
        dicom_dir = row[dir_col]
        
        if not os.path.exists(dicom_dir):
            failed_scans.append((ptid, f"Directory not found: {dicom_dir}"))
            continue
        dcm_files = [f for f in os.listdir(dicom_dir) if f.lower().endswith('.dcm')]
        if not dcm_files:
            failed_scans.append((ptid, f"No DICOM files found in: {dicom_dir}"))
            continue
        
        success, error = preprocess_dicom_directory(dicom_dir, OUTPUT_DIR, ptid)
        
        if success:
            successful_scans.append(ptid)
        else:
            failed_scans.append((ptid, error))
    
    total_time = time.time() - start_time
    
    print(f"\nPreprocessing completed!")
    print(f"Total time taken: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Successfully processed: {len(successful_scans)}/{len(df)} scans")
    print(f"Average time per scan: {total_time/len(df):.2f} seconds")
    
    if successful_scans:
        print(f"\nSuccessfully processed PTIDs: {successful_scans[:10]}{'...' if len(successful_scans) > 10 else ''}")
    
    if failed_scans:
        print(f"\nFailed scans ({len(failed_scans)}):")
        for ptid, error in failed_scans[:10]: 
            print(f"  PTID {ptid}: {error}")
        if len(failed_scans) > 10:
            print(f"  ... and {len(failed_scans) - 10} more failures")
    
    summary_path = os.path.join(OUTPUT_DIR, "preprocessing_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"DICOM Preprocessing Summary\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Total scans processed: {len(df)}\n")
        f.write(f"Successful: {len(successful_scans)}\n")
        f.write(f"Failed: {len(failed_scans)}\n")
        f.write(f"Success rate: {len(successful_scans)/len(df)*100:.1f}%\n")
        f.write(f"Total time: {total_time:.2f} seconds\n")
        f.write(f"Average time per scan: {total_time/len(df):.2f} seconds\n")
        f.write(f"\nSuccessful PTIDs:\n")
        for ptid in successful_scans:
            f.write(f"  {ptid}\n")
        f.write(f"\nFailed PTIDs:\n")
        for ptid, error in failed_scans:
            f.write(f"  {ptid}: {error}\n")
    
    print(f"\nSummary saved to: {summary_path}")

if __name__ == "__main__":
    main()