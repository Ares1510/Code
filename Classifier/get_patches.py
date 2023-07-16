import numpy as np
import pandas as pd
import SimpleITK as sitk

def normalize(array):
    max_HU = 400.
    min_HU = -1000.
    array = (array - min_HU) / (max_HU - min_HU)
    array[array > 1] = 1.
    array[array < 0] = 0.
    return array

def get_patches(pid):
    series_path = f'data/LIDC-IDRI-{pid}/'
    series_reader = sitk.ImageSeriesReader()
    series_ids = series_reader.GetGDCMSeriesIDs(series_path)
    series_id = series_ids[0]
    series_reader.SetFileNames(series_reader.GetGDCMSeriesFileNames(f'data/LIDC-IDRI-{pid}/', series_ids[0]))
    dicom_series = series_reader.Execute()

    df = pd.read_csv('data/candidates_V2_8.csv')
    df = df[df['seriesuid'] == series_id]

    label = df['class'].values
    num_can = df.shape[0]
    world_coords = df[['coordX', 'coordY', 'coordZ']].values
    img_arr = sitk.GetArrayViewFromImage(dicom_series)
    origin = np.tile(dicom_series.GetOrigin(), (num_can, 1))
    pixel_coords = (np.round(np.absolute(world_coords - origin) / dicom_series.GetSpacing())).astype(int)

    patches = []
    for can in range(num_can):
        voxel_cord = pixel_coords[can,:]
        x_pos = int(voxel_cord[0])
        y_pos = int(voxel_cord[1])
        z_pos = int(voxel_cord[2])

        patch_size = 25
        x_lower = np.max([0, x_pos - patch_size])
        x_upper = np.min([x_pos + patch_size, dicom_series.GetWidth()])
        
        y_lower = np.max([0, y_pos - patch_size])
        y_upper = np.min([y_pos + patch_size, dicom_series.GetHeight()])
         
        patch = img_arr[z_pos, y_lower:y_upper, x_lower:x_upper]
        # patch = normalize(patch)
        patches.append(patch)
    return np.asarray(patches), np.asarray(label)

def get_all_patches():
    pid = ['0385', '0406', '0614', '0677', '0698', '0819', '0939', '1004']
    patches = []
    labels = []
    for p in pid:
        patch, label = get_patches(p)
        patches.append(patch)
        labels.append(label)
    patches = np.concatenate(patches, axis=0)
    labels = np.concatenate(labels, axis=0)
    return patches, labels


if __name__ == '__main__':
    patches, labels = get_all_patches()
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    neg_idx = np.random.choice(neg_idx, size=(pos_idx.shape[0] * 10), replace=False)
    idx = np.concatenate([pos_idx, neg_idx], axis=0)
    patches = patches[idx]
    labels = labels[idx]

    np.save('data/patches.npy', patches)
    np.save('data/labels.npy', labels)

