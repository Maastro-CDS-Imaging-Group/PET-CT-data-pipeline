import numpy as np
import matplotlib.pyplot as plt

import SimpleITK as sitk


# 1. Matplotlib wrappers to display sitk images

def display_tile(sitk_image,
                 sagittal_slice_idxs=[], coronal_slice_idxs=[], axial_slice_idxs=[],
                 window_level = None, window_width = None,
                 margin=0.05, dpi=80):

    ndarray = sitk.GetArrayFromImage(sitk_image)
    spacing = sitk_image.GetSpacing()

    # Apply window
    if window_level != None and window_width != None:
        w_min = window_level - window_width//2
        w_max = window_level + window_width//2
        ndarray[ndarray < w_min] = w_min
        ndarray[ndarray > w_max] = w_max


    fig, [ax1,ax2,ax3] = plt.subplots(3)
    #fig.set_size_inches(0.5*18.5, 0.8*10.5)
    figsize = (1500/dpi, 1000/dpi)
    fig.set_size_inches(*figsize)
    fig.set_dpi(dpi)
    fig.subplots_adjust(hspace=0.2, top=0.75, bottom=0.25, left=0.25, right=0.75)


    # Extract axial slices --
    axial_slices = []
    for idx in axial_slice_idxs:
        image2d = ndarray[idx, :, :]
        axial_slices.append(image2d)

    axial_slices = np.hstack(axial_slices)

    n_rows = image2d.shape[0] # #rows of the 2d array - corresponds to sitk image height
    n_cols = image2d.shape[1] # #columns of the 2d array - corresponds to sitk image width
    extent = (0, len(axial_slice_idxs)*n_cols*spacing[0], n_rows*spacing[1], 0)
    ax1.imshow(axial_slices, extent=extent, interpolation=None, cmap='gray')
    ax1.set_title(f"Axial slices: {axial_slice_idxs}")
    ax1.axis('off')


    # Extract coronal slices --
    coronal_slices = []
    for idx in coronal_slice_idxs:
        image2d = ndarray[:, idx, :]
        image2d = np.rot90(image2d, 2)
        coronal_slices.append(image2d)

    coronal_slices = np.hstack(coronal_slices)

    n_rows = image2d.shape[0] # #rows of the 2d array - corresponds to sitk image depth
    n_cols = image2d.shape[1] # #columns of the 2d array - corresponds to sitk image width
    extent = (0, len(coronal_slice_idxs)*n_cols*spacing[0], n_rows*spacing[2], 0)
    ax2.imshow(coronal_slices, extent=extent, interpolation=None, cmap='gray')
    ax2.set_title(f"Coronal slices: {coronal_slice_idxs}")
    ax2.axis('off')


    # Extract sagittal slices --
    sagittal_slices = []
    for idx in sagittal_slice_idxs:
        image2d = ndarray[:, :, idx]
        image2d = np.rot90(image2d, k=2)
        image2d = np.flip(image2d, axis=1)
        sagittal_slices.append(image2d)

    sagittal_slices = np.hstack(sagittal_slices)

    n_rows = image2d.shape[0] # #rows of the 2d array - corresponds to sitk image depth
    n_cols = image2d.shape[1] # #columns of the 2d array - corresponds to sitk image height
    extent = (0, len(sagittal_slice_idxs)*n_cols*spacing[1], n_rows*spacing[2], 0)
    ax3.imshow(sagittal_slices, extent=extent, interpolation=None, cmap='gray')
    ax3.set_title(f"Sagittal slices: {sagittal_slice_idxs}")
    ax3.axis('off')

    plt.show()



# 2. Functions to display segmentation and registration results