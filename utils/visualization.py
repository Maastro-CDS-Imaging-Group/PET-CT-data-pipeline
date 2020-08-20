import numpy as np
import matplotlib.pyplot as plt

import SimpleITK as sitk


# 1. Matplotlib wrappers to display sitk images

def custom_imshow(sitk_image, title=None, margin=0.05, dpi=80):
    nda = sitk.GetArrayViewFromImage(sitk_image)
    spacing = sitk_image.GetSpacing()
        
    if nda.ndim == 3:
        # fastest dim, either component or x
        c = nda.shape[-1]
        # the the number of components is 3 or 4 consider it an RGB image
        if not c in (3,4):
            nda = nda[nda.shape[0]//2,:,:]
    
    elif nda.ndim == 4:
        c = nda.shape[-1]
        if not c in (3,4):
            raise Runtime("Unable to show 3D-vector Image")
        # take a z-slice
        nda = nda[nda.shape[0]//2,:,:,:]
            
    ysize = nda.shape[0]
    xsize = nda.shape[1]
      
    # Make a figure big enough to accommodate an axis of xpixels by ypixels
    # as well as the ticklabels, etc...
    figsize = (1 + margin) * ysize / dpi, (1 + margin) * xsize / dpi

    fig = plt.figure(figsize=figsize, dpi=dpi)
    # Make the axis the right size...
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
    
    extent = (0, xsize*spacing[1], ysize*spacing[0], 0)
    
    t = ax.imshow(nda, extent=extent, interpolation=None)
    
    if nda.ndim == 2:
        t.set_cmap("gray")
    if(title):
        plt.title(title)


def custom_imshow_tile(sitk_image, xslices=[], yslices=[], zslices=[], title=None, margin=0.05, dpi=80):
    size = sitk_image.GetSize()
    img_xslices = [sitk_image[s,:,:] for s in xslices]
    img_yslices = [sitk_image[:,s,:] for s in yslices]
    img_zslices = [sitk_image[:,:,s] for s in zslices]
    
    maxlen = max(len(img_xslices), len(img_yslices), len(img_zslices))

    img_null = sitk.Image([0,0], sitk_image.GetPixelID(), sitk_image.GetNumberOfComponentsPerPixel())
    
    img_slices = []
    d = 0
    
    if len(img_xslices):
        img_slices += img_xslices + [img_null]*(maxlen-len(img_xslices))
        d += 1
    if len(img_yslices):
        img_slices += img_yslices + [img_null]*(maxlen-len(img_yslices))
        d += 1
    if len(img_zslices):
        img_slices += img_zslices + [img_null]*(maxlen-len(img_zslices))
        d +=1
    
    if maxlen != 0:
        if img.GetNumberOfComponentsPerPixel() == 1:
            sitk_image = sitk.Tile(img_slices, [maxlen,d])
        #TO DO check in code to get Tile Filter working with vector images
        else:
            img_comps = []
            for i in range(0, sitk_image.GetNumberOfComponentsPerPixel()):
                img_slices_c = [sitk.VectorIndexSelectionCast(s, i) for s in img_slices]
                img_comps.append(sitk.Tile(img_slices_c, [maxlen,d]))
            sitk_image = sitk.Compose(img_comps)
            
    myshow(sitk_image, title, margin, dpi)


# 2. Functions to display segmentation and registration results