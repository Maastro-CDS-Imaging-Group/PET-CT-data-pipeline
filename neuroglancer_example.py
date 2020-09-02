from __future__ import print_function

import argparse
import numpy as np

import SimpleITK as sitk
import neuroglancer

from data_utils.io import *


# Optional arguments
ap = argparse.ArgumentParser()
ap.add_argument(
    '-a',
    '--bind-address',
    help='Bind address for Python web server.  Use 127.0.0.1 (the default) to restrict access '
    'to browers running on the local machine, use 0.0.0.0 to permit access from remote browsers.')
ap.add_argument(
    '--static-content-url', help='Obtain the Neuroglancer client code from the specified URL.')
args = ap.parse_args()
if args.bind_address:
    neuroglancer.set_server_bind_address(args.bind_address)
if args.static_content_url:
    neuroglancer.set_static_content_source(url=args.static_content_url)


# Read image
image_path = "./data/HECKTOR/hecktor_train/hecktor_nii/CHGJ007/CHGJ007_ct.nii.gz"
print("Loading original CT ...")
ct_sitk = read_nifti(image_path)
ct_np = sitk.GetArrayFromImage(ct_sitk)


# Create the viewer
viewer = neuroglancer.Viewer()

dimensions = neuroglancer.CoordinateSpace(
    names=['x', 'y', 'z'],
    units='mm',
    scales=[100, 100, 100])

with viewer.txn() as s:
    s.dimensions = dimensions
    s.layers.append(
        name='Sample CT',
        layer=neuroglancer.LocalVolume(
            data=ct_np,
            dimensions=neuroglancer.CoordinateSpace(
                names=['x', 'y', 'z'],
                units=['mm','mm','mm'],
                scales=[100, 100, 100]),
            voxel_offset=(0, 0, 0),
        ),
        shader="""
void main() {
  emitRGB(vec3(toNormalized(getDataValue(0)),
               toNormalized(getDataValue(1)),
               toNormalized(getDataValue(2))));
}
""") # Do I need this shader? Looks like it's for RGB


print(viewer)