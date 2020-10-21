import numpy as np


class PatchAggregator3D():
    def __init__(self, patch_size=(128,128,32), volume_size=(144,144,48), focal_point_stride=(5,5,5), overlap_handling=None):

        """
        All coordinates and shapes are in (W,H,D) format
        """

        self.patch_size = patch_size
        self.volume_size = volume_size
        self.focal_point_stride = focal_point_stride
        self.overlap_handling = overlap_handling  # None or 'union'

        self.valid_focal_points = self._get_valid_focal_points() # Valid focal points in volume coordinates


    def _get_valid_focal_points(self):
        patch_size = np.array(self.patch_size)
        valid_indx_range = [
                            np.zeros(3) + np.floor(patch_size/2),
                            np.array(self.volume_size) - np.ceil(patch_size/2)
                           ]

        z_range = np.arange(valid_indx_range[0][0], valid_indx_range[1][0] + 1, self.focal_point_stride[0]).astype(np.int)
        y_range = np.arange(valid_indx_range[0][1], valid_indx_range[1][1] + 1, self.focal_point_stride[1]).astype(np.int)
        x_range = np.arange(valid_indx_range[0][2], valid_indx_range[1][2] + 1, self.focal_point_stride[2]).astype(np.int)
        zs, ys, xs = np.meshgrid(z_range, y_range, x_range, indexing='ij')
        zs, ys, xs = zs.flatten(), ys.flatten(), xs.flatten()

        valid_focal_points = [(zs[i], ys[i], xs[i]) for i in range(zs.shape[0])]
        return valid_focal_points


    def aggregate(self, patches_list):
        # Define a zeros array of shape volume_size
        full_volume = np.zeros(self.volume_size)

        patch_size = np.array(self.patch_size)

        for i, patch in enumerate(patches_list):

            # Find the indices of the volume where the patch needs to be placed
            global_focal_point = np.array(self.valid_focal_points[i])
            global_start_idxs = global_focal_point.astype(np.int) - np.floor(patch_size/2).astype(np.int)
            x1, y1, z1 = global_start_idxs
            x2, y2, z2 = global_start_idxs + patch_size

            # Handle overlap or not
            if self.overlap_handling is None:
                full_volume[x1:x2, y1:y2, z1:z2] = patch

            if self.overlap_handling == 'union':
                full_volume_copy = full_volume.copy()
                full_volume[x1:x2, y1:y2, z1:z2] = patch
                full_volume = np.maximum(full_volume, full_volume_copy)

        return full_volume



if __name__ == '__main__':
    volume_size = (144,144,48)
    patch_size = (128,128,32)
    focal_point_stride = (10,10,10)
    patch_aggregator = PatchAggregator3D(patch_size,
                                     volume_size,
                                     focal_point_stride,
                                     overlap_handling='union')

    patches_list = [np.ones(patch_size) for i in range(8)]

    recovered_labelmap = patch_aggregator.aggregate(patches_list)