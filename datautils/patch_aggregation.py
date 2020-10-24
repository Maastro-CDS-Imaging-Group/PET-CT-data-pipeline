import numpy as np


class PatchAggregator3D():
    def __init__(self, patch_size=(128,128,32), volume_size=(144,144,48), focal_point_stride=(5,5,5), overlap_handling=None, unpadding=0):

        """
        All arguments are apecified in (W,H,D) format
        """

        self.patch_size = list(patch_size)
        self.volume_size = list(volume_size)
        self.focal_point_stride = list(focal_point_stride)

        # Convert to (D,H,W) ordering
        self.patch_size.reverse()
        self.volume_size.reverse()
        self.focal_point_stride.reverse()

        self.overlap_handling = overlap_handling  # None or 'union'
        self.unpadding = unpadding

        # Increase the volume size to account for the padding used during patch sampling
        if unpadding > 0:
            self.volume_size = [s + unpadding for s in self.volume_size] # (D,H,W)

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
        """
        Args:
            patches_list: List of numpy arrays. Shape of the each array is in (D,H,W) ordering
        Returns:
            full_volume: Aggregated from the patches. shape has dim (D,H,W) ordering
        """
        # Define a zeros array of shape volume_size
        full_volume = np.zeros(self.volume_size)

        patch_size = np.array(self.patch_size)

        for i, patch in enumerate(patches_list):

            # Find the indices of the volume where the patch needs to be placed
            global_focal_point = np.array(self.valid_focal_points[i])
            global_start_idxs = global_focal_point.astype(np.int) - np.floor(patch_size/2).astype(np.int)
            z1, y1, x1 = global_start_idxs
            z2, y2, x2 = global_start_idxs + patch_size

            # Handle overlap or not
            if self.overlap_handling is None:
                full_volume[z1:z2, y1:y2, x1:x2] = patch

            elif self.overlap_handling == 'union':
                full_volume_copy = full_volume.copy()
                full_volume[z1:z2, y1:y2, x1:x2] = patch
                full_volume = np.maximum(full_volume, full_volume_copy)

        # If padding was used during patch sampling, remove it from the full volume
        if self.unpadding > 0:
            full_volume = full_volume[:-self.unpadding, :-self.unpadding, :-self.unpadding]

        return full_volume



def get_pred_labelmap_patches_list(pred_prob_patches):
    """
    Get a list of predicted labelmap patches from the the model's predicted batch of probabilities patches

    Args:
        pred_prob_patches: Tensor. Batch of predicted probabilites patches. Shape (N,C,D,H,W)
    Returns:
        pred_labelmap_patches_list: List of length N. Each element is a tensor of hape (D,H,W)
    """
    pred_labelmap_patches_list = []

    for i in range(pred_prob_patches.shape[0]):

        # Convert to numpy, collapse channel dim for the predicted patch
        pred_patch = pred_prob_patches[i].numpy() # Shape (C,D,H,W)
        pred_patch = np.argmax(pred_patch, axis=0)  # Shape (D,H,W)

        # Accumulate in the list
        pred_labelmap_patches_list.append(pred_patch)

    return pred_labelmap_patches_list



if __name__ == '__main__':

    import torch
    from patch_sampling import PatchSampler3D, get_num_valid_patches

    np.random.seed(0)

    # Arguments specified in (W,H,D) ordering
    volume_size = (144,144,48)
    patch_size = (128,128,32)
    focal_point_stride = (20,20,20)
    padding = 4

    patch_sampler = PatchSampler3D(patch_size,
                                   sampling='sequential',
                                   focal_point_stride=focal_point_stride,
                                   padding=padding)

    patch_aggregator = PatchAggregator3D(patch_size,
                                        volume_size,
                                        focal_point_stride,
                                        overlap_handling=None,
                                        unpadding=padding)

    num_valid_patches = get_num_valid_patches(patch_size, volume_size, focal_point_stride, padding=padding)
    print("Num patches:", num_valid_patches)

    # (D,H,W) order followed internally
    random_labelmap = np.random.randint(low=0, high=2, size=(volume_size[2], volume_size[1], volume_size[0]))

    patches_list = patch_sampler.get_samples(subject_dict={'GTV-labelmap': torch.from_numpy(random_labelmap)},
                                    num_patches=num_valid_patches)

    patches_list = [patch['GTV-labelmap'] for patch in patches_list]

    recovered_labelmap = patch_aggregator.aggregate(patches_list)


    def dice(labelmap_1, labelmap_2):
        intersection = np.sum(labelmap_1 * labelmap_2)
        dice_score = 2 * intersection / (np.sum(labelmap_1) + np.sum(labelmap_2))
        return dice_score

    print(dice(random_labelmap, recovered_labelmap))