
import sys
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

sys.path.append("../")
from datasets.hecktor_petct_dataset import HECKTORPETCTDataset
from datasets.hecktor_unimodal_dataset import HECKTORUnimodalDataset
from datautils.preprocessing import Preprocessor
from datautils.patch_sampling import PatchSampler2D, PatchSampler3D, PatchQueue



data_dir = "/home/chinmay/Datasets/HECKTOR/hecktor_train/crFH_rs113_hecktor_nii"
patient_id_filepath = "../hecktor_meta/patient_IDs_train.txt"
preprocessor = Preprocessor()



def test_patch_orientation():
    PET_CT_dataset = HECKTORPETCTDataset(data_dir,
                                patient_id_filepath,
                                mode='training',
                                preprocessor=preprocessor,
                                input_representation='separate-volumes',
                                augment_data=False)

    patch_sampler = PatchSampler2D(patch_size=(450,450))
    patch_queue = PatchQueue(PET_CT_dataset,
                         max_length=16,
                         samples_per_volume=8,
                         sampler=patch_sampler,
                         num_workers=0,
                         shuffle_subjects=True,
                         shuffle_patches=True)

    patch_loader = DataLoader(patch_queue, batch_size=1)

    fig, axs = plt.subplots(1,3)
    for i, batch_of_patches in enumerate(patch_loader):
        print("Batch:",i+1)
        print(batch_of_patches['PET'].shape)
        PET_np = batch_of_patches['PET'][0,0,:,:].numpy()
        CT_np = batch_of_patches['CT'][0,0,:,:].numpy()
        axs[0].imshow(np.clip(PET_np, -2, 8), cmap='gist_rainbow')
        axs[1].imshow(np.clip(CT_np, -150, 150), cmap='gray')
        axs[2].imshow(batch_of_patches['GTV-labelmap'][0,:,:], cmap='gray')
        plt.show()
        break


def test_dataloader_limit():
    PET_dataset = HECKTORUnimodalDataset(data_dir,
                                patient_id_filepath,
                                mode='cval-CHMR-validation',
                                preprocessor=preprocessor,
                                input_modality='PET',
                                augment_data=False)
    print("Dataset length:", len(PET_dataset))

    patch_sampler = PatchSampler2D(patch_size=(128,128))
    patch_queue = PatchQueue(PET_dataset,
                         max_length=4,
                         samples_per_volume=1,
                         sampler=patch_sampler,
                         num_workers=0,
                         shuffle_subjects=True,
                         shuffle_patches=True)

    print("patch queue length:", len(patch_queue))

    patch_loader = DataLoader(patch_queue, batch_size=1)

    print("Testing patch loader length ...")
    counter = 0
    for batch_of_patchs in patch_loader:
        counter += 1
        print(counter)



def test_sequential_sampling():
    PET_CT_dataset = HECKTORPETCTDataset(data_dir,
                                patient_id_filepath,
                                mode='cval-CHMR-validation',
                                preprocessor=preprocessor,
                                input_representation='multichannel-volume',
                                augment_data=False)
    subject_dict = PET_CT_dataset[0]

    print("Testing 3D ...")
    patch_size_3d = (448,448,98)
    print("Total valid patches:", get_num_valid_patches(patch_size_3d))
    patch_sampler_3d = PatchSampler3D(patch_size=patch_size_3d, sampling='sequential')
    focal_pts = patch_sampler_3d._sample_valid_focal_points(subject_dict, get_num_valid_patches(patch_size_3d))
    print("Focal pts:", focal_pts)
    sample_patch = patch_sampler_3d.get_samples(subject_dict, num_patches=1)[0]
    print("Sample patch shape:", sample_patch['PET-CT'].shape, sample_patch['GTV-labelmap'].shape)

    print("\nTesting 2D ...")
    patch_size_2d = (448,448)
    print("Total valid patches:", get_num_valid_patches(patch_size_2d))
    patch_sampler_2d = PatchSampler2D(patch_size=patch_size_2d, sampling='sequential')
    focal_pts = patch_sampler_2d._sample_valid_focal_points(subject_dict, get_num_valid_patches(patch_size_2d))
    #print("Focal pts:", focal_pts)
    sample_patch = patch_sampler_2d.get_samples(subject_dict, num_patches=1)[0]
    print("Sample patch shape:", sample_patch['PET-CT'].shape, sample_patch['GTV-labelmap'].shape)


def test_focal_point_stride(focal_point_stride=3):

    PET_dataset = HECKTORUnimodalDataset(data_dir,
                                patient_id_filepath,
                                mode='cval-CHMR-validation',
                                preprocessor=preprocessor,
                                input_modality='PET',
                                augment_data=False)
    subject_dict = PET_dataset[0]

    print("focal_point_stride:", focal_point_stride)

    print("Testing 3D ...")
    patch_size_3d = (448,448,98)
    patch_sampler_3d = PatchSampler3D(patch_size=patch_size_3d, sampling='sequential', focal_point_stride=focal_point_stride)
    num_valid_patches = get_num_valid_patches(patch_size_3d, focal_point_stride=focal_point_stride)
    focal_pts = patch_sampler_3d._sample_valid_focal_points(subject_dict, num_valid_patches)
    print("Focal pts:", focal_pts)

    print("Testing 2D ...")
    patch_size_2d = (448,448)
    patch_sampler_2d = PatchSampler2D(patch_size=patch_size_2d, sampling='sequential', focal_point_stride=focal_point_stride)
    num_valid_patches = get_num_valid_patches(patch_size_2d, focal_point_stride=focal_point_stride)
    focal_pts = patch_sampler_2d._sample_valid_focal_points(subject_dict, num_valid_patches)
    #print("Focal pts:", focal_pts)



if __name__ == '__main__'

    test_focal_point_stride(focal_point_stride=3)