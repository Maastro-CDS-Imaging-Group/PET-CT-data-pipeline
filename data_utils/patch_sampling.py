import random

from torch.utils.data import Dataset, DataLoader, RandomSampler


class PatchSampler():
    pass


class PatchQueue(Dataset):

    def __init__(self, dataset, max_length, samples_per_volume, sampler, num_workers, shuffle_subjects=True, shuffle_patches=True):
        self.dataset = dataset
        self.max_length = max_length
        self.samples_per_volume = samples_per_volume
        self.sampler = sampler
        self.num_workers = num_workers
        self.shuffle_subjects = shuffle_subjects
        self.shuffle_patches = shuffle_patches

        # Additional properties
        self.iterations_per_epoch = len(self.dataset) * self.samples_per_volume

        self.subject_samples_list = [] # List of dicts -- To store sets of full volumes of fetched patients
        self.patches_list = []  # List of dicts -- Main data structure implementing the patch queue


    def __len__(self):
        return self.iterations_per_epoch


    def __getitem__(self):

        if len(self.patches_list) == 0:
            self.fill_queue()

        sample_dict = self.patches_list.pop()
        return sample_dict


    def fill_queue(self): # TODO
        # Parallely fetch subjects

        # Sample patches from each one of them and populate the queue

        # Shuffle the queue
        pass
