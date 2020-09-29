#!/usr/local_rwth/bin/zsh


# Job configuration ---

#SBATCH --job-name=resample_113
#SBATCH --output=outputs/resample_113.%j.log

## OpenMP settings
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G

## Request for a node with 2 Tesla P100 GPUs
## #SBATCH --gres=gpu:pascal:2

#SBATCH --time=2:00:00

## TO use the UM DKE project account
## #SBATCH --account=um_dke


# Load CUDA 
#module load cuda

# Debug info
#echo; echo
#nvidia-smi
#echo; echo

# Execute script
python_interpreter="../../../maastro_env/bin/python3"
python_file="../tools/sitk_resampling.py"

$python_interpreter $python_file --source_dir "../../../Datasets/HECKTOR/hecktor_train/hecktor_nii" \
                                 --target_dir "../../../Datasets/HECKTOR/hecktor_train/rs113_hecktor_nii" \
                                 --new_spacing 1 1 3


#------------------------
# Note: All relative paths are relative to the directory containing the job script