import warnings
warnings.filterwarnings("ignore")

import os
from glob import glob
from IPython.display import FileLink
from typing import Union, Tuple, Optional
import argparse
import numpy as np
import imageio.v3 as imageio
from matplotlib import pyplot as plt
from skimage.measure import label as connected_components
import pandas as pd 
import torch
from tqdm import tqdm

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.util.util import get_random_colors

import micro_sam.training as sam_training
from micro_sam.sample_data import fetch_tracking_example_data, fetch_tracking_segmentation_data
from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str)
parser.add_argument("--model", type=str, default= "vit_h")
parser.add_argument("--folder", type=str, default="all")
args = parser.parse_args()



root_dir = '/data/difrischiamm/seg_sam/micro-sam'

checkpoint_name = args.checkpoint
device = "cuda" if torch.cuda.is_available() else "cpu" 
model_type = args.model




DATA_FOLDER = os.path.join(root_dir, "data/ma+nf1")
best_checkpoint = os.path.join(root_dir, "combo_models", "checkpoints", checkpoint_name, "best.pt")



# ### Let's run the automatic instance segmentation (AIS)

# %%
def run_automatic_instance_segmentation(
    image: np.ndarray,
    checkpoint_path: Union[os.PathLike ,str],
    model_type: str = model_type,
    device: Optional[Union[str, torch.device]] = None,
    tile_shape: Optional[Tuple[int, int]] = None,
    halo: Optional[Tuple[int, int]] = None,
):
    """Automatic Instance Segmentation (AIS) by training an additional instance decoder in SAM.

    NOTE: AIS is supported only for `µsam` models.

    Args:
        image: The input image.
        checkpoint_path: The path to stored checkpoints.
        model_type: The choice of the `µsam` model.
        device: The device to run the model inference.
        tile_shape: The tile shape for tiling-based segmentation.
        halo: The overlap shape on each side per tile for stitching the segmented tiles.

    Returns:
        The instance segmentation.
    """
    # Step 1: Get the 'predictor' and 'segmenter' to perform automatic instance segmentation.
    predictor, segmenter = get_predictor_and_segmenter(
        model_type=model_type,  # choice of the Segment Anything model
        checkpoint=None, #checkpoint_path,  #! Write NONE if you want to use baseline SAM
        device=device,  # the device to run the model inference.
        is_tiled=(tile_shape is not None),  # whether to run automatic segmentation.
    )

    # Step 2: Get the instance segmentation for the given image.
    prediction = automatic_instance_segmentation(
        predictor=predictor,  # the predictor for the Segment Anything model.
        segmenter=segmenter,  # the segmenter class responsible for generating predictions.
        input_path=image,  # the filepath to image or the input array for automatic segmentation.
        ndim=2,  # the number of input dimensions.
        tile_shape=tile_shape,  # the tile shape for tiling-based prediction.
        halo=halo,  # the overlap shape for tiling-based prediction.
    )

    return prediction


# %%
assert os.path.exists(best_checkpoint), "Please train the model first to run inference on the finetuned model."
#assert train_instance_segmentation is True, "Oops. You didn't opt for finetuning using the decoder-based automatic instance segmentation."
import SimpleITK as sitk

#compute dice and hausdorff distance between two segmentations
def dice(seg1, seg2, label=1):
    """Compute the Dice coefficient between two segmentations."""
    overlapfilter = sitk.LabelOverlapMeasuresImageFilter()
    overlapfilter.Execute(seg1, seg2)
    dice = overlapfilter.GetDiceCoefficient(label)
    return dice

def hausdorff(seg1, seg2):
    """Compute the Hausdorff distance between two segmentations."""
    hausdorfffilter = sitk.HausdorffDistanceImageFilter()
    hausdorfffilter.Execute(seg1, seg2)
    return hausdorfffilter.GetHausdorffDistance()

unlabeled_dir = os.path.join(DATA_FOLDER, args.folder)#fetch_tracking_example_data(DATA_FOLDER)
unlabeled_paths = sorted(glob(os.path.join(unlabeled_dir, "*")))
# Let's check the first 5 images. Feel free to comment out the line below to run inference on all images.
#image_paths = image_paths[:5]

os.makedirs(os.path.join(root_dir, 'all_outputs', checkpoint_name), exist_ok = True)

out_df = pd.DataFrame(columns=['Name', 'Dice','Hausdorff'])

#predict on all images and save the results
for image_path in tqdm(unlabeled_paths):
    core = image_path.split('/')[-1]
    print(core)
    image = imageio.imread(image_path)

    prediction = run_automatic_instance_segmentation(
        image=image,
        checkpoint_path=best_checkpoint,
        model_type=model_type,
        device=device,
    )

    binary_mask = (prediction > 0).astype(np.uint8) * 255

    out_path = os.path.join(root_dir, 'all_outputs', checkpoint_name, core)
                            
    imageio.imwrite(out_path, binary_mask)

    
    print(f"image shape {image.shape} and prediction shape {prediction.shape} final shape {binary_mask.shape}")

out_df.to_csv(os.path.join(root_dir, 'all_outputs', checkpoint_name, "stats.csv"), index = False)

