

# %%

#! USE THE sam ENVIRONMENT

import warnings
warnings.filterwarnings("ignore")

import os
from glob import glob
#from IPython.display import FileLink
from typing import Union, Tuple, Optional
import argparse
import numpy as np
import imageio.v3 as imageio
from matplotlib import pyplot as plt
from skimage.measure import label as connected_components
import pandas as pd 
import torch

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.util.util import get_random_colors

import micro_sam.training as sam_training
from micro_sam.sample_data import fetch_tracking_example_data, fetch_tracking_segmentation_data
from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation

# %% [markdown]
# ### Let's download the dataset

# %%
# NOTE: Please set 'root_dir' to the desired path, either where your data exists, or where you would like to download the data.
root_dir = '/data/difrischiamm/seg_sam/micro-sam'  # Overwrite this to point to your desired path.

DATA_FOLDER = os.path.join(root_dir, "data")
os.makedirs(DATA_FOLDER, exist_ok=True)

# This will download the image and segmentation data for training.
image_dir = os.path.join(DATA_FOLDER, 'ma+nf1', 'images')#fetch_tracking_example_data(DATA_FOLDER)
segmentation_dir = os.path.join(DATA_FOLDER, 'ma+nf1', 'masks')#= fetch_tracking_segmentation_data(DATA_FOLDER)


# %%
image_paths = sorted(glob(os.path.join(image_dir, "*")))
segmentation_paths = sorted(glob(os.path.join(segmentation_dir, "*")))

raw_key, label_key = "*.jpg", "*.png"


train_roi = np.s_[:10, :, :]
val_roi = np.s_[10:, :, :]



parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--root_dir", type=str, default="/data/difrischiamm/seg_sam/micro-sam")
parser.add_argument("--model", type=str, default= "vit_h")
parser.add_argument("--objects", type=int, default= 4)
parser.add_argument("--pred_model", type=str, default="vit_h")
parser.add_argument("--batch", type=int, default=3)
args = parser.parse_args()



batch_size = args.batch  # the training batch size
patch_shape = (1, 512, 512)  # the size of patches for training

train_instance_segmentation = True


sampler = MinInstanceSampler(min_size=25)  # NOTE: The choice of 'min_size' value is paired with the same value in 'min_size' filter in 'label_transform'.

train_loader = sam_training.default_sam_loader(
    raw_paths=image_dir,
    raw_key=raw_key,
    label_paths=segmentation_dir,
    label_key=label_key,
    with_segmentation_decoder=train_instance_segmentation,
    patch_shape=patch_shape,
    batch_size=batch_size,
    is_seg_dataset=True,
    rois=None,
    shuffle=True,
    raw_transform=sam_training.identity,
    sampler=sampler,
)

val_loader = sam_training.default_sam_loader(
    raw_paths=image_dir,
    raw_key=raw_key,
    label_paths=segmentation_dir,
    label_key=label_key,
    with_segmentation_decoder=train_instance_segmentation,
    patch_shape=patch_shape,
    batch_size=batch_size,
    is_seg_dataset=True,
    rois=None,
    shuffle=True,
    raw_transform=sam_training.identity,
    sampler=sampler,
)

# %%
# Let's check how our samples look from the dataloader
check_loader(train_loader, 4, plt=True)

# %% [markdown]
# ### Run the actual model finetuning

# %%
# All hyperparameters for training.
n_objects_per_batch = args.objects  # the number of objects per batch that will be sampled
device = "cuda" if torch.cuda.is_available() else "cpu"  # the device/GPU used for training
n_epochs = args.epochs # how long we train (in epochs)


model_type = args.model

base_ckpt_path = os.path.join(root_dir, "combo_models", "checkpoints", 'baseline', "best.pt")

torch.save(
    {
        'state_dict': util.statedict()
        'model_type': model_type,
        'epoch':0,
    },
    base_ckpt_path
)



# The name of the checkpoint. The checkpoints will be stored in './checkpoints/<checkpoint_name>'
checkpoint_name = f"2nf1+ma_sam_{model_type}_pred{args.pred_model}_epochs{n_epochs}_objects{n_objects_per_batch}_batch{batch_size}"

# %% [markdown]
# **NOTE**: The user needs to decide whether to finetune the Segment Anything model, or the `µsam`'s "finetuned microscopy models" for their dataset. Here, we finetune on the Segment Anything model for simplicity. For example, if you choose to finetune the model from the light microscopy generalist models, you need to update the `model_type` to `vit_b_lm` and it takes care of initializing the model with the desired weights)

# %%
# Run training
sam_training.train_sam(
    name=checkpoint_name,
    save_root=os.path.join(root_dir, "combo_models"),
    model_type=model_type,
    train_loader=train_loader,
    val_loader=val_loader,
    n_epochs=n_epochs,
    n_objects_per_batch=n_objects_per_batch,
    with_segmentation_decoder=train_instance_segmentation,
    device=device,
)

# %%
# Let's spot our best checkpoint and download it to get started with the annotation tool
best_checkpoint = os.path.join(root_dir, "combo_models", "checkpoints", checkpoint_name, "best.pt")


# %% [markdown]
# ### Let's run the automatic instance segmentation (AIS)

# %%
def run_automatic_instance_segmentation(
    image: np.ndarray,
    checkpoint_path: Union[os.PathLike ,str],
    model_type: str = args.pred_model,
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
        checkpoint=checkpoint_path,  # overwrite to pass your own finetuned model.
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
assert train_instance_segmentation is True, "Oops. You didn't opt for finetuning using the decoder-based automatic instance segmentation."
import SimpleITK as sitk

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

unlabeled_dir = os.path.join(DATA_FOLDER, 'ma+nf1', 'test/images')#fetch_tracking_example_data(DATA_FOLDER)
unlabeled_paths = sorted(glob(os.path.join(unlabeled_dir, "*")))
# Let's check the first 5 images. Feel free to comment out the line below to run inference on all images.
#image_paths = image_paths[:5]

os.makedirs(os.path.join(root_dir, 'combined_outputs', checkpoint_name), exist_ok = True)

out_df = pd.DataFrame(columns=['Name', 'Dice','Hausdorff'])

for image_path in unlabeled_paths:
    core = image_path.split('/')[-1]
    print(core)
    image = imageio.imread(image_path)
    mask = imageio.imread(os.path.join(root_dir, "data/ma+nf1/test/masks", core.split(".")[0] + ".png"))

    prediction = run_automatic_instance_segmentation(
        image=image,
        checkpoint_path=best_checkpoint,
        model_type=model_type,
        device=device,
    )
    print(f"image shape {image.shape} and prediction shape {prediction.shape}")
    binary_mask = (prediction > 0).astype(np.uint8) * 255
    #print(mask.max(), binary_mask.max())
    dice_val = dice(sitk.GetImageFromArray(mask), sitk.GetImageFromArray(binary_mask), 255)
    haus_val = hausdorff(sitk.GetImageFromArray(mask), sitk.GetImageFromArray(binary_mask))

    new_add = pd.DataFrame([{
        "Name": core,
        "Dice": dice_val,
        "Hausdorff": haus_val
    }])

    #print(dice_val, haus_val)
    out_df = pd.concat([out_df, new_add], ignore_index = True)
    # Visualize the predictions
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    fig.suptitle(f"Dice:{round(dice_val, 4)} Hausdorff:{round(haus_val, 4)}")

    ax[0].imshow(image, cmap="gray")
    ax[0].axis("off")
    ax[0].set_title("Input Image")

    #ax[1].imshow(prediction, cmap=get_random_colors(prediction), interpolation="nearest")
    ax[1].imshow(binary_mask, cmap='gray', interpolation="nearest")
    ax[1].axis("off")
    ax[1].set_title("Predictions (AIS)")

    plt.savefig(os.path.join(root_dir, 'combined_outputs', checkpoint_name, core))
    #Use SITK to save this image 
    plt.close()

out_df.to_csv(os.path.join(root_dir, 'combined_outputs', checkpoint_name, "stats.csv"), index = False)

#I want to see the training and validation curves
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
log_dir = os.path.join(root_dir, "combo_models", "logs", checkpoint_name)

event_files = [f for f in os.listdir(log_dir) if "tfevents" in f]
assert len(event_files) > 0, "No TensorBoard event files found in log_dir"

event_path = os.path.join(log_dir, event_files[-1])
print("Using event file:", event_path)

ea = event_accumulator.EventAccumulator(event_path)
ea.Reload()

# helper to convert TB scalar list → (steps, values)
def extract_scalar(ea, tag):
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values

train_steps, train_loss = extract_scalar(ea, "train/loss")
val_steps, val_loss = extract_scalar(ea, "validation/loss")
_, val_metric = extract_scalar(ea, "validation/metric")  # e.g. IoU or similar

plt.figure(figsize=(8, 5))
plt.plot(train_steps, train_loss, label="train loss")
plt.xlabel("iteration")
plt.ylabel("loss")
plt.legend()
plt.title("Training Loss")
plt.grid(True)
plt.savefig(os.path.join(root_dir, 'combined_outputs', checkpoint_name, 'train_loss.jpg'))

plt.figure(figsize=(8, 5))
plt.plot(val_steps, val_metric, label="validation metric")
plt.xlabel("iteration or epoch (depending on logging)")
plt.ylabel("metric")
plt.legend()
plt.title("Validation Metric")
plt.grid(True)
plt.savefig(os.path.join(root_dir, 'combined_outputs', checkpoint_name, 'train_loss.jpg'))

