# %%
import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# run this as python3 tutorials/mamba_train_example.py
# i.e. from the root directory
from sae_lens.training.config import LanguageModelSAERunnerConfig

cfg = LanguageModelSAERunnerConfig(
    # Data Generating Function (Model + Training Distibuion)
    model_name="tinystories_1Layer_21M_32k_ar",
    model_class_name="HookedTransformer",
    # hook_point="blocks.0.hook_mlp_out",
    hook_point="blocks.0.hook_resid_pre",
    hook_point_layer=0,
    # hook_point_eval="blocks.1.hook_ssm_output",  # we compare this when replace hook_point activations with autoencode.decode(autoencoder.encode( hook_point activations))
    d_in=1024,
    dataset_path="/home/sboughorbel/Projects/NeuroX/code/tinystories/tinystories_arabic_tokenized_valid/",  # this is a tokenized language dataset on Huggingface for the Tiny Stories corpus.
    # dataset_path="NeelNanda/openwebtext-tokenized-9b",
    is_dataset_tokenized=True,
    mse_loss_normalization=None,  # 
    # SAE Parameters
    expansion_factor=16,
    b_dec_init_method="geometric_median",
    # Training Parameters
    lr=0.0008,
    l1_coefficient=0.001,
    lr_scheduler_name="constant",
    train_batch_size=128,
    context_size=1024,
    lr_warm_up_steps=50,
    # Activation Store Parameters
    n_batches_in_buffer=32,
    training_tokens=1000_000,
    store_batch_size=16,
    # Dead Neurons and Sparsityq
    use_ghost_grads=False,
    feature_sampling_window=1000,
    dead_feature_window=1000,
    dead_feature_threshold=1e-4,
    # WANDB
    log_to_wandb=True,
    wandb_project="sae_training_custom_llama",
    # wandb_entity=None,
    wandb_log_frequency=100,
    # Misc
    device="cuda",
    seed=42,
    n_checkpoints=1,
    checkpoint_path="/home/sboughorbel/Projects/NeuroX/code/SAELens-Fork/checkpoints/",
    dtype=torch.float32,
)

from sae_lens.training.lm_runner import language_model_sae_runner

language_model_sae_runner(cfg)

# %%
