#!/usr/bin/env python3
"""Main training script for Lyra-1.

This script handles the training pipeline for the Lyra multimodal language model,
supporting both single-GPU and multi-GPU training via Accelerate.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

logger = get_logger(__name__, log_level="INFO")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(description="Train Lyra-1 multimodal language model")

    # Model arguments
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from HuggingFace Hub.",
    )
    parser.add_argument(
        "--vision_tower",
        type=str,
        default=None,
        help="Path or identifier for the vision encoder.",
    )

    # Data arguments
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the training data JSON file.",
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        default=None,
        help="Root folder containing training images.",
    )

    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory to save model checkpoints and logs.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Total number of training epochs.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps before a backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Initial learning rate (after warmup).",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay coefficient for AdamW optimizer.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
        help="Fraction of total steps used for linear LR warmup.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine",
        choices=["linear", "cosine", "constant", "constant_with_warmup"],
        help="Learning rate scheduler type.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16 mixed precision training.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use float16 mixed precision training.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log training metrics every N steps.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save a checkpoint every N steps.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint directory to resume training from.",
    )

    args = parser.parse_args()
    return args


def setup_logging(accelerator: Accelerator) -> None:
    """Configure logging for the training process."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # Suppress verbose logs on non-main processes
    if not accelerator.is_main_process:
        logging.getLogger().setLevel(logging.WARNING)


def main() -> None:
    """Entry point for Lyra-1 training."""
    args = parse_args()

    # Determine mixed precision setting
    mixed_precision = "no"
    if args.bf16:
        mixed_precision = "bf16"
    elif args.fp16:
        mixed_precision = "fp16"

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with="tensorboard",
        project_dir=os.path.join(args.output_dir, "logs"),
    )

    setup_logging(accelerator)
    set_seed(args.seed)

    # Create output directory
    if accelerator.is_main_process:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Training configuration: {args}", main_process_only=True)
    logger.info(
        f"Distributed training: {accelerator.distributed_type}, "
        f"Num processes: {accelerator.num_processes}, "
        f"Device: {accelerator.device}",
        main_process_only=True,
    )

    # Verify CUDA availability for GPU training
    if not torch.cuda.is_available():
        logger.warning(
            "CUDA is not available. Training on CPU may be extremely slow."
        )

    logger.info(
        "Training pipeline initialised. Implement model, dataset, and "
        "optimiser setup in subsequent modules.",
        main_process_only=True,
    )

    accelerator.end_training()


if __name__ == "__main__":
    main()
