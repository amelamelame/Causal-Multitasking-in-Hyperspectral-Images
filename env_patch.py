"""
Patch for data/taiga_dataset.py — ensure env_id is always in batch dict.
The strip_mask.npy is optional; if absent, all pixels get env_id=0.
This patch is already integrated — no action needed.
Just confirms the __getitem__ returns env_id.
"""
# This file documents that env_id is included in batch output.
# The existing taiga_dataset.py already handles env_id via the
# strip_mask.npy file (7 flight strips in TAIGA).
# If strip_mask.npy is not available, env_id defaults to 0 for all pixels.
