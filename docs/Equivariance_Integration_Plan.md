# Plan: Integrate SK Equivariance Loss with 3+3 Randomized Grid

Goal
- Replace the fixed 2x3 grid of 6 distinct views with a 3+3 grid: 3 original views and their transformed versions, shuffled in random order.
- The model does not know correspondences. Only the loss uses the hidden pairing to compute SK Equivariance on corresponding pairs.
- Keep existing SK sharpening loss; add equivariance loss term from StableKeypoints.

Relevant current code
- zero123plus/model.py
  - MVDiffusion.prepare_batch_data: builds grid from 6 views and returns cond_imgs, target_imgs (B,6,C,H,W), target_grids (B,C,3H,2W).
  - MVDiffusion.forward_unet_with_sk: collects attention into sk_ref_dict via SKAttnProc in pipeline.
  - MVDiffusion.compute_sk_losses_from_attention: aggregates attention and computes SK sharpening loss.
- zero123plus/pipeline.py
  - SKAttnProc stores raw SK attention scores per layer into sk_ref_dict[name].
  - scale_latents/image helpers.
- src/utils/attention_extraction.py
  - extract_sk_attention_auto_dimensions: splits 2x3 grid attention into per-view maps [6, view_spatial, 16].
- StableKeypoints/optimization/losses.py
  - equivariance_loss(embeddings_initial, embeddings_transformed, transform, index): uses transform.inverse and MSE.
  - sharpening_loss and utilities in StableKeypoints/utils/keypoint_utils.py.

High-level approach
1) Data: From the batch’s 6 views, sample 3 views. Create 3 transformed versions with a known invertible transform. Concatenate originals + transformed, randomize order into a 2x3 grid.
2) Book-keeping: Save a pairing map and the transform objects per pair for loss only (not visible to the model).
3) Train: Run as usual to collect SK attention. During loss, compute:
   - Sharpening on all 6 views (as today).
   - Equivariance only on corresponding (orig, transformed) pairs using the pre-applied known transform.
4) Weight and log the new equivariance loss with sk_loss_weights["equivariance"].

Detailed implementation steps

A) Build an invertible transform utility (new helper)
- Create a small wrapper that records parameters and provides:
  - forward(image_or_map: Tensor[B or K, H, W]) -> warped
  - inverse(image_or_map: Tensor[B or K, H, W]) -> warped back
- Backed by torchvision.transforms.v2.functional affine/grid_sample for: rotation, scale, translation, optional flip.
- Ensure consistent coordinate system for both images and attention maps (bilinear, align_corners=False, padding_mode="zeros").
- File: src/utils/equivariance.py (new)
  - class InvertibleAffine:
    - __init__(angle, translate, scale, shear=(0,0), flip=False)
    - forward(tensor, is_image_like=True)
    - inverse(tensor, is_image_like=True)
  - Factory: build_random_invertible_transform(H, W, cfg) -> InvertibleAffine with reasonable ranges.

B) Modify data preparation to create 3+3 randomized grid
- File: zero123plus/model.py, prepare_batch_data
  - From target_imgs (B,6,C,H,W): sample 3 indices per batch element without replacement.
  - For each selected view v, create transformed image v_t = T_v(v) using InvertibleAffine.forward (on image space).
  - Build a 6-tensor per sample: originals [3] + transformed [3].
  - Randomly permute the 6 and place into the 2x3 grid (shape remains (B,6,C,H,W) then rearranged to (B,C,3H,2W)).
  - Build and return an equivariance meta:
    - eq_meta = {
      "pairs": List[(orig_idx_in_grid, trans_idx_in_grid)] of length 3,
      "transforms": List[InvertibleAffine] aligned with pairs,
      "orig_src_indices": original view indices picked from the 6 (optional debug)
    }
- Update return signature to: cond_imgs, target_imgs_6, target_grids, eq_meta.
- Update training_step to receive eq_meta.

C) Propagate metadata to loss computation
- In training_step, after forward pass and sk_ref_dict collection, pass eq_meta to compute_sk_losses_from_attention:
  - Change signature to compute_sk_losses_from_attention(sk_ref_dict, eq_meta).
  - Backward compatibility: if eq_meta is None, skip eq term.

D) Extract per-view SK attention maps (already available)
- In compute_sk_losses_from_attention:
  - Use extract_sk_attention_auto_dimensions on each collected layer in sk_ref_dict.
  - Aggregate layers as done today to get avg_attention_maps: [6, view_spatial, 16].
  - Reshape to spatial [6, 16, H, W] when needed.
  - Normalize per keypoint with softmax over spatial for both sharpening and equivariance.

E) Compute Equivariance loss on corresponding pairs only
- For i in range(3):
  - (o_idx, t_idx) = eq_meta["pairs"][i]
  - T = eq_meta["transforms"][i]
  - Get attention maps:
    - A_o: [16, H, W] from view o_idx
    - A_t: [16, H, W] from view t_idx
  - Optional: select top keypoints to reduce compute and stabilize:
    - Use find_top_k_gaussian + furthest_point_sampling (StableKeypoints/utils/keypoint_utils.py) on A_o averaged across spatial to pick K indices.
    - Or reuse same top-k used by sharpening (keep K=10 by default).
  - Call losses.equivariance_loss:
    - embeddings_initial = A_o[selected]  # [K, H, W]
    - embeddings_transformed = A_t[selected]  # [K, H, W]
    - transform = T  # must expose inverse() over batched maps
    - index = 0 (we are single-process in this call; the SK signature expects index; you may ignore or pass 0)
  - Accumulate across the 3 pairs; average to get sk_equivariance_loss.
- Add to loss_dict under train/sk_equivariance_loss and to total_loss with weight sk_loss_weights["equivariance"].

F) Keep sharpening loss as-is
- Continue computing the sharpening loss on the aggregated views (all 6) using compute_sharpening_loss_batch.

G) Configuration and toggles
- Extend MVDiffusion.__init__ sk_loss_weights to include key "equivariance": e.g., default 0.1–0.5.
- Add flags:
  - use_equivariance_mode=True to enable 3+3 grid building.
  - equiv_top_k=10, sigma=1.0 (shared with sharpening) for stability.

H) Logging and monitoring
- Log train/sk_equivariance_loss, and normalized variant (divide by weight) in log_wandb_metrics.
- Optionally log sample visualizations: show the randomized grid order with overlays indicating which pair indices were used (debug only).

I) Validation/inference
- Validation stays with the default dataset behavior (no equivariance pairs). keep prepare_batch_data able to switch off equivariance mode during validation, or return eq_meta=None and compute only standard losses.

Edge cases and notes
- Ensure transforms applied to images are exactly mirrored on attention maps during loss.
- Use consistent resize/antialiasing; apply transforms before resizing to the training H,W to avoid inconsistencies.
- Maintain device/dtype for transforms and warps (match model device, float32).
- If heads/layers missing, guard compute_sk_losses_from_attention gracefully.

Testing checklist
1) Unit-test InvertibleAffine.forward/inverse on synthetic blobs; verify round-trip.
2) Visual check: build a mini-batch with annotated pairs; inspect the randomized grid and mapping.
3) Sanity: with zero equivariance weight, loss/training equals prior behavior.
4) Enable equivariance with simple transforms (small translation); ensure loss > 0 and decreases on overfitting batch.
5) Ensure no NaNs: add eps in softmax normalization for attention maps.

Milestones
- M1: Implement transform utility + data path to 3+3 grid with metadata (no loss changes).
- M2: Add eq loss computation using existing SK functions; wire weights and logging.
- M3: Ablation: test different K, transform ranges, and target layers for attention aggregation.
