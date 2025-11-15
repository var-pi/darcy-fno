import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_single_example_all_resolutions(model, test_loaders, data_processor, device='cpu'):
	"""
	Plots ground truth, prediction, and difference for a single input 
	across multiple resolutions.
	"""
	model.eval()
	model.to(device)

	n_res = len(test_loaders)
	fig, axes = plt.subplots(3, n_res, figsize=(4 * n_res, 12))

	# Compute global vmin and vmax across resolutions for GT/pred
	all_gt = []
	all_pred = []
	all_diff = []

	with torch.no_grad():
		# We'll take the first example in each resolution
		for col_idx, res_idx in enumerate(test_loaders):
			test_loader = test_loaders[res_idx]
			it = iter(test_loader)
			next(it)
			next(it)
			sample = next(it)  # take first batch

			x = sample['x'][0:1].to(device)  # take first example in batch
			x = data_processor.in_normalizer.transform(x)
			y_true = sample['y'][0:1].to(device)
			y_pred = model(x)
			y_pred = data_processor.out_normalizer.inverse_transform(y_pred)

			# Convert to 2D numpy arrays
			y_true_np = y_true.squeeze().cpu().numpy()
			y_pred_np = y_pred.squeeze().cpu().numpy()
			diff_np = y_true_np - y_pred_np

			all_gt.append(y_true_np)
			all_pred.append(y_pred_np)
			all_diff.append(diff_np)

	vmin = min([min(np.min(a) for a in all_gt), min(np.min(a) for a in all_pred)])
	vmax = max([max(np.max(a) for a in all_gt), max(np.max(a) for a in all_pred)])
	max_abs = max(np.max(np.abs(diff_np)) for diff_np in all_diff)
	vmin_diff, vmax_diff = -max_abs, max_abs

	for col_idx, (y_true_np, y_pred_np, diff_np) in enumerate(zip(all_gt, all_pred, all_diff)):
		# Plot Ground Truth
		ax = axes[0, col_idx] if n_res > 1 else axes[0]
		im = ax.imshow(y_true_np, cmap='viridis', vmin=vmin, vmax=vmax)
		ax.set_title(f'Res {y_true_np.shape[-1]}x{y_true_np.shape[-2]}')
		ax.axis('off')
		if col_idx == n_res - 1:
			plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

		# Plot Prediction
		ax = axes[1, col_idx] if n_res > 1 else axes[1]
		im = ax.imshow(y_pred_np, cmap='viridis', vmin=vmin, vmax=vmax)
		ax.axis('off')
		if col_idx == n_res - 1:
			plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

		# Plot Difference
		ax = axes[2, col_idx] if n_res > 1 else axes[2]
		im = ax.imshow(diff_np, cmap='bwr', vmin=vmin_diff, vmax=vmax_diff)
		ax.axis('off')
		if col_idx == n_res - 1:
			plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

	plt.tight_layout()
	plt.show()
