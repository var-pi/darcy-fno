from neuralop.models import FNO
from vis import plot_single_example_all_resolutions
import torch
from neuralop.data.datasets import load_darcy_flow_small

def main():
	train_loader, test_loaders, data_processor = load_darcy_flow_small(
		n_train=1000,
		batch_size=12,
		n_tests=[100, 100, 100, 100],
		test_resolutions=[16, 32, 64, 128],
		test_batch_sizes=[12, 12, 12, 12],
		encode_input=True,
		encode_output=True,
	)
	
	model = FNO.from_checkpoint(save_folder='./models/', save_name='example_fno')

	plot_single_example_all_resolutions(model, test_loaders, data_processor)

if __name__ == "__main__":
    main()
