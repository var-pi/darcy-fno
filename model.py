from neuralop.models import FNO
import torch
import torch.nn as nn
import torch.optim as optim
from neuralop.data.datasets import load_darcy_flow_small
from neuralop.training import Trainer
from vis import plot_single_example_all_resolutions

def main():
	model = FNO(
		n_modes=(32, 32),
		hidden_channels=24,
		in_channels=1,
		out_channels=1
	)

	train_loader, test_loaders, data_processor = load_darcy_flow_small(
		n_train=1000,
		batch_size=12,
		n_tests=[100],
		test_resolutions=[16],
		test_batch_sizes=[12],
		encode_input=True,
		encode_output=True,
	)

	# Define optimizer
	optimizer = optim.Adam(model.parameters(), lr=1e-3)

	scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

	# Define loss functions
	train_loss = lambda output, **sample: nn.MSELoss()(output, sample['y'])
	eval_losses = {'mse': mse, 'pi_mse': train_loss}

	# Create the trainer
	trainer = Trainer(
		model=model,
		n_epochs=20,
		data_processor=data_processor,
		verbose=True
	)

	# train the model
	trainer.train(
		train_loader=train_loader,
		test_loaders=test_loaders,
		optimizer=optimizer,
		scheduler=scheduler,
		regularizer=False,
		training_loss=train_loss,
		eval_losses=eval_losses,
	)

	model.save_checkpoint(save_folder='./models', save_name='physics_informed_fno')	

if __name__ == "__main__":
    main()
