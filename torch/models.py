import torch
import numpy as np
import torch.nn as nn

from toolkit.pytorch_transformers.architectures.unet import UNet

CUSTOM_NETWORKS = {'VGG11': {'model': UNet11,
                                 'model_config': {'pretrained': True},
                                 'init_weights': False}
                  }


class UNet(Model):
	"""
		UNet Implementation
	"""

	def __init__(self, architecture_config, training_config):
		super().__init__(architecture_config, training_config)
		self.activation_function = self.architecture_config['model_params']['activation']
		self.set_model()


	def set_model(self):
		encoder = self.architecture_config['model_params']['encoder']
		if (encoder == 'Default'):
			self.model = UNet(**self.architecture_config['model_params'])
		else:
			config = CUSTOM_NETWORKS[encoder]
			self.model = config['model'](num_classes=self.architecture_config['model_params']['out_channels'],
										 **config['model_config'])
			self._initialize_model_weights = lambda: None

		