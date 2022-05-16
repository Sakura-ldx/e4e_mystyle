import os

import dlib
from torch.utils.data import Dataset
from PIL import Image

from configs import paths_config
from utils import data_utils
from utils import alignment


class ImagesDataset(Dataset):

	def __init__(self, source_root, target_root, opts, target_transform=None, source_transform=None):
		if os.path.basename(source_root)[-3:] == 'txt' and os.path.basename(target_root)[-3:] == 'txt':
			with open(source_root, mode='r') as f:
				text = f.readlines()
				self.source_paths = [line.strip().split(' ')[0][1:] for line in text]
			f.close()
			with open(target_root, mode='r') as f:
				text = f.readlines()
				self.target_paths = [line.strip().split(' ')[0][1:] for line in text]
			f.close()
		else:
			self.source_paths = sorted(data_utils.make_dataset(source_root))
			self.target_paths = sorted(data_utils.make_dataset(target_root))
		self.source_transform = source_transform
		self.target_transform = target_transform
		self.opts = opts

	def __len__(self):
		return len(self.source_paths)

	def __getitem__(self, index):
		from_path = self.source_paths[index]
		from_im = Image.open(from_path)
		from_im = from_im.convert('RGB')

		to_path = self.target_paths[index]
		to_im = Image.open(to_path)
		to_im = to_im.convert('RGB')
		if self.target_transform:
			to_im = self.target_transform(to_im)

		if self.source_transform:
			from_im = self.source_transform(from_im)
		else:
			from_im = to_im

		return from_im, to_im
