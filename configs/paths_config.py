dataset_paths = {
	#  Face Datasets (In the paper: FFHQ - train, CelebAHQ - test)
	'ffhq': '/home/ssd1/Database/FFHQ/',
	'celeba_test': '/home/ssd1/Database/celebA/Img/Img/',

	#  Cars Dataset (In the paper: Stanford cars)
	'cars_train': '',
	'cars_test': '',

	#  Horse Dataset (In the paper: LSUN Horse)
	'horse_train': '',
	'horse_test': '',

	#  Church Dataset (In the paper: LSUN Church)
	'church_train': '',
	'church_test': '',

	#  Cats Dataset (In the paper: LSUN Cat)
	'cats_train': '',
	'cats_test': '',

	'PMT_train': './PMT_dataset/inversion_train.txt',
	'PMT_test': './PMT_dataset/inversion_test.txt'
}

model_paths = {
	'stylegan_ffhq': 'pretrained_models/stylegan2-ffhq-config-f.pt',
	'ir_se50': 'pretrained_models/model_ir_se50.pth',
	'shape_predictor': 'pretrained_models/shape_predictor_68_face_landmarks.dat',
	'my_shape_predictor': 'pretrained_models/lms.dat',
	'moco': 'pretrained_models/moco_v2_800ep_pretrain.pth'
}
