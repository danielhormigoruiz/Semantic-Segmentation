## ===================================================================================================================================================================================== ##
##																ROAD SCENE SEGMENTATION BY USING VGG 16 PRE-TRAINED NETWORK																## 
##																-----------------------------------------------------------
## @package docstring
## <b>Documentation for Road Scene Segmetation Module. </b> 
## In this module we will develop a Road Scene Segmetation Module by using the VGG16 pre-trained network. The road scene segmentation is a algorithm that classify every pixel on a 
## given image, such as road, pedestrian, traffic sign, and so on.
##
## For this module you will need the following requirements:
## 	- Anaconda: They will allow you to use virtual environments to use Python 3 --> Anaconda for Python 3.6
## 	- Python 3: For this, you can download the following git and use a virtual environment --> https://github.com/udacity/CarND-Term1-Starter-Kit
## 	- TensorFlow: Minimum version 1.0. If your version is lower than 1.0, then run this command --> pip install tensorflow --upgrade
## 	- Numpy: It will be upgrated when TensorFlow is updated. If you have not Numpy, then run this command --> conda install -c anaconda numpy 
## 	- Scipy: It can be downloaded by the following command --> conda install -c anaconda scipy 
##
## <b>Folder structure </b>
##	
## \bInputs
##  	The inputs for this program will be images in which we want to apply road scene segmentantion.
##
## \bOutput
##  	A image with the same size as the input, but with the pixels already classified.
##
## <b>How to get the Kitty Dataset for Road Scene Segmentation? </b>
##	Click on the following link to start the downloading process: http://kitti.is.tue.mpg.de/kitti/data_road.zip
## ===================================================================================================================================================================================== ##

## ============================================================================= ##
## 								Import files									##
##								------------									##
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
from time import time

## ============================================================================= ##
## 								Global flag 									##
##								------------									##
save 				= False # This flag is for avoid errors when testing the functions, due to when training the network we need to save the models, and this 
							# send an error when we are testing the function.
testing_functions 	= False	# This flag is for launch the tests.
training 			= False # --> CHANGE THIS FLAG IF YOU WANT TO TRAIN THE NETWORK.
				 			#     If this flag is True, then we only run the test.
## ================================================================================	##
## 									PROGRAM											##
##									-------											##
helper.print_header_for("Requirements")
## Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

## Check for a GPU
if not tf.test.gpu_device_name():
	warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
	print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

## =====================================================================================================================	##
##											TEST 1. LOAD THE VGG 16 MODEL 												##
##											----------------------------- 												##
def load_vgg(sess, vgg_path):
	## Load Pretrained VGG Model into TensorFlow.
	## :param sess: TensorFlow Session
	## :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
	## :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
   
	## TODO: Implement function
	##   Use tf.saved_model.loader.load to load the model and weights

	## ----------------------------------------------------------------------------------------------------------------- ##
	## 1. Check if exists the .pb file for VGG16 on folder "VGG-model". In case it is not, then we need to transform it.##
	##
	files, nfiles = helper.find_files_with_format(format = "pb", dir_to_search = vgg_path)
	if nfiles == 0:
		warnings.warn("No pretrained VGG16 model found. Downloading pretrained VGG16 model ...")
		helper.maybe_download_pretrained_vgg(vgg_path)
		print("Downloading process done !")
	else:
		print("Pretrained model already exists on dir", vgg_path)

	## ----------------------------------------------------------------------------------------------------------------- ##
	## 2. Define the names of the different fields to get from the model. 												##
	## 																													##
	vgg_tag 					= 'vgg16'
	vgg_input_tensor_name 		= 'image_input:0'
	vgg_keep_prob_tensor_name 	= 'keep_prob:0'
	vgg_layer3_out_tensor_name 	= 'layer3_out:0'
	vgg_layer4_out_tensor_name 	= 'layer4_out:0'
	vgg_layer7_out_tensor_name 	= 'layer7_out:0'
	
	## ----------------------------------------------------------------------------------------------------------------- ##
	## 3. Get the fields from the model. 																				##
	## The tag to introduce on the <c> tf.saved_model.loader.load() function is the model 'vgg16' 																													##
	tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
	vgg_input_tensor 		= tf.get_default_graph().get_tensor_by_name(vgg_input_tensor_name)
	vgg_keep_prob_tensor 	= tf.get_default_graph().get_tensor_by_name(vgg_keep_prob_tensor_name)
	vgg_layer3_out_tensor 	= tf.get_default_graph().get_tensor_by_name(vgg_layer3_out_tensor_name)
	vgg_layer4_out_tensor 	= tf.get_default_graph().get_tensor_by_name(vgg_layer4_out_tensor_name)
	vgg_layer7_out_tensor 	= tf.get_default_graph().get_tensor_by_name(vgg_layer7_out_tensor_name)

	return vgg_input_tensor, vgg_keep_prob_tensor, vgg_layer3_out_tensor, vgg_layer4_out_tensor, vgg_layer7_out_tensor

##--------------------------------------	##
## TEST 1. Load the VGG 16 model process	##
##--------------------------------------	##
if testing_functions:
	helper.print_header_for("First test")
	tests.test_load_vgg(load_vgg, tf)
	#_ = input("First test passed ! Press [ENTER] to continue with the sencond one... ")

## =====================================================================================================================	##
##													TEST 2. LAYERS TEST													##
##													------------------- 												##
def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
	## 	Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
	## :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
	## :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
	## :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
	## :param num_classes: Number of classes to classify
	## :return: The Tensor for the last layer of output
	## TODO: Implement function

	## ************************************************************************************************ ##
	## HOW IS GOING TO BE DONE THE LAYERS CONNECTION ?
	## 
	## 			 		CONVOLUTIONAL (ENCODER)		  		 		DECONVOLUTIONAL (DECODER)
	##		|----------------------------------------------|   |-----------------------------------|
	##     ___ 																				   ___
	##    /	 /		__________ 									  			  ___________	  /	 /							
	##	 /__/ |	  /			 /									 			 /			/    /__/ |
	##	 |  | |	 /__________/ |				      1X1 CONV 					/__________/ |   |	| |
	##	 |  | |  |			| |    ________     __________      ________ 	|		   | |   |	| |
	##	 |	| |  |			| |	  /_______/|   /_________/ |   /_______/|	|		   | |	 |	| |	
	##	 |	|-|-||			| |  |	     | |  |			 |-|->|	      | |	|		   | |   |	| |
	##	 |	| | ||			| |  |		 | |  |__________|/	->|		  | |	|		   | |   |	| |
	##	 |	| | ||			| /  |_______|/					| |_______|/	|		   | /   |	| |
	##	 |	| | ||__________|/								|				|__________|/    |	| |
	##	 |__|/	 -------------------------------------------|								 |__|/
	##								SKIP UNION
	##
	## The skip will be done:
	##
	##				FROM 				TO 				SKIP UNION
	##			-----------------------------------------------------
	##			conv_layer_4		deconv_layer_1     skip_dec_1_layer_4
	##			conv_layer_3		deconv_layer_2     skip_dec_2_layer_3
	## ************************************************************************************************ ##

	## Define the Kernel initializer
	kernel_initializer = tf.random_normal_initializer(stddev=0.01)
	## Define the padding
	padding = 'SAME'

	## Define the kernel_size and stride for the convolutional layers
	conv_kernel_size = 1
	conv_stride = 1

	## Define the kernel_size and stride for the deconvolutional layers
	deconv_kernel_size = (4, 4)
	deconv_stride = (2, 2)

	## Define the kernel_size and stride for the last deconvolutional layer
	deconv_last_kernel_size = (16, 16)
	deconv_last_stride = (8, 8)
	## ============================================================================ ##
	## 										LAYERS 									##
	## 										------ 									##
	## Convolutional layer number 7
	conv_layer_7 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, 1, kernel_initializer = kernel_initializer)
	
	## Deconvolutional layer number 1
	deconv_layer_1 = tf.layers.conv2d_transpose(conv_layer_7, num_classes, (4, 4), (2, 2), padding, kernel_initializer = kernel_initializer)
	## Convolutional layer number 4
	conv_layer_4 = tf.layers.conv2d(vgg_layer4_out, num_classes,  1, 1, kernel_initializer = kernel_initializer)
	## Skip union 1
	skip_dec_1_layer_4 = tf.add(deconv_layer_1, conv_layer_4)

	## Deconvolutional layer number 2
	deconv_layer_2 = tf.layers.conv2d_transpose(skip_dec_1_layer_4, num_classes, (4, 4), (2, 2), padding, kernel_initializer = kernel_initializer)
	## Convolutional layer number 3
	conv_layer_3 = tf.layers.conv2d(vgg_layer3_out, num_classes,  1, 1, kernel_initializer = kernel_initializer)
	## Skip union 2
	skip_dec_2_layer_3 = tf.add(deconv_layer_2, conv_layer_3)

	## Deconvolutional layer number 3
	deconv_layer_3 = tf.layers.conv2d_transpose(skip_dec_2_layer_3, num_classes, (16, 16), (8, 8), padding, kernel_initializer = kernel_initializer)
	
	return deconv_layer_3

##--------------------------------------	##
## TEST 2. Layers test process   		##
##--------------------------------------	##
if testing_functions:
	helper.print_header_for("Second test")
	tests.test_layers(layers)
	#_ = input("Second test passed ! Press [ENTER] to continue with the sencond one... ")

## ===================================================================================================================  ##
##													TEST 3. OPTIMIZING TEST												##
##													-----------------------												##
def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
	## 	Build the TensorFLow loss and optimizer operations.
	## :param nn_last_layer: TF Tensor of the last layer in the neural network
	## :param correct_label: TF Placeholder for the correct label image
	## :param learning_rate: TF Placeholder for the learning rate
	## :param num_classes: Number of classes to classify
	## :return: Tuple of (logits, train_op, cross_entropy_loss)
	## TODO: Implement function

	## ================================================ 
	## Define here your train_function
	train_function = "ADAM"
	## ================================================ 

	## Logits: Reshape from 4D to 2D tensor
	logits = tf.reshape(nn_last_layer, (-1, num_classes))

	## Labels: Reshape from 4D to 2D tensor
	labels = tf.reshape(correct_label, (-1, num_classes))
		
	## Loss function --> optimize this
	cross_entropy_loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits( logits = logits, labels = labels ))
	
	## Training function definition
	if train_function == "SGD":
		train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy_loss)
	elif train_function == "ADAM":
		train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss) 
	else:
		print("The trainning function [{}] is not available.".format(train_function))
		exit(0)

	return logits, train_op, cross_entropy_loss

##--------------------------------------	##
## TEST 3. Optimizing test process 			##
##--------------------------------------	##
if testing_functions:
	helper.print_header_for("Third test")
	tests.test_optimize(optimize)
	#_ = input("Third test passed ! Press [ENTER] to continue with the fourth one... ")

## ===================================================================================================================  ##
##													TEST 4. TRAINING NN TEST											##
##													-----------------------												##
def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
			 correct_label, keep_prob, learning_rate):
	## 	Train neural network and print out the loss during training.
	## :param sess: TF Session
	## :param epochs: Number of epochs
	## :param batch_size: Batch size
	## :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
	## :param train_op: TF Operation to train the neural network
	## :param cross_entropy_loss: TF Tensor for the amount of loss
	## :param input_image: TF Placeholder for input images
	## :param correct_label: TF Placeholder for label images
	## :param keep_prob: TF Placeholder for dropout keep probability
	## :param learning_rate: TF Placeholder for learning rate
	## TODO: Implement function

	## Import needed packages
	import matplotlib.pyplot as plt
	import numpy as np

	# Create the model saver
	global save
	if save:
		saver = tf.train.Saver()

	## min_loss and loss vector definitions	
	min_loss = 2**34
	tloss = np.array([])

	## --------------------------------------------------------------------------- ##
	## --------------------------------------------------------------------------- ##
	##						    TRAINING PROCESS 								   ##
	## --------------------------------------------------------------------------- ##
	## --------------------------------------------------------------------------- ##

	## Training loop
	try:
		for epoch in range(epochs):
			it = 0
			for image, gt_image in get_batches_fn(batch_size):
				start = time()
				## Train and get the loss value
				_, train_loss = sess.run([train_op, cross_entropy_loss], feed_dict = {input_image: image,
																 				correct_label: gt_image,
																 				keep_prob: 0.5,
																 				learning_rate: 0.001})
				stop = time()

				## Add the training loss to the vector
				tloss = np.array(tloss, train_loss)

				## Print the loss
				print("[Epoch.", epoch, "] [It.", it, "] Loss:", train_loss, " -->", str(stop - start), "s")
				
				## Update the min_loss
				if save:
					if train_loss < min_loss:
						min_loss = train_loss
						saver.save(sess, 'model.ckpt')
						print ("Checkpoint saved")

				## Increment the iterator
				it += 1
	except KeyboardInterrupt:
		exit(0)

## -------------------------------------- ##
## TEST 4. Training NN test process		  ##
## -------------------------------------- ##
if testing_functions:
	helper.print_header_for("Fourth test")
	tests.test_train_nn(train_nn)
	print("Fourth test passed ! TESTING PROCESS IS OVER !! ")
	_ = input("Press [ENTER] to continue with the main program ")

def run():
	num_classes = 2
	image_shape = (160, 576)
	epochs = 40
	batch_size = 20
	data_dir = './data'
	runs_dir = './runs'
	## Path to vgg model
	vgg_path = os.path.join(data_dir, 'vgg')
	tests.test_for_kitti_dataset(data_dir)

	## Download pretrained vgg model
	helper.maybe_download_pretrained_vgg(data_dir)
	
	# Change the flag to True
	global save
	save = True
	
	## OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
	## You'll need a GPU with at least 10 teraFLOPS to train on.
	##  https://www.cityscapes-dataset.com/

	with tf.Session() as sess:

		## Create function to get batches
		print("Create function to get batches")
		get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

		## OPTIONAL: Augment Images for better results
		##  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

		## TODO: Build NN using load_vgg, layers, and optimize function
		print("Loading the VGG16 model... ")
		vgg_input_tensor, vgg_keep_prob_tensor, vgg_layer3_out_tensor, vgg_layer4_out_tensor, vgg_layer7_out_tensor = load_vgg(sess, vgg_path)
		print("VGG16 model loaded !")

		print("Creating the layers ... ")
		deconv3 = layers(vgg_layer3_out_tensor, vgg_layer4_out_tensor, vgg_layer7_out_tensor, num_classes)
		print("Layers created !")

		print("Creating the placeholder for labels and variable for learning rate ...")
		correct_label = tf.placeholder(dtype = tf.float32, shape = (None, None, None, num_classes))
		learning_rate = tf.placeholder(dtype = tf.float32) # can not convert float to tensor error
		print("Placeholders created !")

		print("Creating the logits, training operation and loss function for the network ...")
		logits, train_op, cross_entropy_loss = optimize(deconv3, correct_label, learning_rate, num_classes)
		print("Logits, training operation and loss function created !")

		# TODO: Train NN using the train_nn function
		sess.run(tf.global_variables_initializer())
		print("Variables initialized !")

		# We will train if training flag is true or there is no checkpoint saved
		if training or not(os.path.exists('checkpoint')):
			print("\n ===================================================== ")
			print(" Training ....")
			train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, vgg_input_tensor, correct_label, vgg_keep_prob_tensor, learning_rate)
			print(" Training completed !")

			# TODO: Save inference data using helper.save_inference_samples
			#  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
			print("\n ===================================================== ")
			print(" Testing ....")
			helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, vgg_keep_prob_tensor, vgg_input_tensor)
		else:
			# TODO: Save inference data using helper.save_inference_samples
			#  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
			saver = tf.train.Saver()
			saver.restore(sess, tf.train.latest_checkpoint(''))
			print("\n ===================================================== ")
			print(" Testing ....")
			helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, vgg_keep_prob_tensor, vgg_input_tensor)

		## OPTIONAL: Apply the trained model to a video

if __name__ == '__main__':
	run()
