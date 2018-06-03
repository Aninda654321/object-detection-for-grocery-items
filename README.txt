################################# INSTRUCTION TO RUN THE CODE AND RETRAIN THE MODEL ##############################
##################################################################################################################
##### HERE WE ARE PROVIDING THE INSTRUCTION TO RUN THE CODE AND ALSO PACKAGES IMPORTANT AND ALL OTHER        #####
##### IMPORTANT INFORMATION.                                                                                 #####
##################################################################################################################
##################################################################################################################

Download the full TensorFlow object detection repository located at https://github.com/tensorflow/models by clicking the 
“Clone or Download” button and downloading the zip file. Open the downloaded zip file and extract the “models-master”
folder directly into the C:\tensorflow1 directory you just created. Rename “models-master” to just “models”. (Note, this 
tutorial was done using this GitHub commit of the TensorFlow Object Detection API. If portions of this tutorial do not work, it
may be necessary to download and use this exact commit rather than the most up-to-date version.)


#############################################################
########## STEPS NEED TO DONE BEFORE RUNNING THE CODE #######
#############################################################

STEP1:
 We first have to copy the repository from Github what mentioned above 

STEP2:
 After copying the repository follow the steps mentioned below.

STEP3:
  Open your python from command prompt and run the following command
   
   $ C:\> conda create -n tensorflow1 pip python=3.5   
   
   Then, activate the environment by issuing:
   
   $ C:\> activate tensorflow1
   
STEP4:
 Install tensorflow-gpu in this environment by issuing:
 
   $ (tensorflow1) C:\> pip install --ignore-installed --upgrade tensorflow-gpu 
   
   ***NOTE: if your system does not have NVDIA GPU then please run the following command
   
   $ (tensorflow1) C:\>pip install tensorflow  
   
 Install the other necessary packages by issuing the following commands:
 
   $ (tensorflow1) C:\> conda install -c anaconda protobuf
   $ (tensorflow1) C:\> pip install pillow
   $ (tensorflow1) C:\> pip install lxml
   $ (tensorflow1) C:\> pip install Cython
   $ (tensorflow1) C:\> pip install jupyter
   $ (tensorflow1) C:\> pip install matplotlib
   $ (tensorflow1) C:\> pip install pandas
   $ (tensorflow1) C:\> pip install opencv-python
   
 *** Note: The ‘pandas’ and ‘opencv-python’ packages are not needed by TensorFlow, but they are used in the Python scripts to generate TFRecords and to work with images
 
STEP5:
 After this set the path to the to the \models, \models\research, and \models\research\slim by using PYTHONPATH variable 
 
   $ set PYTHONPATH=C:\tensorflow1\models;C:\tensorflow1\models\research;C:\tensorflow1\models\research\slim
   
   *** NOTE: Each time the virtual environment has been triggered this path need to set .
   
 
 
##################################################### 
########### RETRAINING STEPS ######################## 
#####################################################


STEP1: 
 Make sure previous four steps are done correctly (STEP1: - STEP4: from the beginning of this text).
 
STEP2:
 Delete as per instruction given in the following folders
 
 1. The “test_labels.csv” and “train_labels.csv” files in \object_detection\images
 2. All files in \object_detection\inference_graph
 3. All files except labelmap.pbtxt and faster_rcnn_inception_v2_pets.config 
 4. Add new images for training and test located at in \object_detection\images and replace with new images 
 
STEP3:
  compile the Protobuf files, which are used by TensorFlow to configure model and training parameters.
  
  $ protoc --python_out=. .\object_detection\protos\anchor_generator.proto .\object_detection\protos\argmax_matcher.proto .\object_detection\protos\bipartite_matcher.proto .\object_detection\protos\box_coder.proto .\object_detection\protos\box_predictor.proto .\object_detection\protos\eval.proto .\object_detection\protos\faster_rcnn.proto .\object_detection\protos\faster_rcnn_box_coder.proto .\object_detection\protos\grid_anchor_generator.proto .\object_detection\protos\hyperparams.proto .\object_detection\protos\image_resizer.proto .\object_detection\protos\input_reader.proto .\object_detection\protos\losses.proto .\object_detection\protos\matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .\object_detection\protos\model.proto .\object_detection\protos\optimizer.proto .\object_detection\protos\pipeline.proto .\object_detection\protos\post_processing.proto .\object_detection\protos\preprocessor.proto .\object_detection\protos\region_similarity_calculator.proto .\object_detection\protos\square_box_coder.proto .\object_detection\protos\ssd.proto .\object_detection\protos\ssd_anchor_generator.proto .\object_detection\protos\string_int_label_map.proto .\object_detection\protos\train.proto .\object_detection\protos\keypoint_box_coder.proto .\object_detection\protos\multiscale_anchor_generator.proto .\object_detection\protos\graph_rewriter.proto
 
  *** NOTE: activate tensorflow1 before run these steps  
 
STEP4:
  run the following commands from the C:\tensorflow1\models\research directory

  $ (tensorflow1) C:\tensorflow1\models\research> python setup.py build
  $ (tensorflow1) C:\tensorflow1\models\research> python setup.py install  
  
  *** NOTE: activate tensorflow1 before run these steps 
STEP5:
 To generate the TFRecords that serve as input data to the TensorFlow training model
 
  $ (tensorflow1) C:\tensorflow1\models\research\object_detection> python xml_to_csv.py
  
  *** NOTE: to run this go to tensorflow1\models\research\object_detection folder with tensorflow1 activated 
  
STEP6:
 If you added any new class to the image change the following 
 
 1. Within tensorflow1\models\research\object_detection folder "generate_tfrecord.py" add new levels (line 31-69) 
 2. within tensorflow1\models\research\object_detection\training folder change the new levels as per new one in lebelmap.pbtxt file 
 3. within tensorflow1\models\research\object_detection\training folder in faster_rcnn_inception_v2_pets.config Change num_classes(line 9) to the number of different objects you want the classifier to detect.
    and also change num_examples(line 128) to the number of images you have in the \images\test directory.
	
STEP7: 
 From the \object_detection directory, issue the following command to begin training:
 
  $  python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config
  
  *** NOTE: above one need to be executed with tensorflow1 activated 
  
STEP8: 
 After training done (it will take time as per the hardware configuration of the system and for better result run the traing atleast for 20-30K steps untill the classification loss goes down , again based on the training data set)
 export inference graph by issuing following command
 
  $ python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph
  
  *** NOTE: here XXXX in model.ckpt-XXXX will be replaced with the highest-numbered .ckpt file in the tensorflow1\models\research\object_detection\training folder

STEP9:
 Now go to tensorflow1 folder where the repository has been copied from there models\research\object_detection folder.
 within that tensorflow1 environment run following code code 

   $ (tensorflow1) C:\tensorflow1\models\research\object_detection>python object_detection_image.py
   
  before running object_detection_image.py change WRITE_DIR (line 42) as per your convenience 
  and also IMAGE_NAME (line 35) and DIR (line 38) as per your directory 