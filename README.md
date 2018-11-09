
### Convert xml to csv
python3 object_detection/xml_to_csv.py
### Convert input data to TFR
python3 object_detection/generate_tfrecord.py --csv_input=images_meter/train_labels.csv --image_dir=images_meter/train --output_path=train.record

### Train
python3 object_detection/train.py --logtostderr --train_dir=object_detection/training/ --pipeline_config_path=object_detection/training/faster_rcnn_inception_v2_pets.config &

### Tensorboard
ssh -L 16006:127.0.0.1:6006 ubuntu@65.52.136.155
python3 /home/ubuntu/.local/lib/python3.5/site-packages/tensorboard/main.py --logdir=object_detection/training/

### Export inference graph
python3 object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path object_detection/training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix object_detection/training/model.ckpt-8316 --output_directory object_detection/inference_graph

### Predict
python3 object_detection/predict.py -m './object_detection/inference_graph/' -i './object_detection/test/meter3.jpg' -l './object_detection/labelmap.pbtxt' -o './object_detection/results'

### Requirements
tensorflow.__version__==1.9 <br>
cuda.__version__==9.0
cuDNN.__version__==7.0
