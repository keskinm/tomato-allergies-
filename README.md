<h1>Tomato Allergies</h1>

<h2>Results with YOLOv3</h2>

![Training viz](https://i.ibb.co/0jD3T31/chart.png)

<h3>Evaluation specs</h3>

<h4>Data specs</h4>

Dataset is shuffled and split in 2100/450/450 (train/val/test).

Then the images containing tomato bbox (394 such images) in training are upsampled, 
and final training size is 4061 (1706 without tomatoes + 2355 with tomatoes images).

<h4>Yolo specs for training</h4>

`batch=32` ; `subdivisions=16` ; `width=640` ; `height=640` ; `momentum=0.9` ; `decay=0.0005` ; 
`learning_rate=0.001` ; `iterations=2500` 

<h4>Transfer learning</h4>
weights are initialized with darknet53 model trained on Imagenet. 

`wget https://pjreddie.com/media/files/darknet53.conv.74` to get it.

<h4>Error rate</h4>
Error rate on test set = 0.11 with checkpoint in release checkpoint_0.3. 

The detection threshold used is 0.15. 

Note: Overfitting is not fully attained and better ER is possible with more iterations.

<h2>Requirements installation</h2>
Simply `python3 -m pip install .`

<h2>Usage</h2>

<h3>Formatting</h3>

All images could be by default in ./data/assignment_imgs or you can pass your own path in arg. 

First transform the data in darknet format:

`python3 -m tomato_dataset_tool`

You could also upsample tomatoes for your training. Simply run instead: 

`python3 -m tomato_dataset_tool --upsample`

All new files are located in `./data/formated` (including gt text files for classification, i.e. if an image contains 
a tomato)

<h3>Train and test</h3>

Once it's done,
Install darknet: 

`python3 -m darknet_handler --install`

Training: 

`python3 -m darknet_handler --train --ckpts-file-path <path>`

Testing and look at error rate:

`python3 -m darknet_handler --test --ckpts-file-path <path>`

It will create `preds.txt` in darknet-master directory. Then use compute_metrics.py to get the error rate:

`python3 -m compute_metrics --yolo-output-filepath ./darknet-master/pre_metrics.txt --gt-filepath ./formated/test_gt.txt`




 