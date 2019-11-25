# Tomato Allergies


## Results with YOLOv3

![Training viz](https://i.ibb.co/0jD3T31/chart.png)

### Training specs:

#### data specs:
Dataset is shuffled and split in 2100/450/450 (train/val/test).

Then the images containing tomato bbox (394 such images) in training are upsampled, 
and final training size is 4061 (1706 without tomatoes + 2355 with tomatoes images).

#### yolov3 specs

`batch=32` ; `subdivisions=16` ; `width=640` ; `height=640` ; `momentum=0.9` ; `decay=0.0005` ; 
`learning_rate=0.001` ; `iterations=2500` 

### Error rate on test set = 0.11 with checkpoint in release checkpoint_0.3

Note: Overfitting is not fully attained and better ER is possible with 2500 iterations

## Requirements installation:

Simply `python3 -m pip install .`


## Usage:
All images could be by default in ./data/assignment_imgs or you can pass your own path in arg. 

First transform the data in darknet format:

`python3 -m tomato_dataset_tool`

You could also upsample tomatoes for your training. Simply run instead: 

`python3 -m tomato_dataset_tool --upsample`

Once it's done,
Install darknet: 

`python3 -m darknet_handler --install`

Training: 

`python3 -m darknet_handler --train --ckpts-file-path <path>`

Testing and look at error rate:

`python3 -m darknet_handler --test --ckpts-file-path <path>`




 