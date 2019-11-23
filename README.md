# Tomato Allergies

All images could be by default in ./data/assignment_imgs or you can pass your own path in arg. 

## Requirements:

Simply `python3 -m pip install .`


## Usage:
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

 