# Mask_detector

Project based on a multi-face detector together with a mask detector.

![camera example](example.gif)

## Installation

### OS

- Ubuntu 18.04.5 LTS

### Requirements for executing Python file

- cuda 11.0
- python 3.8.6
- thorch 1.6.0
- fastai 2.0.16
- cv2 4.4.0
- numpy 1.18.5
- argparse 1.1
- PIL 7.2.0

### Extra requirements

- pandas 1.1.3
- matplotlib 3.3.1
- bs4 4.9.3

## Usage

Execute the `mask_detector.py` file using flag to select to execution mode:

- c: select webcam to use
- p: locate the photo to analyze
- s: name under which the image will be save

#### Examples:

```python mask_detector.py -c 0```

```python mask_detector.py -p image.png -s image_analized.png```