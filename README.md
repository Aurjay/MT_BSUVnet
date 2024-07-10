## Overview

This is an inference script for BSUVnet implementation of the Master thesis 'Comparison of Different Anomaly-Detection-Algorithms and Neural Network Driven Background-Subtraction Models to Detect Unexpected Objects in Video Streams'
## Dependencies

Install all the packages in requirements.txt file. The code was executed in a Python 3.10.7 envirionment.

## Setup
```bash
pip install -r requirements.txt
```

# or
```bash
conda install --file requirements.txt
```


## Usage
Download the pretrained weights from [BSUVnet GitHub repository](https://github.com/ozantezcan/BSUV-Net-2.0) and specify the appropriate folder in `trained_models/BSUV-Net-2.0.mdl`.

To run the script:

```bash
python inference.py
```

## Reference
This inference code has been refenced and modified from [BSUVnet GitHub repository](https://github.com/ozantezcan/BSUV-Net-2.0)
