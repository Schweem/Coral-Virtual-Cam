# Coral Virtual Cam
Experiments with the google coral edge TPU. Using image segmentation to create a greenscreen virtual cam for OBS. 
---

 *General Warnings*
`DO NOT` pip install pycoral, its not right.
- `pip install git+https://github.com/google-coral/pycoral.git`
- This will collect the latest python coral API rather than some abandoned library

I had generally poor results trying to install the tflite runtime on mac with pip, seemed okay on windows. 

`I also generally cannot guarantee wide compatability right now, I'd like to flesh that out with time. `

## Installation

`Built and tested on python 3.10.15`, some versions of 3.9 and other versions of 3.10 may work but I cannot currently guarantee it. 

- [Install the Coral runtime for your system](https://coral.ai/docs/accelerator/get-started/#requirements)
- [Collect Pycoral API from THIS repo](https://github.com/google-coral/pycoral.git)
- [Grab the tflite runtime for your system](https://pypi.org/project/tflite-runtime/)

If not mentioned here you SHOULD be good to `pip install -r requirements.txt` in the root directory of the project. 

## Usage
Above all else this is just an exploration right now. With requirements installed and your coral setup, run any of the scripts with your coral plugged in to use them. 
- `Coral Test` -- Used to ensure that coral is setup and the models can be loaded
- `Camera Test` -- Simple script for ensuring python can use your system camera
- `Green Screen` -- Test script, uses Deeplabv3 to create a green screen effect on camera.
- `Virutal Cam` -- Virutal camera adapater for use with software like OBS. Splits person out of background with Deeplabv3 semantic segmentation. The script will read out status as it spins up, once running you can open up OBS (or whatever you set it up for) and view it. 
