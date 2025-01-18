# Coral Virtual Cam
 Experiments with the google coral edge TPU. Using image segmentation to create a greenscreen virtual cam for OBS

# FOR MAC USERS -- macARM branch 
*General Warning*
`DO NOT` pip install pycoral, its not right.
- `pip install git+https://github.com/google-coral/pycoral.git`
- This will collect the latest python coral API rather than some abandoned library


Make sure youve installed coral from site:
	- https://coral.ai/docs/accelerator/get-started/#runtime-on-mac


tflite mac runtime
- `pip install --upgrade --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime`

`JAN 2025 SAD DISCLAIMER`
gotta turn on a "sketchy" setting to use OBS virtual camera with python right now on mac. This could be limited to arm macs, haven't tested for intel. 

- Downgrade to OBS 29 (maybe), I had to some people seem to suggest 30 was supposed to fix this 
- Boot into recovery mode 
- Go into the bash terminal in there 
	- `system-override legacy-camera-plugins-without-sw-camera-indication=on`

## Installation

`Built and tested on python 3.10.15`, some versions of 3.9 and other versions of 3.10 may work but I cannot currently guarantee it. 

- [Install the Coral runtime for your system](https://coral.ai/docs/accelerator/get-started/#requirements)
- [Collect Pycoral API from THIS repo](https://github.com/google-coral/pycoral.git)
- [Grab the tflite runtime for your system](https://pypi.org/project/tflite-runtime/)

If not mentioned here you SHOULD be good to `pip install -r requirements.txt` in the root directory of the project. 

## Usage
Above all else this is just an exploration right now. With requirements installed and your coral setup, run any of the scripts with your coral plugged in to use them. 

All scripts other than virtual cam live in the `preflightchecks` directory inside `/src`, this is to keep things clean. 

- `Coral Test` -- Used to ensure that coral is setup and the models can be loaded
- `Camera Test` -- Simple script for ensuring python can use your system camera
- `Green Screen` -- Test script, uses Deeplabv3 to create a green screen effect on camera.
- `Virutal Cam` -- Virutal camera adapater for use with software like OBS. Splits person out of background with Deeplabv3 semantic segmentation. The script will read out status as it spins up, once running you can open up OBS (or whatever you set it up for) and view it. 

---

 <img width="736" alt="image" src="https://github.com/user-attachments/assets/fce7148e-3071-42bd-8f7a-4b41ae1cdab6" />
