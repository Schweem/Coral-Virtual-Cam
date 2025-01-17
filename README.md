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
