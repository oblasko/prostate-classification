## Week 1-2 report
My tasks for week 1 and week 2 were:
  - [X] Environment setup
  - [X] Access and load the provided data

I'll be using Dr. Nowling's machine, because training neural networks on my machine will be too slow. The connection between my machine and Dr. Nowling's can be established by using ssh. The confirmation of environment setup can be found in `keras_output.txt` file here in repository -- it's the output of keras example file mnist_cnn.py that was run remotely on Dr. Nowling's machine.

The data -- Prostate MRI scans from 39 patients were provided by Medical College of Wisconsin. Confirmation that the images were successfully loaded can be found in `loading_images_output.txt` file. Firstly I organized the images by patient id and then grouped the mask with corresponding image. 