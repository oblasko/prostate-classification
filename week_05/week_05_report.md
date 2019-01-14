# Tmux workflow

   1. Creating new tmux session `tmux new -s tensor`
   2. To reconnect into session `tmux attach -t tensor`
   3. To verify session is running `tmux ls`
   4. To activate tensorflow `source /scratch/oblasko/tensorflow/bin/activate`

# Running the mnist example
   To ensure that tensorflow and keras are working properly I ran the basic mnist example.
   Output can be found in file `mnist_output.txt`

   **Final Test loss:** 0.028055675415606312
   **Final Test accuracy:** 0.9902

# Running the prostate classification
 - Step 1: Splitting the data
`python3 split_images_by_patient.py 
        --mask-dir /archive/rnowling/medical-imaging/mcw-prostate/Mask_tif\ copy 
        --image-dir /archive/rnowling/medical-imaging/mcw-prostate/Images_tif\ copy 
        --output-dir /scratch/oblasko/mcw-prostate-split  
        --test-frac 0.25`
**OUTPUT:**
    Found 39 patients
    922 training patients
    313 testing patients

- Step 2: Importing the data (convert to numpy arrays)
`python3 import_data.py 
        --test-mask-dir /scratch/oblasko/mcw-prostate-split/test/masks 
        --test-image-dir /scratch/oblasko/mcw-prostate-split/test/images 
        --train-mask-dir /scratch/oblasko/mcw-prostate-split/train/masks 
        --train-image-dir /scratch/oblasko/mcw-prostate-split/train/images 
        --output-dir /scratch/oblasko/mcw-classification-model`
OUTPUT:
0 negative predictions
0 excluded from test
313 kept
922 training images
313 testing images

- Step 3: Running the classification model

`python classify_images.py \
       --input-dir /scratch/oblasko/mcw-classification-model \
       --output-dir /scratch/oblasko/mcw-classification-model \
       --num-epochs 100`

**Problem**
Running the split script more than once caused assigning patients multiple times and some patients appeared in the training and testing data at the same time, therefore the testing data was biased and that falsely enhanced accuracy of our model -- to ~0.98.
This problem was solved by deleting the `/scratch/oblasko/mcw-prostate-split` directory and running the split script only once. We then got our final model with these metrics:

Final Loss: 0.3868426152121145
Final Accuracy: 0.8913738019169329

Output can be found in the `prostate_classification.txt` file