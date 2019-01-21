# Week 6 objectives

- [X] Split the images by patient
Firstly the images needed to be grouped by patient.
The implementation can be found in `classify_concated_images.py` file -- function `split_images_by_patient`

- [X] Sort the images by slice id for every patient
The implementation can be found in `classify_concated_images.py` file -- function `sort_by_slice`
The structure of the data after splitting and sorting can be found in the `sorted_splitted_images.txt`

- [X] Iterate over the images in a following fashion: 1-2-3, 2-3-4, 3-4-5, 4-5-6, ... and concate 3 images at a time
The implementation can be found in `classify_concated_images.py` file -- function `create_concated_imgs`
- Images have now (N_images,512,512,3) shape
- Masks (N_images,512,512,1) as in previous model

- [X] Generate labels from masks. 
The label is generated for every batch of 3 images from the middle mask.
The implementation can be found in `classify_concated_images.py` file -- function `generate_label`

- [X] Data scaling, data normalization, scaling masks
The implementation can be found in `classify_concated_images.py` file

- [X] Training the model
I managed to train the first model, but there's still room for improvement.
At this point the performance of the concated model is worse compared to the previous model taking one image at a time.
Final Accuracy - train data: 0.8290
Final Loss - train data:  0.3966
Final Accuracy - test data: 0.591
Final Loss -test data: 0.801

The output of training the model is to found in the file `training_concated_model.txt`