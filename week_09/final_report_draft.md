# Final report


## Week report links
:link: [week 1](https://github.com/oblasko/prostate-classification/tree/master/week_02)
:link: [week 2](https://github.com/oblasko/prostate-classification/tree/master/week_02)
:link: [week 3](https://github.com/oblasko/prostate-classification/tree/master/week_03)
:link: [week 4](https://github.com/oblasko/prostate-classification/tree/master/week_05)
:link: [week 5](https://github.com/oblasko/prostate-classification/tree/master/week_05)
:link: [week 6](https://github.com/oblasko/prostate-classification/tree/master/week_06)
:link: [week 7](https://github.com/oblasko/prostate-classification/tree/master/week_07)
:link: [week 8](https://github.com/oblasko/prostate-classification/tree/master/week_08)
:link: [week 9]()
:link: [week 10]()
:link: [week 11]()


## Base model
As a starting point Dr. Nowling has provided me with a model that takes one slice at a time. The ultimate goal for this independent study was to try come up with alternative models that could improve the classification step in the pipeline.
#### Base model final metrics
Loss: 0.387
Accuracy: 0.891

## First proposed model
The first model that Dr. Nowling and I have proposed was a model that instead of taking one slice at a time as a input, takes 3 slices concatenated together. The label is generated from the mask of the middle slice(0 -> no prostate, 1 -> contains prostate).

### Pre-processing data
Before the actual training of the model, the data -- images needed to be pre-processed. First we split and grouped the images by patient. Then we sorted the images by slice_id( ascending order )in every patient's group, iterated over every patient group and created new training and testing images by concatenating 3 images at the time in the following fashion: [slice_1, slice_2, slice_3], [slice_2, slice_3, slice_4], [slice_3, slice_4, slice_5], ... . Padding technique was used for the first and last image, like so: [black_img, slice_1, slice_2], [ slice_n-1, slice_n, black_img]. The dimension of the images was changed from (n_images, 512, 512, 1) to (n_images, 512, 512, 3). 

### Generating labels
The labels were generated from the masks. Every group of 3 concatenated images was labeled by the mask of the middle image. Example: the label for the image group [slice_1, slice_2, slice_3] was generated from the mask that corresponds to slice_2:
0 -> no white pixels in the image(no prostate), 1 -> some white pixels in the image(contains prostate)

### Training & testing split
The images were stratified randomly into training(~75%) and testing set(~25%) by patient before the actual concatenation.

### Training the model
The first model was trained on 100 epochs with dropout_rate = 0.75 and evaluated on the testing data.
#### Metrics:
- Loss: 0.633
- Accuracy: 0.859

### Finding the optimal parameters
In order to find the optimal dropout rate, the model was firstly trained on 100 epochs with different dropout rates.
#### Table 1: evaluation of different dropout rates
| Dropout rate        | Epochs | Accuracy -- testing data  |
| ------------- |:-------------:| -----:|
| 0.0      |  100 - default | 0.5942 |
| 0.1      | 100 - default      |   0.818 |
| 0.2 | 100 - default      |   0.866 |
| 0.5 | 100 - default     |    0.869 |
| 0.75 - default | 100 - default      |    0.859 |

We find out that the model performs best with dropout rate = 0.5 ( **acc = 0.869** ).
Secondly we trained the model on a different number of epochs in order to find the optimal value for it.

#### Table 2: evaluation of different number of epochs
Epochs | Accuracy -- testing data  |
| ------------- |:-------------:|
| 50      |  0.744 |
| 100      | 0.869 |
| 150 | 0.827 |   
| 200 | 0.840 |    

We concluded that the model reaches the highest accuracy of **0.869** when trained on the default number of epochs  - 100 and with dropout rate = 0.5.

#### First model final metrics
Loss: 0.602
Accuracy: 0.869



