# Paper: Classification before Segmentation: More Accurate Prostate Segmentation with U-Net Summary

## Introduction
- Prostate segmentation is a necessary pre-processing step for detection and diagnosis algorithms for prostate disorders and cancers
- Using a U-net model on a prostate segmentation task
- Using a new data set of T2-weighted MR images from 39 patients
- U-net model struggles to perform both classification and segmentation in a single step => motivation for developing classifier for pre-filtering images before segmentation


## Methods
### Data set
- The data set consisted of 39 patients with prostate cancer( mean age 60 years )
- Only T2-weighted images were considered in the pipeline
- The data set was stratified randomly by patient into training - 66% and training - 33% sets
- Ground truth prostate masks were drawn by a single observer

### Segmentation model
- U-Net model was evaluated on combined classification and segmentation tasks as well as pure segmentation tasks
- Pixel intensities of the images were preprocessed to normalize
the pixel intensities to a mean of 0 and standard deviation of 1
- The model was trained using a crossentropy loss function and
the ADAM optimizer
- Images were not augmented
- Dice scores, recall, precision, and the number of connected components were calculated for each predicted mask

### Classification model
- trained and evaluated a Logistic Regression model on a classification task
- features for the LR model are generated from the predicted masks of the U-Net-all model 
- the masks are embedded into a 5D space using UMAP non-linear dimensionality-reduction technique
- The ratio of the segmented area to total image is used as a sixth feature
- Cross-fold validation was used to evaluate the classifier
- Patients were divided into 4 folds, each fold was used to evalaute a model trained on the remaining 3 folds
- The model achieved: 
                       - **AUC of 97.7%**
                       - **Accuracy 93.6%**
                       - **Recall 89.2%**
                       - **Precision 94.9%**

### Integrated pipeline
**3-stage pipeline:**
  1. Classification
  2. Segmentation
  3. Post-processing
   
 147 of 411 testting set images contained prostates => The 147 images were  segmented with the U-Net-prostate model; empty masks were generated for the remaining 264 images => Post processing the masks to remove artifacts 

### U-net-all vs. U-net-prostate
- U-net-all
  - Dice score - 0.758
  - Precision - 67.6%
  - Recall 85.6%

- U-net-prostate (with probability threshold of 0.999)
  - Dice score - 0.907
  - Precision - 83.2%
  - Recall - 83.4%

### Conclusion
Solution of separating classification from segmentation improved the model significantly.
U-net model can learn to distinguish between images with and without prostates but not enough to output empty masks for images without prostates.
