# Week 7 objectives

At our weekly monday meeting, Dr. Nowling noticed that the labels for the masks were generated incorrectly and that's probably the reason why the model performed so poorly. Also the fact that the starting point of generating the lables was second mask instead of the first mask was also an issue. My objectives for this week were to correct the issues mentioned above and to train the model with different dropout rates and different number of epochs and find out which parameter values are "optimal".

## Debugging the labels
Instead of generating the labels from the masks after the concatenating of the images, I decided to do that at the same time as the concatenation itself. This ensures that the labels are generated for the correct image batch as also simplifies the code. I got rid of the `generate_labels` function completely and added the logic into `create_concated_imgs` function. The implementation can be found in the `classify_concated_images.py` file in current folder.

## Training the model with different parameters
I trained the model with different dropout rates as also with different number of epochs to discover how different values can impact the accuracy of the model. The model was trained and evaluated with 5 different dropout rates( 0.0, 0.1, 0.2, 0.5, 0.75 ), all with default number of epochs - 100. Similarly with different number of epochs( 50, 100, 150, 200) with default dropout rate - 0.75. Outputs of the model trainings can be found in `./model-trainings` folder respectively.

### Model evaluation with different dropout rates
| Dropout rate        | Epochs | Accuracy -- testing data  |
| ------------- |:-------------:| -----:|
| 0.0      |  100 - default | 0.5942 |
| 0.1      | 100 - default      |   0.818 |
| 0.2 | 100 - default      |   0.866 |
| 0.5 | 100 - default     |    0.869 |
| 0.75 - default | 100 - default      |    0.859 |

### Model evaluation with different number of epochs
| Dropout rate        | Epochs | Accuracy -- testing data  |
| ------------- |:-------------:| -----:|
| 0.75 - default |  50       | 0.872   |
| 0.75 - default | 100 - default      |   0.859  |
| 0.75 - default | 150      |   0.830  |
| 0.75 - default | 200       |    0.837 |

### Conclusion
As it can be seen above in the tabels, the best performing value for dropout rate is 0.5 with accuracy of 0.869 and the model performs best when trained on 50 epochs( acc - 0.872 ). Let's try to combine those two parameters and see if the accuracy will improve.

**Model evaluation(50 epochs, 0.5 dropout rate) - training data:**
- Accuracy: 0.744
- Loss: 0.752

As we can see the model doesn't perform better when combining the two optimal values. The best performing values for dropout rate and number of epochs at this time are **default dropout rate = 0.75** and **number of epochs = 50** which give us the **accuracy of 0.872**.