# Week 8 objectives

As discovered in the last week, the model performs best with dropout rate equal to 0.5( trained on default 100 epochs ). Let's now see how it performs when the model is trained on a different number of epochs.

## Model evaluation with dropout rate - 0.5
Epochs | Accuracy -- testing data  |
| ------------- |:-------------:|
| 50      |  0.744 |
| 100      | 0.869 |
| 150 | 0.827 |   
| 200 | 0.840 |    

As we can see the model performs best if trained on 100 epochs( **acc = 0.869** ).
The best performing model at this time is still the model trained on 50 epochs and with dropout rate = 0.75( **acc = 0.872** ).