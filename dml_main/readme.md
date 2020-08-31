

## The main steps on our updating:

1. add new criteria into the folder: criteria

2. update parameters.loss_specific_parameters for parameter setting

3. update criteria.__init__.py about the initialization of the newly added criteria



## Be aware the differences between traditional ltr and metric learning:

(1) for traditional ltr, the number of relevant documents is quite small;

(2) for metric learning, the numbers of images per class are relatively comparable.

(3) for metric learning, each class is treated relatively equal, and can be the positive documents.
In other words, there might be the mutual offset when we directly use the traditional ltr loss functions for metric learning.
In other words, metric learning is more like a kind of classification.

- Specturm-Regularization seems needed for ltr loss functions.

- for an efficient testing, it seems that we need colab-gpu with paid account;


## Notes during the process of customization

### Some important settings to be tuned
