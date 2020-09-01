

## The main steps on adding a new criteria:

> add new criteria into the folder: ltr_criteria

> update criteria.__init__.py for initialization of the newly defined criteria

> update parameters.loss_specific_parameters for parameter setting

> test the criteria using dml_main.dml_eval.py


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


## changes made on main.py in order to make it runnable
- set the arguments, such as data path

create a new file: dml_eval.py based on main.py

1> revise {import datasets      as datasets} to {import datasets      as dataset_module} due to naming rules
{import metrics       as metrics} to {import metrics       as metric_module}

2> define the run() function, without the block of INPUT ARGUMENTS
