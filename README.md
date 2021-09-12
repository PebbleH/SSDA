# SSDA

[Code for paper "Constructing Domain Adaptive Semantic Segmentation with Statistical-Structural Priors".](https://github.com/PebbleH/SSDA)



#### Software:

Python(3.6) and Pytorch(1.3.1) is necessary before running the scripts. 



### Preparing Datasets

To validate the network, this repo use the  [Cityscapes]() as the target domain dataset.

To monitor the convergence of the network, we test on Cityscapes valdiation set.
Datalists are in the ```./dataset/cityscapes_list/```.

Put dataset in the`./data/Cityscapes` floder.




## testing

Downloaded the final model  [trained model](https://drive.google.com/file/d/13-zTdC_kS-O0roi7gjYCSPlN1jsqgSPO/view?usp=sharing)

Put the model in the`./snapshots/SSDA` floder.

```
python test.py
```



The complete code will be uploaded after the paper is accepted.

