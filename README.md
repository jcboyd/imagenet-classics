# imagenet-classics

[![AlexNet demo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1c6zBIIW6iSf5dXtLDhPYrqm2YFoR2rix?usp=sharing)

![prediction](https://github.com/jcboyd/imagenet-classics/assets/8886851/aadac30b-84fb-4e7c-92c1-519e9e193a9b)

Train AlexNet (76% top-5 validation accuracy):

```
python main.py ILVSRC2012 \
               ILVSRC2012/list \
               AlexNet_BN \
               --img_size 224 \
               --nb_epochs 12 \
               --nb_batch 256 \
               --lr 0.01 \
               --decay 0.0005
```

Train baseline logistic regression (7% top-5 validation accuracy):

```
python main.py ILVSRC2012/ \
               ILVSRC2012/list \
               LogisticRegression \
               --img_size 56 \
               --nb_epochs 1 \
               --nb_batch 2048 \
               --lr 0.1 \
               --decay 0.00001
```
