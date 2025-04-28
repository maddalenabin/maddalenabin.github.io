---
title: 'Supervised vs unsupervised'
date: 2017-11-19
#modified: 
permalink: /machine-learning-glossary/concepts/supunsup
toc: false
excerpt: "ML concepts: supervised vs unsupervised learning."
header: 
  teaser: "blog/glossary/glossary.png"
tags:
  - ML
  - Glossary
author_profile: false
redirect_from: 
  - /posts/2017/11/glossary-supunsup
sidebar:
  title: "ML Glossary"
  nav: sidebar-glossary
---

{% include base_path %}



## Supervised learning


*Supervised learning tasks tackle problems that have labeled data.*

:bulb: <span class='intuition'> Intuition </span>: It can be thought of a teacher who corrects a multiple choice exam. At the end you will get your average result as well as the correct answer to any of the questions.

Supervised learning can be further separated into two broad type of problems:
* **Classification**: here the output variable $y$ is categorical. We are basically trying to assign one or multiple classes to an observation. Example: is it a cat or not ?
* **Regression**: here the output variable $y$ is continuous. Example : how tall is this person ?

<!-- ### Classification
*The classification problem consists of assigning a set of classes/categories to an observation. I.e* $$\mathbf{x} \mapsto y,\ y \in \{0,1,...,C\}$$

Classification problems can be further separated into:

* **Binary:** There are 2 possible classes. $$C=2,\ y \in \{0,1\}$$
* **Multi-Class:** There are more than 2 possible classes. $$C>2$$
* **Multi-Label:** If labels are not mutually exclusive. Often replaced by $$C$$ binary classification specifying whether an observation should be assigned to each class.

Common evaluation metrics include Accuracy, F1-Score, AUC... I have a [section devoted for these classification metrics](#classification-metrics). -->

## Unsupervised learning

*Unsupervised learning tasks consist in finding structure in **unlabeled** data without a specific desired outcome.*

Unsupervised learning can be further separated into multiple subtasks (the separation is not as clear as in the supervised setting):
* **Clustering**: can you find cluster in the data?
* **Clustering**: can you find cluster in the data ?
* **Density estimation**: what is the underlying probability distribution that gave rise to the data?
* **Dimensionality reduction** how to best compress the data?
* **Outlier detection** which data-point are outliers?

Due to the lack of ground truth labels, it difficult to measure the performance of such methods, but such methods are extremely important due to the amount of accessible unlabeled data.
