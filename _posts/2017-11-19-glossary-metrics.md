---
title: 'Evaluation metrics'
date: 2017-11-19
#modified: 
permalink: /machine-learning-glossary/concepts/metrics
toc: false
excerpt: "ML concepts: evaluation metrics."
header: 
  teaser: "blog/glossary/glossary.png"
tags:
  - ML
  - Glossary
author_profile: false
redirect_from: 
  - /posts/2017/11/glossary-metrics
sidebar:
  title: "ML Glossary"
  nav: sidebar-glossary
---

{% include base_path %}
## Classification Metrics
### Single Metrics

:mag: <span class='note'> Side Notes </span> : The focus is on binary classification but most scores can be generalized to the multi-class setting. Often this is achieved by only considering "correct class" and "incorrect class" in order to make it a binary classification, then you average (weighted by the proportion of observation in the class) the score for each classes.

* **TP** / **TN** / **FN** / **FP:** Best understood through a $$2*2$$ [confusion matrix](#visual-metrics).

<div markdown="1">
![confusion matrix](/images/blog/glossary-old/confusion-matrix.png){:width='477px'}
</div>

* **Accuracy:** correctly classified fraction of observations. 
	*  $ Acc = \frac{Real Positives}{Total} = \frac{TP+FN}{TP+FN+TN+FP}$
	* :bulb: <span class="intuitionText"> In general, how much can we trust the predictions ? </span>
	* :wrench: <span class="practiceText"> Use if no class imbalance and cost of error is the same for both types  </span>
* **Precision** fraction of positive predictions that were actually positive. 
	* $ Prec = \frac{TP}{Predicted Positives} = \frac{TP}{TP+FP}$
	* :bulb: <span class="intuitionText"> How much can we trust positive predictions ? </span>
	* :wrench: <span class="practiceText"> Use if FP are the worst errors </span>
* **Recall** fraction of positive observations that have been correctly predicted. 
	* $ Rec = \frac{TP}{Actual Positives} = \frac{TP}{TP+FN}$
	* :bulb: <span class="intuitionText"> How many actual positives will we find? </span>
	* :wrench: <span class="practiceText"> Use if FN are the worst errors  </span>
* **F1-Score** harmonic mean (good for averaging rates) of recall and precision.
	* $F1 = 2 \frac{Precision * Recall}{Precision + Recall}$
    * If recall is $\beta$ time more important than precision use $F_{\beta} = (1+\beta^2) \frac{Precision * Recall}{\beta^2 Precision + Recall}$
	* :bulb: <span class="intuitionText"> How much to trust our algorithms for the positive class</span>
	* :wrench: <span class="practiceText"> Use if the positive class is the most important one (*i.e.* want a *detector* rather than a *classifier*)</span>

* **Specificity** recall for the negative negatives. 
    * $ Spec = \frac{TN}{Actual Negatives} = \frac{TN}{TN+FP}$
    
* **Log-Loss** measures performance when model outputs a probability $\hat{y_{ic}}$ that observation $n$ is in class $c$
	* Also called **Cross entropy loss** or **logistic loss**
	* $logLoss = - \frac{1}{N} \sum^N_{n=1} \sum^C_{c=1} y_{nc} \ln(\hat{y}_{nc})$
	* Use the natural logarithm for consistency
	* Incorporates the idea of probabilistic confidence
  * Log Loss is the metric that is minimized through Logistic Regression and more generally Softmax
  * :bulb: <span class="intuitionText"> Penalizes more if the model is confident but wrong (see graph below)</span>
  * :bulb: <span class="intuitionText"> Log-loss is the</span>  [cross entropy](/machine-learning-glossary/information/#cross-entropy) <span class="intuitionText"> between the distribution of the true labels and the predictions</span> 
  * :wrench: <span class="practiceText"> Use when you are interested in outputting confidence of results </span>
  * The graph below shows the log loss depending on the confidence of the algorithm that an observation should be classed in the correct category. For multiple observation we compute the log loss of each and then average them.

  <div markdown="1">
  ![log loss](/images/blog/glossary-old/log-loss.png){:width='477px'}
  </div>

* **Cohen's Kappa** Improvement of your classifier compared to always guessing the most probable class
  * $\kappa = \frac{accuracy - percent_{MaxClass}}{1 - percent_{MaxClass}}$
  * Often used to computer inter-rater (*e.g.* 2 humans) reliability: $\kappa = \frac{p_o- p_e}{1 - p_e}$ where $p_o$ is the observed agreement and $p_e$ is the expected agreement due to chance.
  * $ \kappa \leq 1$ (if $<0$ then useless).
  * :bulb: <span class='intuitionText'>Accuracy improvement weighted by class imbalance </span> .
  * :wrench: <span class='practiceText'> Use when high class imbalance and all classes are of similar importance</span>

* **AUC** **A**rea **U**nder the **C**urve. Summarizes curves in a single metric.
  * It normally refers to the [ROC](#visual-metrics) curve. Can also be used for other curves like the precision-recall one.
  * :bulb: <span class='intuitionText'> Probability that a randomly selected positive observation has is predicted with a higher score than a randomly selected negative observation </span> .
  * :mag: <span class='noteText'> AUC evaluates results at all possible cut-off points. It gives better insights about how well the classifier is able to separate between classes </span>. This makes it very different from the other metrics that typically depend on the cut-off threshold (*e.g.* 0.5 for Logistic Regression).
  * :wrench: <span class='practiceText'> Use when building a classifier for users that will have different needs (they could tweak the cut-off point)</span> . From my experience AUC is widely used in statistics (~go-to metric in bio-statistics) but less in machine learning.
  * Random predictions: $AUC = 0.5$. Perfect predictions: $AUC=1$.
 
### Visual Metrics

* **ROC Curve** : **R**eceiver **O**perating **C**haracteristic
  * Plot showing the TP rate vs the FP rate, over a varying threshold.
  * This plot from [wikipedia](https://commons.wikimedia.org/wiki/File:ROC_curves.svg) shows it well:
  
<div markdown="1">
![ROC curve](/images/blog/glossary-old/ROC.png){:width='477px'}
</div>

* **Confusion Matrix** a $C*C$ matrix which shows the number of observation of class $c$ that have been labeled $c', \ \forall c=1 \ldots C \text{ and } c'=1\ldots C$
    * :mag: <span class='noteText'> Be careful: People are not consistent with the axis :you can find real-predicted and predicted-real  </span> .
    * Best understood with an example:

    <div markdown="1">
    ![Multi Confusion Matrix](/images/blog/glossary-old/multi-confusion-matrix.png){:width='477px'}
    </div>

:information_source: <span class="resources"> Resources </span>: [Additional scores based on confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix)