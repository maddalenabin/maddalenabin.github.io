---
title: 'Generative vs discriminative'
date: 2017-11-19
#modified: 
permalink: /machine-learning-glossary/concepts/gendisc
toc: false
excerpt: "ML concepts: generative vs discriminative."
header: 
  teaser: "blog/glossary/glossary.png"
tags:
  - ML
  - Glossary
author_profile: false
redirect_from: 
  - /posts/2017/11/glossary-gendisc
sidebar:
  title: "ML Glossary"
  nav: sidebar-glossary
---

{% include base_path %}

These two major model types, distinguish themselves by the approach they are taking to learn. Although these distinctions are not task-specific task, you will most often hear those in the context of classification.

## Differences
{:.no_toc}

In classification, the task is to identify the category $y$ of an observation, given its features $\mathbf{x}$: $y \vert \mathbf{x}$. There are 2 possible approaches:

* **Discriminative** learn the *decision boundaries* between classes.
    * :bulb: <span class='intuitionText'> Tell me in which class is this observation given past data</span>. 
    * Can be **probabilistic** or **non-probabilistic** models. If probabilistic, the prediction is $\hat{y}=arg\max_{y=1 \ldots C} \, p(y \vert \mathbf{x})$. If non probabilistic, the model "draws" a boundary between classes, if the point $\mathbf{x}$ is on one side of of the boundary then predict $y=1$ if it is on the other then $y=2$ (multiple boundaries for multiple class).
    * Directly models what we care about: $y \vert \mathbf{x}$.
    * :school_satchel: As an example, for language classification, the  discriminative model would learn to <span class='exampleText'>distinguish between languages from their sound but wouldn't understand anything</span>.

* **Generative** model the *distribution* of each classes.
    * :bulb: <span class='intuitionText'> First "understand" the meaning of the data, then use your knowledge to classify</span>. 
    * Model the joint distribution $p(y,\mathbf{x})$ (often using $p(y,\mathbf{x})=p(\mathbf{x} \vert y)p(y)$). Then find the desired conditional probability through Bayes theorem: $p(y \vert \mathbf{x})=\frac{p(y,\mathbf{x})}{p(\mathbf{x})}$. Finally, predict $\hat{y}=arg\max_{y=1 \ldots C} \, p(y \vert \mathbf{x})$ (same as discriminative).
    * Generative models often use more assumptions to as t is a harder task.
    * :school_satchel: To continue with the previous example, the generative model would first <span class='exampleText'>learn how to speak the language and then classify from which language the words come from</span>.

## Pros / Cons
{:.no_toc}

Some of advantages / disadvantages are equivalent with different wording. These are rule of thumbs !

* **Discriminative**:
    <ul style="list-style: none;">
      <li > :white_check_mark: Such models <span class="advantageText">need less assumptions</span>  as they are tackling an easier problem. </li>
      <li > :white_check_mark: <span class="advantageText"> Often less bias => better if more data.</span> </li>
      <li > Often :x:<span class="disadvantageText"> slower convergence rate </span>. Logistic Regression requires $O(d)$ observations to converge to its asymptotic error. </li>
      <li > :x: <span class="disadvantageText"> Prone to over-fitting </span> when there's less data, as it doesn't make assumptions to constrain it from finding inexistent patterns.  </li>
      <li > Often :x: <span class="disadvantageText"> More variance. </span> </li>
      <li > :x: <span class="disadvantageText"> Hard to update the model </span> with new data (online learning). </li>
      <li > :x: <span class="disadvantageText"> Have to retrain model when adding new classes. </span> </li>
      <li > :x: <span class="disadvantageText"> In practice needs additional regularization / kernel / penalty functions.</span> </li>
    </ul >


* **Generative** 
  <ul style="list-style: none;">
      <li > :white_check_mark: <span class="advantageText"> Faster convergence rate => better if less data </span>. Naive Bayes only requires $O(\log(d))$ observations to converge to its asymptotic rate. </li>
      <li > Often :white_check_mark: <span class="advantageText"> less variance. </span> </li>
      <li > :white_check_mark: <span class="advantageText"> Can easily update the model  </span> with new data (online learning).  </li>
      <li > :white_check_mark: <span class="advantageText"> Can generate new data </span> by looking at $p(\mathbf{x} \vert y)$.  </li>
      <li > :white_check_mark: <span class="advantageText"> Can handle missing features</span> .  </li>
      <li > :white_check_mark: <span class="advantageText"> You don't need to retrain model when adding new classes </span>  as the parameters of classes are fitted independently.</li>
      <li > :white_check_mark: <span class="advantageText"> Easy to extend to the semi-supervised case. </span>  </li>
      <li > Often :x: <span class="disadvantageText"> more Biais. </span> </li>
      <li > :x: <span class="disadvantageText"> Uses computational power to compute something we didn't ask for.</span> </li>

    </ul >

:wrench: <span class='practice'> Rule of thumb </span>: If your need to train the best classifier on a large data set, use a **discriminative model**. If your task involves more constraints (online learning, semi supervised learning, small dataset, ...) use a **generative model**.

<div class="exampleBoxed" markdown="1">

Let's illustrate the advantages and disadvantage of both methods with an <span class='exampleText'> example </span> . Suppose we are asked to construct a classifier for the "true distribution" below. There are two training sets: "small sample" and "large sample". Suppose that the generator assumes point are generated from a Gaussian. 

<div style="display:flex;" markdown="1">
<div style="flex:1; padding-right:2%" markdown="1">
![discriminative vs generative true distribution](/images/blog/glossary-old/discriminative-generative-true.png)
</div>

<div style="flex:1; padding-right:2%" markdown="1">
![discriminative vs generative small sample](/images/blog/glossary-old/discriminative-generative-small.png)
</div>

<div style="flex:1; padding-right:2%" markdown="1">
![discriminative vs generative large sample](/images/blog/glossary-old/discriminative-generative-large.png)
</div>
</div>

How well will the algorithms distinguish the classes in each case ?

* **Small Sample**:
    * The *discriminative* model never saw examples at the bottom of the blue ellipse. It will not find the correct decision boundary there.
    * The *generative* model assumes that the data follows a normal distribution (ellipse). It will therefore infer the correct decision boundary without ever having seen data points there!


<div style="display:flex;" markdown="1">
<div style="flex:1; padding-right:2%" markdown="1">
![small sample discriminative](/images/blog/glossary-old/small-discriminative.png)
</div>

<div style="flex:1; padding-right:2%" markdown="1">
![small sample generative](/images/blog/glossary-old/small-generative.png)
</div>
</div>

* **Large Sample**:
    * The *discriminative* model is not restricted by assumptions and can find small red cluster inside the blue one.
    * The *generative* model assumes that the data follows a Gaussian distribution (ellipse) and won't be able to find the small red cluster.

<div style="display:flex;" markdown="1">
<div style="flex:1; padding-right:2%" markdown="1">
![large sample discriminative](/images/blog/glossary-old/large-discriminative.png)
</div>

<div style="flex:1; padding-right:2%" markdown="1">
![large sample generative](/images/blog/glossary-old/large-generative.png)
</div>
</div>

This was simply an example that hopefully illustrates the advantages and disadvantages of needing more assumptions. Depending on their assumptions, some generative models would find the small red cluster.
</div>

## Examples of Algorithms

### Discriminative
* Logistic Regression
* Softmax
* Traditional Neural Networks
* Conditional Random Fields
* Maximum Entropy Markov Model
* [Decision Trees](/machine-learning-glossary/models/trees)


### Generative
* [Naives Bayes](/machine-learning-glossary/models/naivebayes)
* Gaussian Discriminant Analysis
* Latent Dirichlet Allocation
* Restricted Boltzmann Machines
* Gaussian Mixture Models
* Hidden Markov Models 
* Sigmoid Belief Networks
* Bayesian networks
* Markov random fields

### Hybrid
* Generative Adversarial Networks

:information_source: <span class='resources'> Resources </span> : A. Ng and M. Jordan have a [must read paper](https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf) on the subject, T. Mitchell summarizes very well these concepts in [his slides](http://www.cs.cmu.edu/~ninamf/courses/601sp15/slides/07_GenDiscr2_2-4-2015.pdf), and section 8.6 of [K. Murphy's book](https://www.cs.ubc.ca/~murphyk/MLbook/) has a great overview of pros and cons, which strongly influenced the devoted section above.