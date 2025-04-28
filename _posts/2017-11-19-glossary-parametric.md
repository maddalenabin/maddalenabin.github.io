---
title: 'Parametric vs non parametric'
date: 2017-11-19
#modified: 
permalink: /machine-learning-glossary/concepts/parametric
toc: false
excerpt: "ML concepts: parametric vs non parametric."
header: 
  teaser: "blog/glossary/glossary.png"
tags:
  - ML
  - Glossary
author_profile: false
redirect_from: 
  - /posts/2017/11/glossary-parametric
sidebar:
  title: "ML Glossary"
  nav: sidebar-glossary
---

{% include base_path %}




These 2 types of methods distinguish themselves based on their answer to the following question: "Will I use the same amount of memory to store the model trained on $100$ examples than to store a model trained on $10 000$ of them ? "
If yes then you are using a *parametric model*. If not, you are using a *non-parametric model*.

+ **Parametric**:
  - :bulb: <span class='intuitionText'> The memory used to store a model trained on $100$ observations is the same as for a model trained on $10 000$ of them  </span>. 
  - I.e: The number of parameters is fixed.
  - :white_check_mark: <span class='advantageText'> Computationally less expensive </span> to store and predict.
  - :white_check_mark: <span class='advantageText'> Less variance. </span> 
  - :x: <span class='disadvantageText'> More bias.</span> 
  - :x: <span class='disadvantageText'> Makes more assumption on the data</span> to fit less parameters.
  - :school_satchel: <span class='example'> Example </span> : K-Means clustering, Linear Regression, Neural Networks:
  
  <div markdown="1">
  ![Linear Regression](/images/blog/glossary-old/Linear-regression.png){:width='300px'}
  </div>


+ **Non Parametric**: 
  - :bulb: <span class='intuitionText'> I will use less memory to store a model trained on $100$ observation than for a model trained on $10 000$ of them  </span>. 
  - I.e: The number of parameters is grows with the training set.
  - :white_check_mark: <span class='advantageText'> More flexible / general.</span> 
  - :white_check_mark: <span class='advantageText'> Makes less assumptions. </span> 
  - :white_check_mark: <span class='advantageText'> Less bias. </span> 
  - :x: <span class='disadvantageText'> More variance.</span> 
  - :x: <span class='disadvantageText'> Bad if test set is relatively different than train set.</span> 
  - :x: <span class='disadvantageText'> Computationally more expensive </span> as it has to store and compute over a higher number of "parameters" (unbounded).
  - :school_satchel: <span class='example'> Example </span> : K-Nearest Neighbors clustering, RBF Regression, Gaussian Processes:

  <div markdown="1">
  ![RBF Regression](/images/blog/glossary-old/RBF-regression.png){:width='300px'}
  </div>

:wrench: <span class='practice'> Practical </span> : <span class='practiceText'>Start with a parametric model</span>. It's often worth trying a non-parametric model if: you are doing <span class='practiceText'>clustering</span>, or the training data is <span class='practiceText'>not too big but the problem is very hard</span>.

:mag: <span class='note'> Side Note </span> : Strictly speaking any non-parametric model could be seen as a infinite-parametric model. So if you want to be picky: next time you hear a colleague talking about non-parametric models, tell him it's in fact parametric. I decline any liability for the consequence on your relationship with him/her :sweat_smile: . 

