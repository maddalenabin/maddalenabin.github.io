---
title: 'No free lunch'
date: 2017-11-19
#modified: 
permalink: /machine-learning-glossary/concepts/lunch
toc: false
excerpt: "ML concepts: no free lunch."
header: 
  teaser: "blog/glossary/glossary.png"
tags:
  - ML
  - Glossary
author_profile: false
redirect_from: 
  - /posts/2017/11/glossary-lunch
sidebar:
  title: "ML Glossary"
  nav: sidebar-glossary
---

{% include base_path %}

*There is no one model that works best for every problem.*

Let's try predicting the next fruit in the sequence:

<div class="centerContainer">
:tangerine: :apple: :tangerine: :apple: :tangerine: ...
</div>

You would probably say :apple: right ? Maybe with a lower probability you would say :tangerine: . But have you thought of saying :watermelon: ? I doubt it. I never told you that the sequence was constrained in the type of fruit, but naturally we make assumptions that the data "behaves well". <span class='intuitionText'> The point here is that without any knowledge/assumptions on the data, all future data are equally probable. </span> 

The theorem builds on this, and states that every algorithm has the same performance when averaged over all data distributions. So the average performance of a deep learning classifier is the same as random classifiers.

:mag: <span class='note'> Side Notes </span> :
* You will often hear the name of this theorem when someone asks a question starting with "what is the **best** [...] ?".
* In the real world, things tend to "behave well". They are for example often (locally) continuous. In such settings some algorithms are definitely better than others.
* Since the theorem publication in 1996, other methods have kept the lunch metaphor. For example: [kitchen sink](https://en.wikipedia.org/wiki/Kitchen_sink_regression) algorithms, [random kitchen sink](https://people.eecs.berkeley.edu/~brecht/papers/08.rah.rec.nips.pdf), [fastfood](https://arxiv.org/pdf/1408.3060.pdf), [à la carte](https://pdfs.semanticscholar.org/7e66/9999c097479c35e3f31aabdd2888f74b2e3e.pdf), and that's one of the reason why I decided to stick with fruit examples in this blog :wink:.
* The theorem has been extend to optimization and search algorithms.

:information_source: <span class='resources'> Resources </span> : D. Wolpert's [proof](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.390.9412&rep=rep1&type=pdf).