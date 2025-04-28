---
title: 'General ML concepts'
date: 2017-11-19
#modified: 
permalink: /machine-learning-glossary/concepts
toc: true
excerpt: "ML concepts: general."
header: 
  teaser: "blog/glossary/glossary.png"
tags:
  - ML
  - Glossary
redirect_from: 
  - /posts/2017/11/glossary-concepts/
author_profile: false
sidebar:
  title: "ML Glossary"
  nav: sidebar-glossary
---

{% include base_path %}





## Information Theory

### Information Content

Given a random variable $X$ and a possible outcome $x_i$ associated with a probability $p_X(x_i)=p_i$, the information content (also called self-information or surprisal) is defined as :

$$\operatorname{I} (p_i) = - log(p_i)$$

:bulb: <span class="intuition"> Intuition </span>: The "information" of an event is higher when the event is less probable. Indeed, if an event is not surprising, then learning about it does not convey much additional information.

:school_satchel: <span class='example'> Example </span> : "it is currently cold in Antarctica" does not convey much information as you knew it with high probability. "It is currently very warm in Antarctica" carries a lot more information and you would be surprised about it.

:mag: <span class="note"> Side note </span> : 

* $\operatorname{I} (p_i) \in [0,\infty[$
* Don't confuse the *information content* in information theory with the everyday word which refers to "meaningful information". <span class="exampleText"> A book with random letters will have more information content because each new letter would be a surprise to you. But it will definitely not have more meaning than a book with English words </span>.

### Entropy

<details open>
  <summary>Long Story Short</summary>
  <div markdown="1">
$$H(X) = H(p) \equiv \mathbb{E}\left[\operatorname{I} (p_i)\right] = \sum_{i=1}^K p_i \ \log(\frac{1}{p_i}) = - \sum_{i=1}^K p_i\  log(p_i)$$

:bulb: <span class="intuition"> Intuition </span>:

* The entropy of a random variable is the expected [information-content](#entropy). I.e. the <span class="intuitionText"> expected amount of surprise you would have by observing a random variable  </span>. 
* <span class="intuitionText"> Entropy is the expected number of bits (assuming $log_2$) used to encode an observation from a (discrete) random variable under the optimal coding scheme </span>. 


:mag: <span class="note"> Side notes </span> :

* $H(X) \geq 0$
* Entropy is maximized when all events occur with uniform probability. If $X$ can take $n$ values then : $max(H) = H(X_{uniform})= \sum_i^K \frac{1}{K} \log(\frac{1}{ 1/K} ) = \log(K)$

</div>
</details>

<p></p>


<details>
  <summary>Long Story Long</summary>
  <div markdown="1">
  
The concept of entropy is central in both thermodynamics and information theory, and I find that quite amazing. It originally comes from statistical thermodynamics and is so important, that it is carved on Ludwig Boltzmann's grave (one of the father of this field). You will often hear:

* **Thermodynamics**: *Entropy is a measure of disorder*
* **Information Theory**: *Entropy is a measure of information*

These 2 way of thinking may seem different but in reality they are exactly the same. They essentially answer: <span class="intuitionText"> how hard is it to describe this thing? </span>

I will focus here on the information theory point of view, because its interpretation is more intuitive for machine learning. I also don't want to spend to much time thinking about thermodynamics, as [people that do often commit suicide](http://www.eoht.info/page/Founders+of+thermodynamics+and+suicide) :flushed:.

$$H(X) = H(p) \equiv \mathbb{E}\left[\operatorname{I} (p_i)\right] = \sum_{i=1}^K p_i \ \log(\frac{1}{p_i}) = - \sum_{i=1}^K p_i\  log(p_i)$$

 In information theory there are 2 intuitive way of thinking of entropy. These are best explained through an <span class="example"> example </span> : 

<div class="exampleBoxed">
<div markdown="1">
:school_satchel: Suppose my friend [Claude](https://en.wikipedia.org/wiki/Claude_Shannon) offers me to join him for a NBA game (Cavaliers vs Spurs) tonight. Unfortunately I can't come, but I ask him to record who scored each field goals. Claude is very geeky and uses a binary phone which can only write 0 and 1. As he doesn't have much memory left, he wants to use the smallest possible number of bits.

1. From previous games, Claude knows that Lebron James will very likely score more than the old (but awesome :basketball: ) Manu Ginobili. Will he use the same number of bits to indicate that Lebron scored, than he will for Ginobili ? Of course not, he will allocate less bits for Lebron's buckets as he will be writing them down more often. He's essentially exploiting his knowledge about the distribution of field goals to reduce the expected number of bits to write down. It turns out that if he knew the probability $p_i$ of each player $i$ to score, he should encode their name with $nBit(p_i)=\log_2(1/p_i)$ bits. This has been intuitively constructed by Claude (Shannon) himself as it is the only measure (up to a constant) that satisfies axioms of information measure. The intuition behind this is the following:
	*  <span class="intuitionText"> Multiplying probabilities of 2 players scoring should result in adding their bits. </span> Indeed imagine Lebron and Ginobili have respectively 0.25 and 0.0625 probability of scoring the next field goal. Then, the probability that Lebron scores the 2 next field goals would be the same than Ginobili scoring a single one ($p(Lebron)*p(Lebron) = 0.25 * 0.25 = 0.0625 = Ginobili$). We should thus allocate 2 times less bits for Lebron, so that on average we always add the same number of bits per observation. $nBit(Lebron) = \frac{1}{2} * nBit(Ginobili) = \frac{1}{2} * nBit(p(Lebron)^2)$. The logarithm is a function that turns multiplication into sums as required. The number of bits should thus be of the form $nBit(p_i) = \alpha * \log(p_i) + \beta $
	* <span class="intuitionText"> Players that have higher probability of scoring should be encoded by a lower number of bits </span>. I.e $nBit$ should decrease when $p_i$ increases: $nBit(p_i) = - \alpha * \log(p_i) + \beta, \alpha > 0  $
	* <span class="intuitionText"> If Lebron had $100%$ probability of scoring, why would I have bothered asking Claude to write anything down ? I would have known everything *a priori* </span>. I.e $nBit$ should be $0$ for $p_i = 1$ : $nBit(p_i) = \alpha * \log(p_i), \alpha > 0  $

2. Now Claude sends me the message containing information about who scored each bucket. Seeing that Lebron scored will surprise me less than Ginobili. I.e Claude's message gives me more information when telling me that Ginobili scored. If I wanted to quantify my surprise for each field goal, I should make a measure that satisfies the following conditions:
	* <span class="intuitionText">The lower the probability of a player to score, the more surprised I will be </span>. The measure of surprise should thus be a decreasing function of probability: $surprise(x_i) = -f(p_i) * \alpha, \alpha > 0$.
	* Supposing that players scoring are independent of one another, it's reasonable to ask that my surprise seeing Lebron and Ginobili scoring in a row should be the same than the sum of my surprise seeing that Lebron scored and my surprise seeing that Ginobili scored. *I.e.* <span class="intuitionText"> Multiplying independent probabilities should sum the surprise </span>: $surprise(p_i * x_j) = surprise(p_i) + surprise(p_j)$.
	* Finally, <span class="intuitionText"> the measure should be continuous given probabilities </span>. $surprise(p_i) = -\log(p_{i}) * \alpha, \alpha > 0$

Taking $\alpha = 1 $ for simplicity, we get $surprise(p_i) = -log(p_i) =  nBit(p_i)$. We thus derived a formula for computing the surprise associated with event $x_i$ and the optimal number of bits that should be used to encode that event. This value is called information content $I(p_i)$. <span class="intuitionText">In order to get the average surprise / number of bits associated with a random variable $X$ we simply have to take the expectation over all possible events</span> (i.e average weighted by probability of event). This gives us the entropy formula $H(X) = \sum_i p_i \ \log(\frac{1}{p_i}) = - \sum_i p_i\  log(p_i)$

</div>
</div>

From the example above we see that entropy corresponds to : 
<div class="intuitionText">
<div markdown="1">
* **to the expected number of bits to optimally encode a message**
* **the average amount of information gained by observing a random variable** 
</div>
</div>

:mag: <span class="note"> Side notes </span> :

* From our derivation we see that the function is defined up to a constant term $\alpha$. This is the reason why the formula works equally well for any logarithmic base, indeed changing the base is the same as multiplying by a constant. In the context of information theory we use $\log_2$.
* Entropy is the reason (second law of thermodynamics) why putting an ice cube in your *Moscow Mule* (my go-to drink) doesn't normally make your ice cube colder and your cocktail warmer. I say "normally" because it is possible but very improbable : ponder about this next time your sipping your own go-to drink :smirk: ! 

:information_source: <span class="resources"> Resources </span>: Excellent explanation of the link between [entropy in thermodynamics and information theory](http://www.askamathematician.com/2010/01/q-whats-the-relationship-between-entropy-in-the-information-theory-sense-and-the-thermodynamics-sense/), friendly [ introduction to entropy related concepts](https://rdipietro.github.io/friendly-intro-to-cross-entropy-loss/)

</div>
</details>
<p></p>

### Differential Entropy
Differential entropy (= continuous entropy), is the generalization of entropy for continuous random variables.

Given a continuous random variable $X$ with a probability density function $f(x)$:

$$h(X) = h(f) := - \int_{-\infty}^{\infty} f(x) \log {f(x)} \ dx$$

If you had to make a guess, which distribution maximizes entropy for a given variance ? You guessed it : it's the **Gaussian distribution**.

:mag: <span class="note"> Side notes </span> : Differential entropy can be negative.

### Cross Entropy
We [saw that](#entropy) entropy is the expected number of bits used to encode an observation of $X$ under the optimal coding scheme. In contrast <span class="intuitionText"> cross entropy is the expected number of bits to encode an observation of $X$ under the wrong coding scheme</span>. Let's call $q$ the wrong probability distribution that is used to make a coding scheme. Then we will use $-log(q_i)$ bits to encode the $i^{th}$ possible values of $X$. Although we are using $q$ as a wrong probability distribution, the observations will still be distributed based on $p$. We thus have to take the expected value over $p$ :

$$H(p,q) = \mathbb{E}_p\left[\operatorname{I} (q_i)\right] = - \sum_i p_i \log(q_i)$$

From this interpretation it naturally follows that:
* $H(p,q) > H(p), \forall q \neq p$
* $H(p,p) = H(p)$

:mag: <span class="note"> Side notes </span> : Log loss is often called cross entropy loss, indeed it is the cross-entropy between the distribution of the true labels and the predictions.

### Kullback-Leibler Divergence
The Kullback-Leibler Divergence (= relative entropy = information gain) from $q$ to $p$ is simply the difference between the cross entropy and the entropy:

$$
\begin{align*} 
D_{KL}(p\|q) &= H(p,q) - H(p) \\
&= [- \sum_i p_i \log(q_i)] - [- \sum_i p_i \log(p_i)] \\
&= \sum_i p_i \log(\frac{p_i}{q_i})
\end{align*} 
$$

:bulb: <span class="intuition"> Intuition </span>

* KL divergence corresponds to the number of additional bits you will have to use when using an encoding scheme based on the wrong probability distribution $q$ compared to the real $p$ .
* KL divergence says in average how much more surprised you will be by rolling a loaded dice but thinking it's fair, compared to the surprise of knowing that it's loaded.
* KL divergence is often called the **information gain** achieved by using $p$ instead of $q$
* KL divergence can be thought as the "distance" between 2 probability distribution. Mathematically it's not a distance as it's none symmetrical. It is thus more correct to say that it is a measure of how a probability distribution $q$ diverges from an other one $p$.
	
KL divergence is often used with probability distribution of continuous random variables. In this case the expectation involves integrals:

$$D_{KL}(p \parallel q) = \int_{- \infty}^{\infty} p(x) \log(\frac{p(x)}{q(x)}) dx$$

In order to understand why KL divergence is not symmetrical, it is useful to think of a simple example of a dice and a coin (let's indicate head and tails by 0 and 1 respectively). Both are fair and thus their PDF is uniform. Their entropy is trivially: $H(p_{coin})=log(2)$ and $H(p_{dice})=log(6)$. Let's first consider $D_{KL}(p_{coin} \parallel p_{dice})$. The 2 possible events of $X_{dice}$ are 0,1 which are also possible for the coin. The average number of bits to encode a coin observation under the dice encoding, will thus simply be $log(6)$, and the KL divergence is of $log(6)-log(2)$ additional bits. Now let's consider the problem the other way around: $D_{KL}(p_{dice} \parallel p_{coin})$. We will use $log(2)=1$ bit to encode the events of 0 and 1. But how many bits will we use to encode $3,4,5,6$ ? Well the optimal encoding for the dice doesn't have any encoding for these as they will never happen in his world. The KL divergence is thus not defined (division by 0). The KL divergence is thus not symmetric and cannot be a distance.

:mag: <span class="note"> Side notes </span> : Minimizing cross entropy with respect to $q$ is the same as minimizing $D_{KL}(p \parallel q)$. Indeed the 2 equations are equivalent up to an additive constant (the entropy of $p$) which doesn't depend on $q$.

### Mutual Information

$$
\begin{align*} 
\operatorname{I} (X;Y) = \operatorname{I} (Y;X) 
&:= D_\text{KL}\left(p(x, y) \parallel p(x)p(y)\right) \\
&=  \sum_{y \in \mathcal Y} \sum_{x \in \mathcal X}
    { p(x,y) \log{ \left(\frac{p(x,y)}{p(x)\,p(y)} \right) }}
\end{align*} 
$$

:bulb: <span class="intuition"> Intuition </span>: The mutual information between 2 random variables X and Y measures how much (on average) information about one of the r.v. you receive by knowing the value of the other. If $X,\ Y$ are independent, then knowing $X$ doesn't give information about $Y$ so $\operatorname{I} (X;Y)=0$ because $p(x,y)=p(x)p(y)$. The maximum information you can get about $Y$ from $X$ is all the information of $Y$ *i.e.* $H(Y)$. This is the case for $X=Y$ : $\operatorname{I} (Y;Y)= \sum_{y \in \mathcal Y} p(y) \log{ \left(\frac{p(y)}{p(y)\,p(y)} \right) = H(Y) }$ 

:mag: <span class="note"> Side note </span> : 

* The mutual information is more similar to the concept of [entropy](#entropy) than to [information content](#information-content). Indeed, the latter was only defined for an *outcome* of a random variable, while the entropy and mutual information are defined for a r.v. by taking an expectation.
* $\operatorname{I} (X;Y) \in [0, min(\operatorname{I} (X), \operatorname{I} (Y;Y))]$
* $\operatorname{I} (X;X) =  \operatorname{I} (X)$
* $\operatorname{I} (X;Y) =  0 \iff X \,\bot\, Y$
* The generalization of mutual information to $V$ random variables $X_1,X_2,\ldots,X_V$ is the [Total Correlation](https://en.wikipedia.org/wiki/Total_correlation): $C(X_1, X_2, \ldots, X_V) := \operatorname{D_{KL}}\left[ p(X_1, \dots, X_V) \parallel p(X_1)p(X_2)\dots p(X_V)\right]$. It denotes the total amount of information shared across the entire set of random variables. The minimum $C_\min=0$ when no r.v. are statistically dependent. The maximum total correlation occurs when a single r.v. determines all the others : $C_\max = \sum_{i=1}^V H(X_i)-\max\limits_{X_i}H(X_i)$.
* Data Processing Inequality: for any Markov chain $X \rightarrow Y \rightarrow Z$: $\operatorname{I} (X;Y) \geq \operatorname{I} (X;Z)$
* Reparametrization Invariance: for invertible functions $\phi,\psi$: $\operatorname{I} (X;Y) = \operatorname{I} (\phi(X);\psi(Y))$

### Machine Learning and Entropy
This is all interesting, but why are we talking about information theory concepts in machine learning :sweat_smile: ? Well it turns our that many ML algorithms can be interpreted with entropy related concepts.

The 3 major ways we see entropy in machine learning are through:

* **Maximizing information gain** (i.e entropy) at each step of our algorithm. <span class="exampleText">Example</span>:
	
	* When building <span class="exampleText">decision trees you greedily select to split on the attribute which maximizes information gain</span> (i.e the difference of entropy before and after the split). Intuitively you want to know the value of the attribute, that decreases the randomness in your data by the largest amount.

* **Minimizing KL divergence between the actual unknown probability distribution of observations $p$ and the predicted one $q$**. <span class="exampleText">Example</span>:

	* The Maximum Likelihood Estimator (MLE) of our parameters $\hat{ \theta }_{MLE}$ <span class="exampleText"> are also the parameter which minimizes the KL divergence between our predicted distribution $q_\theta$ and the actual unknown one $p$ </span> (or the cross entropy). I.e 

$$\hat{ \theta }_{MLE} = arg\min_{ \theta } \, NLL= arg\min_{ \theta } \, D_{KL}(p \parallel q_\theta ) = arg\min_{ \theta } \, H(p,q_\theta ) $$

* **Minimizing  KL divergence between the computationally intractable $p$ and a simpler approximation $q$**. Indeed machine learning is not only about theory but also about how to make something work in practice.<span class="exampleText">Example</span>:

  - This is the whole point of <span class="exampleText"> **Variational Inference** (= variational Bayes) which approximates posterior probabilities of unobserved variables that are often intractable due to the integral in the denominator. Thus turning the inference problem to an optimization one</span>. These methods are an alternative to Monte Carlo sampling methods for inference (*e.g.* Gibbs Sampling). In general sampling methods are slower but asymptotically exact.


## No Free Lunch Theorem

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


## Parametric vs Non Parametric
These 2 types of methods distinguish themselves based on their answer to the following question: "Will I use the same amount of memory to store the model trained on $100$ examples than to store a model trained on $10 000$ of them ? "
If yes then you are using a *parametric model*. If not, you are using a *non-parametric model*.

+ **Parametric**:
  - :bulb: <span class='intuitionText'> The memory used to store a model trained on $100$ observations is the same as for a model trained on $10 000$ of them  </span>. 
  - I.e: The number of parameters is fixed.
  - :white_check_mark: <span class='advantageText'> Computationally less expensive </span> to store and predict.
  - :white_check_mark: <span class='advantageText'> Less variance. </span> 
  - :x: <span class='disadvantageText'> More bias.</span> 
  - :x: <span class='disadvantageText'> Makes more assumption on the data</span> to fit less parameters.
  - :school_satchel: <span class='example'> Example </span> : [K-Means](#k-means) clustering, [Linear Regression](#linear-regression), Neural Networks:
  
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
  - :school_satchel: <span class='example'> Example </span> : [K-Nearest Neighbors](#k-nearest-neighbors) clustering, RBF Regression, Gaussian Processes:

  <div markdown="1">
  ![RBF Regression](/images/blog/glossary-old/RBF-regression.png){:width='300px'}
  </div>

:wrench: <span class='practice'> Practical </span> : <span class='practiceText'>Start with a parametric model</span>. It's often worth trying a non-parametric model if: you are doing <span class='practiceText'>clustering</span>, or the training data is <span class='practiceText'>not too big but the problem is very hard</span>.

:mag: <span class='note'> Side Note </span> : Strictly speaking any non-parametric model could be seen as a infinite-parametric model. So if you want to be picky: next time you hear a colleague talking about non-parametric models, tell him it's in fact parametric. I decline any liability for the consequence on your relationship with him/her :sweat_smile: . 

