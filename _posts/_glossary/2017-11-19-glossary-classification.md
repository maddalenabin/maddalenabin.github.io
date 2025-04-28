---
title: 'Classification'
date: 2017-11-19
modified: 2019-02-18
permalink: /machine-learning-glossary/supervised/classification
toc: true
excerpt: "ML concepts: classification."
header: 
  teaser: "blog/glossary/glossary.png"
tags:
  - ML
  - Supervised
  - Glossary
redirect_from: 
  - /posts/2017/11/glossary-supervised/
author_profile: false
sidebar:
  title: "ML Glossary"
  nav: sidebar-glossary
---

{% include base_path %}

## Classification


:wavy_dash: <span class="compare"> Compare to </span> : 
[Regression](#regression)


### Decision Trees

<div>
<details open>
<summary>Overview</summary>

<div class="container-fluid">
  <div class="row text-center">
    <div class="col-xs-12 col-sm-6">
        <a href="#supervised-learning" class="infoLink">Supervised</a>
    </div>
    <div class="col-xs-12 col-sm-6">
        <a href="#classification" class="infoLink">Classification</a>
        or
        <a href="#regression" class="infoLink">Regression</a>
    </div>
    <div class="col-xs-12 col-sm-6">
        <a href="#generative-vs-discriminative" class="infoLink">Discriminative</a>
    </div>
    <div class="col-xs-12 col-sm-6">
        <a href="#parametric-vs-non-parametric" class="infoLink">Non-Parametric</a>
    </div>
    <div class="col-xs-12 col-sm-6">
        <a href="#generative-vs-discriminative" class="infoLink">Non-Probabilistic</a>
    </div>
    <div class="col-xs-12 col-sm-6">
        <span class="info">Piecewise Linear Decision Boundary</span>
    </div>
  </div>
</div>

<div markdown='1'>
* :bulb: <span class='intuition'> Intuition </span> :
    * Split the training data based on “the best” question(*e.g.* is he older than 27 ?). Recursively split the data while you are unhappy with the classification results.
    * Decision trees are basically the algorithm to use for the "20 question" game. [Akinator](http://en.akinator.com/) is a nice example of what can been implemented with decision trees. Akinator is probably based on fuzzy logic expert systems (as it can work with wrong answers) but you could do a simpler version with decision trees.
    * "Optimal" splits are found by maximization of [information gain](#machine-learning-and-entropy) or similar methods.
* :wrench: <span class='practice'> Practical </span> :
    * Decision trees thrive when you need a simple and interpretable model but the relationship between $y$ and $\mathbf{x}$ is complex.
    * Training Complexity : <span class='practiceText' markdown='1'> $O(MND + ND\log(N) )$ </span> . 
    * Testing Complexity : <span class='practiceText' markdown='1'> $O(MT)$ </span> .
    * Notation Used : $M=depth$ ; $$N= \#_{train}$$ ; $$D= \#_{features}$$ ; $$T= \#_{test}$$.
* :white_check_mark: <span class='advantage'> Advantage </span> :
    * <span class='advantageText'>  Interpretable </span> .
    * Few hyper-parameters.
    * Needs less data cleaning :
        * No normalization needed.
        * Can handle missing values.
        * Handles numerical and categorical variables.
    * Robust to outliers.
    * Doesn't make assumptions regarding the data distribution.
    * Performs feature selection.
    * Scales well.
* :x: <span class='disadvantage'> Disadvantage </span> :
    * Generally poor accuracy because greedy selection.
    * <span class='disadvantageText'> High variance</span> because if the top split changes, everything does.
    * Splits are parallel to features axes => need multiple splits to separate 2 classes with a 45° decision boundary.
    * No online learning.
</div>
</details>
</div> 
<p></p>

The basic idea behind building a decision tree is to :
1. Find an optimal split (feature + threshold). *I.e.* the split which minimizes the impurity (maximizes information gain). 
2. Partition the dataset into 2 subsets based on the split above.
3. Recursively apply $1$ and $2$ this to each new subset until a stop criterion is met.
4. To avoid over-fitting: prune the nodes which "aren't very useful". 

Here is a little gif showing these steps: 
<div markdown="1">
![Building Decision Trees Classification](/images/blog/glossary-old/decision-tree-class.gif){:width="477px" height="358px"}
</div>


Note: For more information, please see the "*details*" and "*Pseudocode and Complexity*" drop-down below.

<div>
<details>
<summary>Details</summary> 
<div markdown='1'>
The idea behind decision trees is to partition the input space into multiple regions. *E.g.* region of men who are more than 27 years old. Then predict the most probable class for each region, by assigning the mode of the training data in this region. Unfortunately, finding an optimal partitioning is usually computationally infeasible ([NP-complete](https://people.csail.mit.edu/rivest/HyafilRivest-ConstructingOptimalBinaryDecisionTreesIsNPComplete.pdf)) due to the combinatorially large number of possible trees. In practice the different algorithms thus use a greedy approach. *I.e.* each split of the decision tree tries to maximize a certain criterion regardless of the next splits. 

*How should we define an optimality criterion for a split?* Let's define an impurity (error) of the current state, which we'll try to minimize. Here are 3 possible state impurities:

* **Classification Error**:  
    * :bulb: <span class='intuitionText'> The accuracy error : $1-Acc$</span> of the current state. *I.e.* the error we would do, by stopping at the current state.
    * $$ClassificationError = 1 - \max_c (p(c))$$

* **[Entropy](#entropy)**:  
    * :bulb: <span class='intuitionText'> How unpredictable are the classes</span> of the current state. 
    * Minimize the entropy corresponds to maximizing the [information gain](#machine-learning-and-entropy).
    * $$Entropy = - \sum_{c=1}^C p(c) \log_2 \ p(c)$$

* **Gini Impurity**:  
    * :bulb: <span class='intuitionText'> Expected ($\mathbb{E}[\cdot] = \sum_{c=1}^C p(c) (\cdot) $) probability of misclassifying ($\sum_{c=1}^C p(c) (1-\cdot)$) a randomly selected element, if it were classified according to the label distribution ($\sum_{c=1}^C p(c) (1-p(c))$)</span> .
    * $$ClassificationError =  \sum_c^C p_c (1-p_c) = 1- \sum_c^C p_c^2$$

Here is a quick graph showing the impurity depending on a class distribution in a binary setting:

<div markdown='1'>
![Impurity Measure](/images/blog/glossary-old/impurity.png){:width='477px'}
</div>

:mag: <span class='note'> Side Notes </span>: 

* Classification error may seem like a natural choice, but don't get fooled by the appearances: it's generally worst than the 2 other methods:
    *  It is "more" greedy than the others. Indeed, it only focuses on the current error, while Gini and Entropy try to make a purer split which will make subsequent steps easier. <span class='exampleText'> Suppose we have a binary classification with 100 observation in each class $(100,100)$. Let's compare a split which divides the data into $(20,80)$ and $(80,20)$, to an other split which would divide it into $(40,100)$ and $(60,0)$. In both case the accuracy error would be of $0.20\%$. But we would prefer the second case, which is **pure** and will not have to be split further. Gini impurity and the Entropy would correctly chose the latter. </span> 
    *  Classification error takes only into account the most probable class. So having a split with 2 extremely probable classes will have a similar error to split with one extremely probable class and many improbable ones.
* Gini Impurity and Entropy [differ less than 2% of the time](https://www.unine.ch/files/live/sites/imi/files/shared/documents/papers/Gini_index_fulltext.pdf) as you can see in the plot above. Entropy is a little slower to compute due to the logarithmic operation.

**When should we stop splitting?** It is important not to split too many times to avoid over-fitting. Here are a few heuristics that can be used as a stopping criterion:

* When the number of training examples in a a leaf node is small.
* When the depth reaches a threshold.
* When the impurity is low.
* When the purity gain due to the split is small.

Such heuristics require problem-dependent thresholds (hyperparameters), and can yield relatively bad results. For example decision trees might have to split the data without any purity gain, to reach high purity gains at the following step. It is thus common to grow large trees using the number of training example in a leaf node as a stopping criterion. To avoid over-fitting, the algorithm would prune back the resulting tree. In CART, the pruning criterion $C_{pruning}(T)$ balances impurity and model complexity by regularization. The regularized variable is often the number of leaf nodes $\vert T \vert$, as below:

$$C_{pruning}(T) = \sum^{\vert T \vert }_{v=1} I(T,v) + \lambda \vert T \vert$$

$\lambda$ is selected via [cross validation](#cross-validation) and trades-off impurity and model complexity, for a given tree $T$, with leaf nodes $v=1...\vertT \vert$ using Impurity measure $I$.

**Variants**: there are various decision tree methods, that differ with regards to the following points:

* Splitting Criterion ? Gini / Entropy.
* Technique to reduce over-fitting ?
* How many variables can be used in a split ?
* Building binary trees ?
* Handling of missing values ?
* Do they handle regression?
* Robustness to outliers?

Famous variants:
* **ID3**: first decision tree implementation. Not used in practice. 
* **C4.5**: Improvement over ID3 by the same developer. Error based pruning. Uses entropy. Handles missing values. Susceptible to outliers. Can create empty branches.
* **CART**: Uses Gini.  Cost complexity pruning. Binary trees. Handles missing values. Handles regression. Not susceptible to outliers.
* **CHAID**: Finds a splitting variable using Chi-squared to test the dependency between a variable and a response. No pruning. Seems better for describing the data, but worst for predicting.

Other variants include : C5.0 (next version of C4.5, probably less used because it's patented), MARS.

:information_source: <span class='resources'> Resources </span> : A comparative study of [different decision tree methods](http://www.academia.edu/34100170/Comparative_Study_Id3_Cart_And_C4.5_Decision_Tree_Algorithm_A_Survey).
</div>
</details>
</div> 
<p></p>

<div>
<details>
<summary>Pseudocode and Complexity</summary>
<div markdown='1'>

* **Pseudocode**
The simple version of a decision tree can be written in a few lines of python pseudocode:

```python
def buildTree(X,Y):
    if stop_criteria(X,Y) :
        # if stop then store the majority class
        tree.class = mode(X) 
        return Null

    minImpurity = infinity
    bestSplit = None
    for j in features:
        for T in thresholds:
            if impurity(X,Y,j,T) < minImpurity:
                bestSplit = (j,T)
                minImpurity = impurity(X,Y,j,T) 

    X_left,Y_Left,X_right,Y_right = split(X,Y,bestSplit)

    tree.split = bestSplit # adds current split
    tree.left = buildTree(X_left,Y_Left) # adds subsequent left splits
    tree.right buildTree(X_right,Y_right) # adds subsequent right splits

return tree

def singlePredictTree(tree,xi):
    if tree.class is not Null:
        return tree.class

    j,T = tree.split
    if xi[j] >= T:
        return singlePredictTree(tree.right,xi)
    else:
        return singlePredictTree(tree.left,xi)

def allPredictTree(tree,Xt):
    t,d = Xt.shape
    Yt = vector(d)
    for i in t:
        Yt[i] = singlePredictTree(tree,Xt[i,:])

    return Yt
```

* **Complexity**
I will be using the following notation: $$M=depth$$ ; $$K=\#_{thresholds}$$ ; $$N = \#_{train}$$ ; $$D = \#_{features}$$ ; $$T = \#_{test}$$ . 

Let's first think about the complexity for building the first decision stump (first function call):

* In a decision stump, we loop over all features and thresholds $O(KD)$, then compute the impurity. The impurity depends solely on class probabilities. Computing probabilities means looping over all $X$ and count the $Y$ : $O(N)$. With this simple pseudocode, the time complexity for building a stump is thus $O(KDN)$. 
* In reality, we don't have to look for arbitrary thresholds, only for the unique values taken by at least an example. *E.g.* no need of testing $feature_j>0.11$ and $feature_j>0.12$ when all $feature_j$ are either $0.10$ or $0.80$. Let's replace the number of possible thresholds $K$ by training set size $N$. $O(N^2D)$
* Currently we are looping twice over all $X$, once for the threshold and once to compute the impurity. If the data was sorted by the current feature, the impurity could simply be updated as we loop through the examples. *E.g.* when considering the rule $feature_j>0.8$ after having already considered $feature_j>0.7$, we do not have to recompute all the class probabilities: we can simply take the probabilities from $feature_j>0.7$ and make the adjustments knowing the number of example with $feature_j==0.7$. For each feature $j$ we should first sort all data $O(N\log(N))$ then loop once in $O(N)$, the final would be in $O(DN\log(N))$.

We now have the complexity of a decision stump. You could think that finding the complexity of building a tree would be multiplying it by the number of function calls: Right? Not really, that would be an over-estimate. Indeed, at each function call, the training data size $N$ would have decreased. The intuition for the result we are looking for, is that at each level $l=1...M$ the sum of the training data in each function is still $N$. Multiple function working in parallel with a subset of examples take the same time as a single function would, with the whole training set $N$. The complexity at each level is thus still $O(DN\log(N))$ so the complexity for building a tree of depth $M$ is $O(MDN\log(N))$. Proof that the work at each level stays constant:

At each iterations the dataset is split into $\nu$ subsets of $k_i$ element and a set of $n-\sum_{i=1}^{\nu} k_i$. At every level, the total cost would therefore be (using properties of logarithms and the fact that $k_i \le N$ ) : 

$$
\begin{align*}
cost &= O(k_1D\log(k_1)) + ... + O((N-\sum_{i=1}^{\nu} k_i)D\log(N-\sum_{i=1}^{\nu} k_i))\\
    &\le O(k_1D\log(N)) + ... + O((N-\sum_{i=1}^{\nu} k_i)D\log(N))\\
    &= O(((N-\sum_{i=1}^{\nu} k_i)+\sum_{i=1}^{\nu} k_i)D\log(N)) \\
    &= O(ND\log(N))   
\end{align*} 
$$

The last possible adjustment I see, is to sort everything once, store it and simply use this precomputed data at each level. The final training complexity is therefore <span class='practiceText'> $O(MDN + ND\log(N))$ </span> .

The time complexity of making predictions is straightforward: for each $t$ examples, go through a question at each $M$ levels. *I.e.* <span class='practiceText'> $O(MT)$ </span> .
</div>
</details>
</div> 
<p></p>


### Naive Bayes

<div>
<details open>
<summary>Overview</summary>

<div class="container-fluid">
  <div class="row text-center">
    <div class="col-xs-12 col-sm-6">
        <a href="#supervised-learning" class="infoLink">Supervised</a>
    </div>
    <div class="col-xs-12 col-sm-6">
            <a href="#supervised-learning" class="infoLink">Classification</a>
        </div>
    <div class="col-xs-12 col-sm-6">
        <a href="#generative-vs-discriminative" class="infoLink">Generative</a>
    </div>
    <div class="col-xs-12 col-sm-6">
        <a href="#parametric-vs-non-parametric" class="infoLink">Parametric</a>
    </div>
    <div class="col-xs-12 col-sm-6">
        Gaussian Case: <span class="info">Piecewise Quadratic Decision Boundary</span>
    </div>
    <div class="col-xs-12 col-sm-6">
       Discrete Case: <span class="info">Piecewise Linear Decision Boundary</span>
    </div>
  </div>
</div>


<div markdown='1'>
* :bulb: <span class='intuition'> Intuition </span> :
    * In the discrete case: "advance counting". *E.g.* Given a sentence $x_i$ to classify as spam or not, count all the times each word $w^i_j$ was in previously seen spam sentence and predict as spam if the total (weighted) "spam counts" is larger than the number of "non spam counts".
    * Use a conditional independence assumption ("naive") to have better estimates of the parameters even with little data.
* :wrench: <span class='practice'> Practical </span> :
    * Good and simple baseline.
    * Thrive when the number of features is large but the dataset size is not.
    * Training Complexity : <span class='practiceText' markdown='1'> $O(ND)$ </span> . 
    * Testing Complexity : <span class='practiceText' markdown='1'> $O(TDK)$ </span> .
    * Notation Used : $$N= \#_{train}$$ ; $$K= \#_{classes}$$ ; $$D= \#_{features}$$ ; $$T= \#_{test}$$.
* :white_check_mark: <span class='advantage'> Advantage </span> :
    * Simple to understand and implement.
    * Fast and scalable (train and test).
    * Handles online learning.
    * Works well with little data.
    * Not sensitive to irrelevant features.
    * Handles real and discrete data.
    * Probabilistic.
    * Handles missing value.
* :x: <span class='disadvantage'> Disadvantage </span> :
    * Strong conditional independence assumption of features given labels.
    * Sensitive to features that have not often been seen (mitigated by smoothing).
</div>
</details>
</div> 
<p></p>

Naive Bayes is a family of generative models that predicts $p(y=c \vert\mathbf{x})$ by assuming that all features are conditionally independent given the label: $x_i \perp x_j \vert y , \forall i,j$. This is a simplifying assumption that very rarely holds in practice, which makes the algorithm "naive". Classifying with such an assumption is very easy:

$$
\begin{aligned}
\hat{y} &= arg\max_c p(y=c\vert\mathbf{x}, \pmb\theta) \\
&= arg\max_c \frac{p(y=c, \pmb\theta)p(\mathbf{x}\vert y=c, \pmb\theta) }{p(x, \pmb\theta)} &  & \text{Bayes Rule} \\
&= arg\max_c \frac{p(y=c, \pmb\theta)\prod_{j=1}^D p(x_\vert y=c, \pmb\theta) }{p(x, \pmb\theta)} &  & \text{Conditional Independence Assumption} \\
&= arg\max_c p(y=c, \pmb\theta)\prod_{j=1}^D p(x_j\vert y=c, \pmb\theta)  &  & \text{Constant denominator}
\end{aligned}
$$

Note that because we are in a classification setting $y$ takes discrete values, so $p(y=c, \pmb\theta)=\pi_c$ is a categorical distribution.

You might wonder why we use the simplifying conditional independence assumption. We could directly predict using $\hat{y} = arg\max_c p(y=c, \pmb\theta)p(\mathbf{x} \vert y=c, \pmb\theta)$. <span class='intuitionText'> The conditional assumption enables us to have better estimates of the parameters $\theta$ using less data </span>. Indeed, $p(\mathbf{x} \vert y=c, \pmb\theta)$ requires to have much more data as it is a $D$ dimensional distribution (for each possible label $c$), while $\prod_{j=1}^D p(x_j \vert y=c, \pmb\theta)$ factorizes it into $D$ 1-dimensional distributions which requires a lot less data to fit due to [curse of dimensionality](#curse-of-dimensionality). In addition to requiring less data, it also enables to easily mix different family of distributions for each features.

We still have to address 2 important questions: 

* What family of distribution to use for $p(x_j \vert y=c, \pmb\theta)$  (often called the *event model* of the Naive Bayes classifier)?
* How to estimate the parameters $\theta$?

#### Event models of Naive Bayes

The family of distributions to use is an important design choice that will give rise to specific types of Naive Bayes classifiers. Importantly the family of distribution $p(x_j \vert y=c, \pmb\theta)$ does not need to be the same $\forall j$, which enables the use of very different features (*e.g.* continuous and discrete). In practice, people often use Gaussian distribution for continuous features, and Multinomial or Bernoulli distributions for discrete features :

**Gaussian Naive Bayes :**

Using a Gaussian distribution is a typical assumption when dealing with continuous data $x_j \in \mathbb{R}$. This corresponds to assuming that each feature conditioned over the label is a univariate Gaussian:

$$p(x_j \vert y=c, \pmb\theta) = \mathcal{N}(x_j;\mu_{jc},\sigma_{jc}^2)$$

Note that if all the features are assumed to be Gaussian, this corresponds to fitting a multivariate Gaussian with a diagonal covariance : $p(\mathbf{x} \vert y=c, \pmb\theta)= \mathcal{N}(\mathbf{x};\pmb\mu_{c},\text{diag}(\pmb\sigma_{c}^2))$.

<span class='intuitionText'> The decision boundary is quadratic as it corresponds to ellipses (Gaussians) that intercept </span>. 

**Multinomial Naive Bayes :**

In the case of categorical features $x_j \in \\{1,..., K\\}$ we can use a Multinomial distribution, where $\theta_{jc}$ denotes the probability of having feature $j$ at any step of an example of class $c$  :

$$p(\pmb{x} \vert y=c, \pmb\theta) = \operatorname{Mu}(\pmb{x}; \theta_{jc}) = \frac{(\sum_j x_j)!}{\prod_j x_j !} \prod_{j=1}^D \theta_{jc}^{x_j}$$

<span class='practiceText'> Multinomial Naive Bayes is typically used for document classification </span> , and corresponds to representing all documents as a bag of word (no order). We then estimate  (see below)$\theta_{jc}$ by counting word occurrences to find the proportions of times word $j$ is found in a document classified as $c$. 

<span class='noteText'> The equation above is called Naive Bayes although the features $x_j$ are not technically independent because of the constraint $\sum_j x_j = const$</span>. The training procedure is still the same because for classification we only care about comparing probabilities rather than their absolute values, in which case Multinomial Naive Bayes actually gives the same results as a product of Categorical Naive Bayes whose features satisfy the conditional independence property.

Multinomial Naive Bayes is a linear classifier when expressed in log-space :

$$
\begin{aligned}
\log p(y=c \vert \mathbf{x}, \pmb\theta) &\propto \log \left(  p(y=c, \pmb\theta)\prod_{j=1}^D p(x_j \vert y=c, \pmb\theta) \right)\\
&= \log p(y=c, \pmb\theta) + \sum_{j=1}^D x_j \log \theta_{jc} \\
&= b + \mathbf{w}^T_c \mathbf{x} \\
\end{aligned}
$$

**Multivariate Bernoulli Naive Bayes:**

In the case of binary features $x_j \in \\{0,1\\}$ we can use a Bernoulli distribution, where $\theta_{jc}$ denotes the probability that feature $j$ occurs in class $c$:

$$p(x_j \vert y=c, \pmb\theta) = \operatorname{Ber}(x_j; \theta_{jc}) = \theta_{jc}^{x_j} \cdot (1-\theta_{jc})^{1-x_j}$$

<span class='practiceText'> Bernoulli Naive Bayes is typically used for classifying short text </span> , and corresponds to looking at the presence and absence of words in a phrase (no counts). 

<span class='noteText'> Multivariate Bernoulli Naive Bayes is not the same as using Multinomial Naive Bayes with frequency counts truncated to 1</span>. Indeed, it models the absence of words in addition to their presence.

#### Training

Finally we have to train the model by finding the best estimated parameters $\hat\theta$. This can either be done using pointwise estimates (*e.g.* MLE) or a Bayesian perspective.

**Maximum Likelihood Estimate (MLE):**

The negative log-likelihood of the dataset $\mathcal{D}=\\{\mathbf{x}^{(n)},y^{(n)}\\}_{n=1}^N$ is :

$$
\begin{aligned}
NL\mathcal{L}(\pmb{\theta} \vert \mathcal{D}) &= - \log \mathcal{L}(\pmb{\theta} \vert \mathcal{D}) \\
&= - \log \prod_{n=1}^N \mathcal{L}(\pmb{\theta} \vert \mathbf{x}^{(n)},y^{(n)}) & & \textit{i.i.d} \text{ dataset} \\
&= - \log \prod_{n=1}^N p(\mathbf{x}^{(n)},y^{(n)} \vert \pmb{\theta}) \\
&= - \log \prod_{n=1}^N \left( p(y^{(n)} \vert \pmb{\pi}) \prod_{j=1}^D p(x_{j}^{(n)} \vert\pmb{\theta}_j) \right) \\
&= - \log \prod_{n=1}^N \left( \prod_{c=1}^C \pi_c^{\mathcal{I}[y^{(n)}=c]} \prod_{j=1}^D \prod_{c=1}^C p(x_{j}^{(n)} \vert \theta_{jc})^{\mathcal{I}[y^{(n)}=c]} \right) \\
&= - \log \left( \prod_{c=1}^C \pi_c^{N_c} \prod_{j=1}^D \prod_{c=1}^C \prod_{n : y^{(n)}=c} p(x_{j}^{(n)} \vert \theta_{jc}) \right) \\
&= -  \sum_{c=1}^C N_c \log \pi_c + \sum_{j=1}^D \sum_{c=1}^C \sum_{n : y^{(n)}=c} \log p(x_{j}^{(n)} \vert \theta_{jc})  \\
\end{aligned}
$$

As the negative log likelihood decomposes in terms that only depend on $\pi$ and each $\theta_{jc}$ we can optimize all parameters separately.

Minimizing the first term by using Lagrange multipliers to enforce $\sum_c \pi_c$, we get that $\hat\pi_c = \frac{N_c}{N}$. Which is naturally the proportion of examples labeled with $y=c$.

The $\theta_{jc}$ depends on the family of distribution we are using. In the Multinomial case it can be shown to be $\hat\theta_{jc}=\frac{N_{jc}}{N_c}$. Which is very easy to compute, as it only requires to count the number of times a certain feature $x_j$ is seen in an example with label $y=c$.

**Bayesian Estimate:**

The problem with MLE is that it over-fits. For example, if a feature is always present in all training samples (*e.g.* the word "the" in document classification) then the model will break if it sees a test sample without that features as it would give a probability of 0 to all labels. 

By taking a Bayesian approach, over-fitting is mitigated thanks to priors. In order to do so we have to compute the posterior : 

$$
\begin{aligned}
p(\pmb\theta \vert \mathcal{D}) &= \prod_{n=1}^N p(\pmb\theta \vert \mathbf{x}^{(n)},y^{(n)}) & \textit{i.i.d} \text{ dataset} \\
&\propto \prod_{n=1}^N p(\mathbf{x}^{(n)},y^{(n)} \vert \pmb\theta)p(\pmb\theta) \\
&\propto \prod_{c=1}^C \left( \pi_c^{N_c} \cdot p(\pi_c) \right) \prod_{j=1}^D \prod_{c=1}^C \prod_{n : y^{(n)}=c} \left( p(x_{j}^{(n)} \vert \theta_{jc}) \cdot  p(\theta_{jc}) \right) \\
\end{aligned}
$$

Using (factored) conjugate priors (Dirichlet for Multinomial, Beta for Bernoulli, Gaussian for Gaussian), this gives the same estimates as in the MLE case (the prior has the same form than the likelihood and posterior) but regularized. 

The Bayesian framework requires predicting by integrating out all the parameters $\pmb\theta$. The only difference with the first set of equations we have derived for classifying using Naive Bayes, is that the predictive distribution is conditioned on the training data $\mathcal{D}$ instead of the parameters $\pmb\theta$:

$$
\begin{aligned}
\hat{y} &= arg\max_c p(y=c \vert \pmb{x},\mathcal{D}) \\
&= arg\max_c \int p(y=c\vert \pmb{x},\pmb\theta) p(\pmb\theta \vert \mathcal{D}) d\pmb\theta\\
&= arg\max_c \int p(\pmb\theta\vert\mathcal{D}) p(y=c \vert\pmb\theta) \prod_{j=1}^D p(x_j\vert y=c, \pmb\theta) d\pmb\theta \\
&= arg\max_c \left( \int p(y=c\vert\pmb\pi) p(\pmb\pi \vert \mathcal{D}) d\pmb\pi \right) \prod_{j=1}^D \int p(x_j\vert y=c, \theta_{jc}) p(\theta_{jc}\vert\mathcal{D}) d\theta_{jc}  &  & \text{Factored prior} \\
\end{aligned}
$$

As before, the term to maximize decomposes in one term per parameter. All parameters can thus be independently maximized. These integrals are generally intractable, but in the case of the Gaussian, the Multinomial, and the Bernoulli with their corresponding conjugate prior, we can fortunately compute a closed form solution.

In the case of the Multinomial (and Bernoulli), the solution is equivalent to predicting with a point estimate $\hat{\pmb\theta}=\bar{\pmb\theta}$. Where $\bar{\pmb\theta}$ is the mean of the posterior distribution. 

Using a symmetric Dirichlet prior : $p(\pmb\pi)=\text{Dir}(\pmb\pi; \pmb{1}\alpha_\pi)$ we get that $\hat\pi_c = \bar\pi_c = \frac{N_c + \alpha_\pi }{N + \alpha_\pi C}$.

The Bayesian Multinomial naive Bayes is equivalent to predicting with a point estimate :

$$\bar\theta_{jc} = \hat\theta_{jc}=\frac{N_{jc} +  \alpha }{N_c + \alpha D}$$

*I.e.* **Bayesian Multinomial Naive Bayes** with symmetric Dirichlet prior assigns predictive posterior distribution:

$$p(y=c\vert\mathbf{x},\mathcal{D}) \propto \frac{N_c + \alpha_\theta }{N + \alpha_\theta C} \prod_{j=1}^D \frac{N_{jc} +  \alpha_\theta }{N_c + \alpha_\theta D}$$

the corresponding graphical model is:

<div markdown="1">
![Bayesian Naive Bayes](/images/blog/glossary-old/Bayesian_MNB.png){:width='477px'}
</div>


When using a uniform prior $\alpha_\theta=1$, this equation is called **Laplace smoothing** or **add-one smoothing**.  <span class='intuitionText'> $\alpha$ intuitively represents a "pseudocount" $\alpha_\theta$ of features $x_{jc}$ </span>. <span class='exampleText'> For document classification </span> it simply corresponds to giving an initial non zero count to all words, which avoids the problem of having a test document $x^{(t)}$ with $p(y=c\vert x^{(t)})=0$ if it contains a single word $x_{j}^*$ that has never been seen in a training document with label $c$. <span class='practiceText'> $\alpha=1$ is a common choice in examples although smaller often work better </span>.


:mag: <span class='note'> Side Notes </span> : 

* The "Naive" comes from the conditional independence of features given the label. The "Bayes" part of the name comes from the use of Bayes theorem to use a generative model, but it is not a Bayesian method as it does not require marginalizing over all parameters.

* If we estimated the Multinomial Naive Bayes using Maximum A Posteriori Estimate (MAP) instead of MSE or the Bayesian way, we would predict using the mode of the posterior (instead of the mean in the Bayesian case) $\hat\theta_{jc}=\frac{N_{jc} +  \alpha - 1}{N_c + (\alpha -1)D}$ (similarly for $\pmb\pi$). This means that Laplace smoothing could also be interpreted as using MAP with a non uniform prior $\text{Dir}(\pmb\theta; \pmb{2})$. But when using a uniform Dirichlet prior MAP coincides with the MSE.

* Gaussian Naive Bayes is equivalent to Quadratic Discriminant Analysis when each covariance matrix $\Sigma_c$ is diagonal.

* Discrete Naive Bayes and Logistic Regression are a "generative-discriminative pair" as they both take the same form (linear in log probabilities) but estimate parameters differently. For example, in the binary case, Naive Bayes predicts the class with the largest probability. *I.e.* it predicts $C_1$ if $\log \frac{p(C_1 \vert \pmb{x})}{p(C_2 \vert \pmb{x})} = \log \frac{p(C_1 \vert \pmb{x})}{1-p(C_1 \vert \pmb{x})} > 0$. We have seen that discrete Naive Bayes is linear in the log space, so we can rewrite the equation as $\log \frac{p(C_1 \vert \pmb{x})}{1-p(C_1 \vert \pmb{x})} = 2 \log p(C_1 \vert \pmb{x}) - 1 = 2 \left( b + \mathbf{w}^T_c \mathbf{x} \right) - 1 = b' + \mathbf{w'}^T_c \mathbf{x} > 0$. This linear regression on the log odds ratio is exactly the form of Logistic Regression (the usual equation is recovered by solving $\log \frac{p}{1-p} = b + \mathbf{w'}^T \mathbf{x}$). The same can be shown for Multinomial Naive Bayes and Multinomial Logistic Regression.

* For document classification, it is common to replace raw counts in Multinomial Naive Bayes with [tf-idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) weights.

:information_source: <span class='resources'> Resources </span> : See section 3.5 of [K. Murphy's book](https://www.cs.ubc.ca/~murphyk/MLbook/) for all the derivation steps and examples.

## Regression
### Decision Trees
Decision trees are more often used for classification problems. I thus talk at length about them [here](#decision-trees-1).

The 2 differences with decision trees for classification are:
* **What error to minimize for an optimal split?** This replaces the impurity measure in the classification setting. A widely used error function for regression is the [sum of squared error](#mean-squared-error). We don't use the mean squared error so that the subtraction of the error after and before a split make sense. Sum of squared error for region $R$:

$$Error = \sum_{x^{(n)} \in R} (y^{(n)} - \bar{y}_{R})^2$$

* **What to predict for a given space region?** In the classification setting, we predicted the mode of the subset of training data in this space. Taking the mode doesn't make sense for a continuous variable. Now that we've defined an error function above, we would like to predict a value which minimizes this sum of squares error function. This corresponds to the region **average** value. Predicting the mean is intuitively what we would have done. 

Let's look at a simple plot to get a better idea of the algorithm:

<div markdown="1">
![Building Decision Trees Regression](/images/blog/glossary-old/decision-tree-reg.gif){:width='477px' :height='327px'}
</div>

:x: Besides the disadvantages seen in the [decision trees for classification](#decision-trees-1), decision trees for regression suffer from the fact that it predicts a <span class='disadvantageText'> non smooth function  </span>.
