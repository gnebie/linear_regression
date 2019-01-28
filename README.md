# linear_regression

### Librairies utilisées

``` py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

* numpy      : Librairie gestion des mathématique
* panda      : Librairie permettant la manipulation et l'analyse des données
* matplotlib : Librairie destinée à tracer et visualiser des données sous formes de graphiques


## Linear regression :

**The linear regression is a supervised Learning algorithm.**

**It goal is to find a regression line (or a curve with multiple datas) to predict the real-valued output**

 size (x) | price (y)
 ---      |---
  20      | 2000
  30      | 3000
  40      | 4000

training set size (m) |
--                    |
3  |

$$y_i = h(x_i)$$

Linear regression function : $h_\theta(x) = \theta_0 + \theta_1 * x$

## cost function

To calculate the x tetras a cost function J is used

$$J(\theta_0, \theta_1) = {1 \over 2m} {\sum_{i=1}^{m} (h_\theta(x^{(i)} ) - y^{(i)}) ^2 }$$

## Goal : minimize J with gradient descent

Formula :
$$\theta_j := \theta_j - \alpha {\varphi \over \varphi\theta_j}J(\theta_0, \theta_1) $$
Repeat until convergence and simultaneously update all j (0 - 1)
* If a is too small the descent will be long
* If a is to large the descent will diverge

### Linear regression gradient descent

$$\theta_0 := \theta_0 - \alpha {1 \over m} {\sum_{i=1}^{m} (h_\theta(x^{(i)} ) - y^{(i)}) }
$$
$$\theta_1 := \theta_1 - \alpha {1 \over m} {\sum_{i=1}^{m} (h_\theta(x^{(i)} ) - y^{(i)}) . x^{(i)} }$$

### Linear regression gradient descent with multiple values

$x_0 == 1$

Linear regression function : $h_\theta(x) = \theta_0 + \theta_1 * x_1 + ... \theta_n * x_n$

$$\theta_j := \theta_j - \alpha {1 \over m} {\sum_{i=1}^{m} (h_\theta(x^{(i)} ) - y^{(i)}) . x^{(i)} }$$

### tips to improve the datas :

Feature scaling : Get every feature into approximatively e $-1 < x_i < 1$ range

mean normalization :  replace $x_i$ with $x_i - \bar x$ (do not apply for $x_0$)

Declare convvergence when x decreasze than less than $10^{-3}$ in one iteration.

Modify $\alpha$ by $* 3$ or $/ 3$ to find a good value 

### polynomial regression

Copy a x value and give it a new coeficient ($\sqrt x / x^2 / x^3 / ...$) and calculate the new set of values $x_1 = x$ and $x_2 = x^2$
