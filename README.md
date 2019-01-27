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

Linear regression function : $h\theta(x) = \theta_0 + \theta_1 * x$

## cost function

To calculate the x tetras a cost function J is used

$$J(\theta 0, \theta 1) = 1 / 2m {\sum_{i=1}^{m} (h_\theta(x^{(i)} ) - y^{(i)}) ^2 }$$



#### exemple
When $a \ne 0$, there are two solutions to \(ax^2 + bx + c = 0\) and they are
$$x = {-b \pm \sqrt{b^2-4ac} \over 2a}.$$

$$\cos()$$
