# Maximum-Entropy based Stochastic and Symbolic density estimation (MESSY)

In this repository, we present an implementation of MESSY paper published in TMLR:
https://openreview.net/pdf?id=Y2ru0LuQeS

Given samples $X$ of an unknown density $f$, MESSY finds the maximum entropy distribution of the form

$\hat f( x) = Z^{-1}\exp\big( \lambda \cdot H( x) \big) $

where $Z =\int \exp(  \lambda \cdot  H( x)  ) d  x$ is the normalization constant. We note that $\hat f$ is the extremum of the objective functional which maximizes Shannon entropy with the constraint on moments $\mu=\mathbb{E}[H(X)]$, i.e.

$ \mathcal{C} [\mathcal{F}(x)]:=\int \mathcal{F}( x) \log(\mathcal{F}(x)) d x - \sum_{i=1}^{N_b} \lambda_i  \left(\int H_i( x) \mathcal{F}( x) d  x-\mu_i( x)\right)~$.

