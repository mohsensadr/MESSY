# Maximum-Entropy based Stochastic and Symbolic density estimation (MESSY)

In this repository, we present an implementation of MESSY paper published in TMLR:
https://openreview.net/pdf?id=Y2ru0LuQeS

Given samples $X$ of an unknown density $f$, MESSY finds the maximum entropy distribution of the form

$f(x) = \exp\big( \lambda \cdot H( x) \big) / Z $

where $Z=\int \exp(\lambda \cdot H(x)) dx$ is the normalization constant. We note that $f$ is the extremum of the objective functional which maximizes Shannon entropy with the constraint on moments $\mu=\mathbb{E}[H(X)]$.
