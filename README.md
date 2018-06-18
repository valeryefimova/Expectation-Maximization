# Expectation-Maximization

This repository contains an EM-algorithm implementation as a course work for Machine learning Course.

Experiments were conducted with [MNIST](http://yann.lecun.com/exdb/mnist/) dataset.
Other possible datasets are under dataset/cl/.

You can run the default run configuration as:
```
python3 mnist_em.py <filename>
```

This implementation was compared to [Gaussian Mixture](http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture) algorithm from scikit learn.
Hyperparameter optimization made by [SMAC](http://www.ml4aad.org/automated-algorithm-design/algorithm-configuration/smac/) algorithm.

You can run
```  
pythom3 em_gm_comparison.py
```