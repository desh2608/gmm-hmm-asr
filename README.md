# Simple GMM-HMM model for isolated digit recognition
Python implementation of simple GMM and HMM models for isolated digit recognition.

This implementation contains 3 models:

1. Single Gaussian (`sg`): Each digit is modeled using a single Gaussian with diagonal covariance.
2. Gaussian Mixture Model (`gmm`): Each digit is modeled using a mixture of `ncomp` Gaussians, initialized by perturbing the `sg` model.
3. Hidden Markov Model (`hmm`): Each digit is modeled by an HMM consisting of `nstate` states, where the emission probability of each state is a single Gaussian with diagonal covariance.

### How to run

```
python3 submission.py <opt-args> train test
```

* `train` is the training data
* `test` is the test data

The optional arguments are:
* `--mode`: Type of model (`sg`, `gmm`, `hmm`). Default: `sg`
* `--niter`: Number of iterations. Default = 10
* `--ncomp`: Number of components in GMM model. Default = 8
* `--nstate`: Number of states in HMM model. Default = 5
* `--debug`: Uses only top 100 utterances for train and test

### Training data format

I cannot upload the full training and test data (for copyright reasons), but a small sample of the training data can be found at this [Google Drive link](https://drive.google.com/file/d/1NhF7fuX54jau9iXxuitOfm9QRQPHNW2Q/view?usp=sharing). This should help in understanding the format of the data.

### Help

This code is based on a template provided by Shinji Watanabe (Johns Hopkins University), written for a course project.

For assistance, contact `draj@cs.jhu.edu`.
 
