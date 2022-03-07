* Posterior grid plots display data samples (x), IS estimates of the probability of data samples, posterior samples (i.e., a sample from X|z,theta for z sampled from Z|x,lambda.
* Posterior marginal samples: for each class (from 0 to 9), sample a digit from the data, infer a posterior approximation, sample a latent code, predict output distributions, sample a digit, repeat that a number of times, plot the average pixels.
* Prior grid: sample many times from the model (via latent codes sampled from the prior), cluster the sampled digits using a KNN classifier trained on real MNIST, and plot a grid with 9 samples per digit and the average of all samples of the same class.
* Prior marginal samples: this is the average of all samples predicted to have the same class, we also show their frequency.
* Prior corner samples: condition the generator on one of the vertices of the simplex and sample a many times, plot the average pixels per vertex.
* TSNE plots:
    * f: using the face encoding as the representation of the digit
    * mu: using the vector of marginal probabilities of each vertex as the representation of the digit
    * scores: using the scores of the Gibbs distribution over faces as the representation of the diit
    * y: using the actual latent sample (a variable in the simplex) as the representation of the digit


