import numpy as np

class GaussianMixtureModel:
    def __init__(self, n_components, max_iterations=100, tolerance=1e-6, random_state=None):
        # Initialize GMM with the specified number of components and other parameters
        self.n_components = n_components
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.random_state = random_state
        self.weights = None  # Weights for each component
        self.means = None  # Means for each component
        self.covs = None  # Covariances for each component

    def fit(self, X):
        n_samples, n_features = X.shape
        # Initialize weights, means, and covariances based on the data and specified parameters
        np.random.seed(self.random_state)
        self.weights = np.full(self.n_components, 1 / self.n_components)
        random_indices = np.random.choice(n_samples, self.n_components, replace=False)
        self.means = X[random_indices]
        self.covs = [np.cov(X.T)] * self.n_components

        for _ in range(self.max_iterations):
            # E-step: Compute likelihoods and responsibilities
            likelihoods = np.array([self.weights[j] * self._multivariate_gaussian(X, self.means[j], self.covs[j]) for j in range(self.n_components)])
            responsibilities = likelihoods / np.sum(likelihoods, axis=0)

            # M-step: Update weights, means, and covariances
            N = np.sum(responsibilities, axis=1)
            self.weights = N / n_samples
            self.means = np.dot(responsibilities, X) / N[:, None]
            for j in range(self.n_components):
                diff = X - self.means[j]
                self.covs[j] = np.dot(responsibilities[j] * diff.T, diff) / N[j]

    def _multivariate_gaussian(self, X, mean, cov):
        # Calculate the multivariate Gaussian distribution for the given data points
        return np.exp(-0.5 * np.sum(np.dot(X - mean, np.linalg.inv(cov)) * (X - mean), axis=1)) / np.sqrt(np.linalg.det(cov) * (2 * np.pi) ** X.shape[1])

    def score(self, X):
        # Calculate the log-likelihood of the data given the GMM model
        likelihoods = np.array([self._multivariate_gaussian(X, self.means[j], self.covs[j]) for j in range(self.n_components)])
        return np.sum(np.log(np.sum(likelihoods, axis=0)))