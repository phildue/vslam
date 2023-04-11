import numpy as np
import matplotlib.pyplot as plt

# https://stackoverflow.com/questions/41957633/sample-from-a-multivariate-t-distribution-python
def multivariate_t_rvs(m, S, df=np.inf, n=1):
    '''generate random variables of multivariate t distribution
    Parameters
    ----------
    m : array_like
        mean of random variable, length determines dimension of random variable
    S : array_like
        square array of covariance  matrix
    df : int or float
        degrees of freedom
    n : int
        number of observations, return random array will be (n, len(m))
    Returns
    -------
    rvs : ndarray, (n, len(m))
        each row is an independent draw of a multivariate t distributed
        random variable
    '''
    m = np.asarray(m)
    d = len(m)
    if df == np.inf:
        x = np.ones(n)
    else:
        x = np.random.chisquare(df, n) / df
    z = np.random.multivariate_normal(np.zeros(d), S, (n,))
    return m + z/np.sqrt(x)[:, None]   # same output format as random.multivariate_normal

def fit(x, dof, max_iterations=30, convergence_threshold=1e-7, mean_0=None, covariance_0=None):
    n_samples = x.shape[0]
    n_dimensions = x.shape[1]
    covariance = covariance_0 if covariance_0 is not None else np.identity(n_dimensions)
    mean = mean_0 if mean_0 is not None else np.zeros((1, n_dimensions))
    print(f"Estimating {n_dimensions}-dimensional t distribution for {n_samples} samples for {dof} degrees of freedom.")
    step_size = np.inf
    for i in range(max_iterations):
        mean_i = np.zeros((1, n_dimensions))
        u1 = np.zeros((n_samples,))
        information = np.linalg.inv(covariance)
        for n in range(n_samples):
            u1[n] = (dof + n_dimensions) / (dof + (x[n].T @ information @ x[n]))
            mean_i += x[n] * u1[n]
      
        mean_i /= u1.sum()
        covariance_i = np.zeros((n_dimensions, n_dimensions))
        x_mean = x - mean
        for n in range(n_samples):
            cx = u1[n] * np.outer(x_mean[n], x_mean[n])
            covariance_i += cx
        covariance_i /= n_samples
        step_size = np.linalg.norm(covariance - covariance_i)
        
        print(f"iteration = {i}, step_size = {step_size}\n mean = {mean_i}, covariance = {covariance_i}")
        mean = mean_i
        covariance = covariance_i
        if step_size < convergence_threshold:
            break

    return mean, covariance


if __name__ == '__main__':
    
    mean = np.array([0., 0.])
    cov = np.array([[2., 1.],
                    [1., 2.]])
    
    x = multivariate_t_rvs(mean, cov, 5, 10000)

    
    mean_est, cov_est = fit(x, 5, 100, 1e-7)

    print(f"mean = {mean_est}, covariance = {cov_est}")
    np.savetxt('samples_t_distribution.csv', x, delimiter=',', fmt='%.3f')
    
    plt.figure()
    plt.plot(x[:, 0], x[:, 1], ".")
    plt.title("t-Distribution")
    plt.show()