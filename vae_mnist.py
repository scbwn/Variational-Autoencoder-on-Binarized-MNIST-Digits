import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.norm as norm
from autograd.scipy.special import expit as sigmoid
from autograd import grad
from autograd.misc.optimizers import adam

import os
import gzip
import struct
import array
import pickle

import matplotlib.pyplot as plt
import matplotlib.image
from urllib.request import urlretrieve
%matplotlib inline 
seed = npr.RandomState(0)

def diag_gaussian_log_density(x, mu, log_std):
    return np.sum(norm.logpdf(x, mu, np.exp(log_std)), axis=-1)

def unpack_gaussian_params(params):
    # Params of a diagonal Gaussian.
    D = np.shape(params)[-1] // 2
    mean, log_std = params[:, :D], params[:, D:]
    return mean, log_std

def sample_diag_gaussian(mean, log_std, rs):
    return rs.randn(*mean.shape) * np.exp(log_std) + mean

def bernoulli_log_density(b, unnormalized_logprob):
    # returns log Ber(b | mu)
    # unnormalized_logprob is log(mu / (1 - mu)
    # b must be 0 or 1
    s = b * 2 - 1
    return -np.logaddexp(0., -s * unnormalized_logprob)

def relu(x):    return np.maximum(0, x)

def init_net_params(scale, layer_sizes, rs=npr.RandomState(0)):
    """Build a (weights, biases) tuples for all layers."""
    return [(scale * rs.randn(m, n),   # weight matrix
             scale * rs.randn(n))      # bias vector
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

def neural_net_predict(params, inputs):
    """Params is a list of (weights, bias) tuples.
       inputs is an (N x D) matrix.
       Applies batch normalization to every layer but the last."""
    for W, b in params:
        outputs = np.dot(inputs, W) + b  # linear transformation
        inputs = np.tanh(outputs)           # nonlinear transformation
    return outputs

def nn_predict_gaussian(params, inputs):
    # Returns means and diagonal variances
    return unpack_gaussian_params(neural_net_predict(params, inputs))

###############################################################################################################

######## 1. Implementing the Model ########

###############################################################################################################


def log_prior(z):
    N=z.shape[0]
    return -0.5*N*np.log(2*np.pi)-0.5*np.sum(z**2, axis=0)
 
Dz = 2
Dh = 500
Ddata = 28**2

def decoder(z, theta):
    return neural_net_predict(theta, z)
 
def bernoulli_log_density(b, unnormalized_logprob):
    # returns log Ber(b | mu)
    # unnormalized_logprob is log(mu / (1 - mu)
    # b must be 0 or 1
    s = b * 2 - 1
    return -np.logaddexp(0., -s * unnormalized_logprob)

def log_likelihood(z,x,theta):
    return np.sum(bernoulli_log_density(x, decoder(z,theta)))
 
def joint_log_density(z,x,theta):
    return log_prior(z)+log_likelihood(z,x,theta)


###############################################################################################################

######## 2. Amortized Approximate Inference and training ########

###############################################################################################################


def encoder(x, theta):
    return nn_predict_gaussian(theta, x)

def log_q(q_mu, q_logsig, z):
    N=z.shape[0]
    q_sig=np.exp(q_logsig)
    return -0.5*N*np.log(2*np.pi)-0.5*np.sum(((z-q_mu)/q_sig)**2, axis=0)


def elbo(x, enc_params, dec_params):
    mu, logsig = encoder(x,enc_params)
    z = mu + np.exp(logsig)*np.random.randn(mu.shape[0],mu.shape[1])
    logp_estimate = joint_log_density(z,x,dec_params)
    logq_estimate = log_q(mu,logsig,z)
    return (logp_estimate-logq_estimate).mean()

def batch_indices(iter):
    idx = iter % num_batches
    return slice(idx * batch_size, (idx+1) * batch_size)

def loss(combined_params,iter):
    data_idx = batch_indices(iter)
    gen_params, rec_params = combined_params
    return -elbo(train_images[data_idx], rec_params, gen_params)

# Model hyper-parameters
latent_dim = 2
data_dim = 784  # How many pixels in each image (28x28).
gen_layer_sizes = [latent_dim, 500, data_dim]
rec_layer_sizes = [data_dim, 500, latent_dim * 2]

# Training parameters
param_scale = 0.01
batch_size = 200
num_epochs = 100
step_size = 0.001

# Load MNIST
print("Loading training data...")
from loadMNIST import load_mnist
N, train_images, train_labels, test_images, test_labels = load_mnist()
def binarise(images):
    on = images > 0.5
    images = images * 0.0
    images[on] = 1.0
    return images

print("Binarising training data...")
train_images = binarise(train_images)
test_images = binarise(test_images)

def init_net_params(scale, layer_sizes, rs=npr.RandomState(0)):
    """Build a (weights, biases) tuples for all layers."""
    return [(scale * rs.randn(m, n),   # weight matrix
             scale * rs.randn(n))      # bias vector
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

def vae_lower_bound(gen_params, rec_params, data, rs):
    # We use a simple Monte Carlo estimate of the KL
    # divergence from the prior.
    q_means, q_log_stds = encoder(data,rec_params)
    latents = q_means+np.exp(q_log_stds)*np.random.randn(q_means.shape[0], 
                                                         q_means.shape[1])
    log_q_z = log_q(q_means, q_log_stds, latents)
    log_prior_z = log_prior(latents)
    log_p_x_given_z = log_likelihood(latents,data,gen_params)
    elbo_estimate = (log_prior_z+log_p_x_given_z-log_q_z).mean()
    return elbo_estimate


init_gen_params = init_net_params(param_scale, gen_layer_sizes)
init_rec_params = init_net_params(param_scale, rec_layer_sizes)
combined_init_params = (init_gen_params, init_rec_params)

num_batches = int(np.ceil(len(train_images) / batch_size))

# Get gradients of loss using autograd.
loss_grad = grad(loss)

print("     Epoch     |         Loss       |    Test ELBO  ")
def print_perf(combined_params, iter, loss_grad):
    if iter % 1000 == 0:
        gen_params, rec_params = combined_params
        bound = np.mean(loss(combined_params, iter))
        message = "{:15}|{:20}|".format(iter//num_batches, bound)
        if iter % 5000 == 0:
            test_bound = -vae_lower_bound(gen_params, rec_params, 
                                          test_images, seed) / data_dim
            message += "{:20}".format(test_bound)
        print(message)

# The optimizers provided can optimize lists, tuples, or dicts of parameters.
optimized_params = adam(loss_grad, combined_init_params, step_size=step_size,
                        num_iters=num_epochs * num_batches, callback=print_perf)

opt_gen, opt_rec = optimized_params 
print("Final ELBO on Test set: ", -vae_lower_bound(opt_gen, opt_rec, 
                                                   test_images, seed) / data_dim)
pickle.dump(optimized_params, open("saved_params.p", "wb" ))


###############################################################################################################

######## 3. Visualizing Posteriors and Exploring the Model ########

###############################################################################################################

from loadMNIST import plot_images

samples=np.random.randn(10,Dz)
grscl_img = decoder(samples, opt_gen)
binary_img = binarise(grscl_img)
fig, ax = plt.subplots()
title='Samples from trained generative model using ancestral sampling'
ax.set_title(title)
plot_images(np.concatenate((grscl_img, binary_img), axis=0), ax, ims_per_row=10)


encoded_means, encoded_logsig=encoder(train_images, opt_rec)
fig, ax = plt.subplots()
title='Scatter plot of mean vectors in 2D latent space'
ax.set_title(title)
plt.scatter(encoded_means[:,0], encoded_means[:,1], c=train_labels, cmap='viridis')
plt.colorbar()

def lin_intpl(a,b,alpha):
    return alpha*a+(1-alpha)*b

img_list=[]
for pair in range(3):
    couple=np.concatenate((train_images[train_labels==pair],
                           train_images[train_labels==pair+5]),axis=0)
    enc_mean, enc_logsig=encoder(couple, opt_rec)
    for it in range(10):
        new_mean=lin_intpl(enc_mean[0],enc_mean[1],it/10)
        new_logsig=lin_intpl(enc_logsig[0],enc_logsig[1],it/10)
        z=new_mean+np.random.randn(1,Dz)*np.exp(new_logsig)
        img_list.append(decoder(z,opt_gen)[0])
        
fig, ax = plt.subplots()
title='Plot of the Bernoulli means along the interploation'
ax.set_title(title)
plot_images(np.asarray(img_list), ax, ims_per_row=10)


###############################################################################################################

######## 4. Predicting the Bottom of Images given the Top ########

###############################################################################################################



def skillcontour(f, title, fig, ax, colour=None):
    n = 100
    x = np.linspace(-3,3,n)
    y = np.linspace(-3,3,n)
    X,Y = np.meshgrid(x,y) # meshgrid for contour
    z_grid = np.vstack([X.reshape(X.shape[0]*X.shape[1]),
                        Y.reshape(Y.shape[0]*Y.shape[1])]) # add single batch dim
    Z = f(z_grid)
    Z=Z.reshape(n,n)
    max_z = np.max(Z)
    if max_z<=0:
        levels = max_z/np.array([.99, 0.9, 0.8, 0.7,0.6,0.5, 0.4, 0.3, 0.2])
    else: 
        levels = max_z*np.array([.99, 0.9, 0.8, 0.7,0.6,0.5, 0.4, 0.3, 0.2])
    
    if colour == None:
        p1 = ax.contour(X, Y, Z, levels[::-1])
    else:
        p1 = ax.contour(X, Y, Z, levels[::-1], colors=colour)
    ax.clabel(p1, inline=1, fontsize=10)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title(title)
    return

def plot_line_equal_skill():
    x=np.linspace(-3,3,200)
    y=np.linspace(-3,3,200)
    plt.plot(x,y,'g', label='Equal Line')
    plt.legend()
    return

def factorized_gaussian_log_density(mu,logsig,xs):
    sig = np.exp(logsig)
    return np.sum(-0.5*np.log(2*np.pi*sig**2)-0.5*((xs.T-mu)**2)/(sig**2), axis=1)

def top_half_log_jointt(z,x,theta):
    return log_prior(z)+top_half_ll(z.T,x,theta)


def top_half(x):
    return x[:14*28]
def top_half_ll(z,x,theta):
    return np.sum(bernoulli_log_density(x[:14*28], decoder(z,theta)[:14*28]))
def top_half_log_joint(z,x,theta):
    return log_prior(z)+top_half_ll(z,x,theta)


phi_mu = np.random.randn(1,2) # Initial mu, can initialize randomly!
phi_ls = np.random.randn(1,2) # Initial log_sigma, can initialize randomly!
init_params = np.asarray([phi_mu, phi_ls])

def top_half_elbo(init_params):
    # We use a simple Monte Carlo estimate of the KL
    # divergence from the prior.
    q_means = init_params[0]
    q_log_stds=init_params[1] 
    latents = q_means+np.exp(q_log_stds)*np.random.randn(q_means.shape[0], 
                                                         q_means.shape[1])
    log_q_z = log_q(q_means, q_log_stds, latents)
    elbo_estimate = (top_half_log_joint(latents,data,opt_gen)-log_q_z).mean()
    return elbo_estimate


data = train_images[60].reshape(1,train_images.shape[1])
num_itrs=200
lr= 0.01
verbose=0

params_cur = init_params

for i in range(num_itrs):
    from autograd import grad
    grad_params=grad(top_half_elbo)
    params_cur = params_cur + lr*grad_params(params_cur)
    if(verbose):
        print('Iteration: ', i)
        print('Current ELBO: ',top_half_elbo(params_cur))

q_means = params_cur[0]
q_log_stds = params_cur[1]
latents = q_means+np.exp(q_log_stds)*np.random.randn(q_means.shape[0], 
                                                     q_means.shape[1])

def cont_log_q(zs):
    return factorized_gaussian_log_density(q_means,q_log_stds,zs)
def cont_log_joint(zs):
    return top_half_log_jointt(zs,data,opt_gen)

fig, ax = plt.subplots()
title='Joint distribution in red and Approximate posterior in blue'
skillcontour(cont_log_q,title,fig,ax,colour='b')
skillcontour(cont_log_joint,title,fig,ax,colour='r')
plot_line_equal_skill()
plt.show()

im_list=[data[0]]
data[:14*28]=decoder(latents,opt_gen)
im_list.append(data[0])

fig, ax = plt.subplots()
title='Original (top) and Concatenated (bottom) Image'
ax.set_title(title)
plot_images(np.asarray(im_list), ax, ims_per_row=1)
