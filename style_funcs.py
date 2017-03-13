import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
from scipy import misc
import matplotlib.pyplot as plt
import os.path


# Plot images
def show_images(img1, img2, img3):
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(img1)
    plt.title('Content Image')
    plt.subplot(1,3,2)
    plt.imshow(img2)
    plt.title('Style Image')
    plt.subplot(1,3,3)
    plt.imshow(img3)
    plt.title('Variable Image')
    plt.show()


# Load Image
def load_img(image_path,image_size=None):
    if image_size is None:
        im_H = 224
        im_W = 224
        image_size = (im_H,im_W)
    image = mpimg.imread(image_path)
    image = misc.imresize(image, image_size)
    image = image / 255.0
    image = image.astype(np.float32)
    return image


# Create output path
def gen_output_path(cont_p, style_p, path):
    k = 1
    output_fname = cont_p.rsplit('.', 1)[0] + '&' + style_p.rsplit('.', 1)[0]+str(k)
    while os.path.exists(path+output_fname):
        k += 1
        output_fname = output_fname[:-1]+str(k)
    return path+output_fname

# generate init img
def gen_init_img(H, W, content, n_rat):
    content = np.reshape(content, (H, W, 3))
    # init_img = np.random.normal(0, 0.001, size=[H, W, 3]).astype(np.float32)
    init_img = np.random.uniform(0.15, 0.85, size=[H, W, 3]).astype(np.float32)
    # samples = scipy.stats.truncnorm.rvs(
    #     (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma, size=N)
    return init_img*n_rat + content*(1.-n_rat)


# Get layer activations as constant tensors
def get_activations(sess, layers):
    activations = []
    acts = sess.run([layer for layer in layers])
    for act in acts:
        activations.append(tf.constant(act,dtype="float32"))
    return activations


def gram_matrix(layer):
    # Transform to channelsX(H*W) matrix
    layer_shape = layer.get_shape().as_list()
    # print(layer_shape)
    H, W, channels = [layer_shape[1], layer_shape[2],layer_shape[3]]
    N = H*W
    layer_mat = tf.reshape(layer, (N, channels))
    # Gram matrix
    gram = tf.matmul(tf.transpose(layer_mat), layer_mat)
    return gram


# product of dimension sizes of tensor
def dims_prod(tens):
    dims = np.array(tens.get_shape().as_list())
    return np.prod(dims)


# L2 content loss of tensor
def l2_content_loss(cont_tens, input_tens):
    l2 = tf.reduce_sum(tf.square(cont_tens-input_tens))
    dims = np.array(input_tens.get_shape().as_list())
    dims_prod = np.prod(dims)
    # print(norm_factor)
    norm_factor = 1./(2.*dims_prod**0.5)

    # print(norm_factor)
    return l2 * norm_factor


# L2 style loss of tensor
def l2_style_loss(style_tens, input_tens):
    A = gram_matrix(style_tens)
    G = gram_matrix(input_tens)
    l2 = tf.reduce_sum(tf.square(A-G))
    dims = np.array(input_tens.get_shape().as_list())
    dims_prod = np.prod(dims)
    norm_factor = 1./(4.*dims_prod**2.)

    return l2 * norm_factor


def total_variation_loss(img):
    # H regularizer
    img1 = img[:,1:,:,:]
    img2 = img[:,:-1,:,:]
    loss1 = tf.reduce_sum(tf.square(img1-img2)) #/ dims_prod(img1)
    # W regularizer
    img1 = img[:,:,1:,:]
    img2 = img[:,:,:-1,:]
    loss2 = tf.reduce_sum(tf.square(img1-img2)) #/ dims_prod(img1)
    tv_loss = loss1+loss2
    return tv_loss


def optimize_lbfgs(sess, optimizer, loss_, output_img, output_path, plot=False):
    i = 0

    def callback(loss_, output_img):  # callback for the l-bfgs optimizer step
        nonlocal i
        if i % 10 == 0:
            print('Loss at step %d: %f ' % (i, loss_))
        if i % 100 == 0 and i != 0:
            k = i
            misc.imsave(output_path + str(k) + '.jpg', output_img)
            if plot:
                plt.figure()
                plt.imshow(output_img)
                plt.show()
        i += 1

    optimizer.minimize(sess, fetches=[loss_, output_img], loss_callback=callback)
