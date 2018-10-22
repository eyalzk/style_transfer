import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
from scipy import misc
import matplotlib.pyplot as plt
import os.path
import math


# Plot images
def show_images(images):
    plt.figure()
    titles = ['Content Image', 'Style Image', 'Variable Image']
    for i, img in enumerate(images,1):

        plt.subplot(1, 3, i)
        plt.imshow(img)
        plt.title(titles[i-1])
    plt.show()


# Load Image and resize
def load_img(image_path, width=None, height=None):
    image = mpimg.imread(image_path)

    if width is None and height is None:  # output width and height are nor specified
        im_h = 224
        im_w = 224
    elif height is None:  # only width is specified. set height to maintain aspect ratio
        im_w = width
        imshape = image.shape
        rat = im_w/imshape[1]
        im_h = math.floor(imshape[0]*rat)
    elif width is None:
        im_h = height
        imshape = image.shape
        rat = im_h/imshape[0]
        im_w = math.floor(imshape[1]*rat)
    else:  # Custom width & height
        im_h = height
        im_w = width

    image = (misc.imresize(image, (im_h, im_w)) / 255.0).astype(np.float32)
    return image


# Create output path
def gen_result_path(cont_p, style_p, path):
    k = 1
    content_f = os.path.basename(cont_p)
    style_f = os.path.basename(style_p)

    output_fname_nonum = content_f.rsplit('.', 1)[0] + '&' + style_f.rsplit('.', 1)[0]
    output_fname = output_fname_nonum+str(k)
    while os.path.exists(path+output_fname+'.jpg'):
        k += 1
        output_fname = output_fname_nonum+str(k)
    return path+output_fname+'.jpg'


# generate init img
def gen_init_img(height, width, content, noise_ratio):
    ''' generate an initialization image - a mixture of the content image and random noise'''
    content = np.reshape(content, (height, width, 3))

    # init_img = np.random.normal(0, 0.001, size=[H, W, 3]).astype(np.float32)
    init_img = np.random.uniform(0.15, 0.85, size=[height, width, 3]).astype(np.float32)

    return init_img*noise_ratio + content*(1.-noise_ratio)


# Get layer activations as constant tensors
def get_activations(sess, net, layer_names, img):
    # Returns a dictionary with layer names and activation tensors
    print('Getting Activations...')
    # build net using given image
    net.build(img, reset_dict=False)
    # Get tensors of desired layers
    layers_tensors = [getattr(net, layer_name) for layer_name in layer_names]
    # get activations and return tf.constants
    acts = sess.run([layer for layer in layers_tensors])
    activations = [tf.constant(act,dtype="float32") for act in acts]

    return dict(zip(layer_names,activations))


# Calculate Gram matrix
def gram_matrix(layer):
    # Transform to channelsX(H*W) matrix
    layer_shape = layer.get_shape().as_list()
    # print(layer_shape)
    height, width, channels = [layer_shape[1], layer_shape[2], layer_shape[3]]
    n = height*width
    layer_mat = tf.reshape(layer, (n, channels))
    # Gram matrix
    gram = tf.matmul(tf.transpose(layer_mat), layer_mat)
    return gram


# product of dimension sizes of tensor
def dims_prod(tensor):
    dims = np.array(tensor.get_shape().as_list())
    return np.prod(dims)


# L2 content loss of tensor
def l2_content_loss(cont_tens, input_tens):
    l2 = tf.reduce_sum(tf.square(cont_tens-input_tens))
    dims_product = dims_prod(input_tens)
    norm_factor = 1./(2.*dims_product**0.5)
    return l2 * norm_factor


# L2 style loss of tensor
def l2_style_loss(style_tens, input_tens):
    A = gram_matrix(style_tens)
    G = gram_matrix(input_tens)
    l2 = tf.reduce_sum(tf.square(A-G))
    dims_product = dims_prod(input_tens)
    norm_factor = 1./(4.*dims_product**2.)
    return l2 * norm_factor


def total_variation_loss(img):
    shift = 2
    # H regularizer
    img1 = img[:, shift:, :, :]
    img2 = img[:, :-shift, :, :]
    loss1 = tf.reduce_sum(tf.square(img1-img2)) #/ dims_prod(img1)
    # W regularizer
    img1 = img[:, :, shift:, :]
    img2 = img[:, :, :-shift, :]
    loss2 = tf.reduce_sum(tf.square(img1-img2)) #/ dims_prod(img1)
    tv_loss = loss1+loss2
    return tv_loss


def optimize_lbfgs(sess, optimizer, loss, output_img, output_path, plot=False):
    iteration_counter = 0

    def callback(loss_, output_img_):  # callback for the l-bfgs optimizer step
        nonlocal iteration_counter
        if iteration_counter % 10 == 0:
            print('Loss at step %d: %f ' % (iteration_counter, loss_))
        if iteration_counter % 100 == 0 and iteration_counter != 0:
            # k = iteration_counter
            save_path = output_path.rsplit('.',1)[0] + str(iteration_counter) + '.jpg'
            misc.imsave(save_path, output_img_)
            if plot:
                plt.figure()
                plt.imshow(output_img_)
                plt.show()
        iteration_counter += 1

    optimizer.minimize(sess, fetches=[loss, output_img], loss_callback=callback)


def optimize_adam(sess, optimizer, losses, lr, result_img, result_path, num_iterations):

    for step in range(num_iterations):
        _, loss_eval, lr_eval = sess.run([optimizer, losses['total_loss'], lr])
        if step % 10 == 0:
            print('Loss at step %d: %f  ;  Learning Rate: %f' % (step, loss_eval, lr_eval))
            [l1, l2, l3] = sess.run([losses['content_loss'], losses['style_loss'], losses['tv_loss']])
            print('C_loss: %f, S_loss: %f, tv_loss: %f' % (l1, l2, l3))
        if step % 100 == 0 and step != 0:
            result = sess.run(result_img)
            plt.figure()
            plt.imshow(result)
            misc.imsave(result_path+str(step)+'.jpg', result)
            plt.show()

