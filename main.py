from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import vgg_clipped
import style_funcs as stylf
from scipy import misc
import math
import argparse
from default_params import get_default_params
import os.path


def main(run_args):
    
    # Number of iterations
    num_iterations = run_args.iterations

    # width of output image
    out_W = run_args.out_Width

    # Names of layers to extract
    content_layer_names = run_args.content_layer_names
    style_layer_names = run_args.style_layer_names

    # Weights to apply to each layer in calculation of the loss
    w_content = np.array(run_args.w_content) * 1.0  # Weights for each layer
    w_content /= np.sum(w_content)

    w_style = np.array(run_args.w_style) * 1.0
    w_style /= np.sum(w_style)

    # Load Content Image, resize for width to fit desired output width
    content_path = run_args.content_path
    content_image = stylf.load_img(content_path, width=out_W)
    
    # Extract needed output image height to maintain aspect ratio
    out_H = content_image.shape[0]
    
    # Load Style Image
    style_path = run_args.style_path
    style_image = stylf.load_img(style_path, width=out_W, height=out_H)

    # Output file path
    result_path = run_args.result_path
    _, file_extension = os.path.splitext(result_path)
    if not file_extension:  # if only directory was given, generate a file name
        if not os.path.isdir(result_path):
            os.mkdir(result_path)
        result_path = stylf.gen_result_path(content_path, style_path, result_path)
    print('Output path: '+result_path)

    # Generate white noise image for initialization
    noise_ratio = run_args.noise_ratio  # init image is noise_ratio*noise + (1-noise_ratio)*content_image
    init_img = stylf.gen_init_img(out_H, out_W, content_image, noise_ratio)

    # Plot images
    stylf.show_images(content_image, style_image, init_img)

    # Prepare Tensorflow graph
    graph = tf.Graph()
    with graph.as_default():
        print('='*70 + 'INITIALIZING' + '='*70)

        input_img = tf.Variable(np.expand_dims(init_img, 0), dtype="float32")
        result_img = tf.clip_by_value(tf.squeeze(input_img, [0]), 0, 1)

        # Initialize VGG net
        vgg_net = vgg_clipped.Vgg19()

    with tf.Session(graph=graph, config=tf.ConfigProto(log_device_placement=False)) as sess:

        # Get content and style representations
        content_activations = stylf.get_activations\
            (sess=sess, net=vgg_net, layer_names=content_layer_names, img=np.expand_dims(content_image, 0))
        style_activations = stylf.get_activations\
            (sess=sess, net=vgg_net, layer_names=style_layer_names, img=np.expand_dims(style_image, 0))

        # Input activations
        input_layer_names = list(set(content_layer_names+style_layer_names))  # list of all needed layers
        # build net using input image
        vgg_net.build(input_img, reset_dict=True)
        # Get tensors of desired layers
        input_activations = dict(zip(input_layer_names, [getattr(vgg_net, layer_name) for layer_name in input_layer_names]))

        # Content loss
        content_loss = tf.add_n([w*stylf.l2_content_loss(content_activations[content_layer_names[i]],
                                                       input_activations[content_layer_names[i]])
                                                       for i, w in enumerate(w_content) if w != 0])

        print('Content loss based on layers: ', ", ".join([ l_name for i, l_name in enumerate(content_layer_names) if w_content[i]]))

        style_loss = tf.add_n([w*stylf.l2_style_loss(style_activations[style_layer_names[i]],
                                                       input_activations[style_layer_names[i]])
                                                       for i, w in enumerate(w_style) if w != 0])

        print('Style loss based on layers: ', ", ".join([ l_name for i, l_name in enumerate(style_layer_names) if w_style[i]]))

        # Total Variation loss
        tv_loss = stylf.total_variation_loss(input_img)

        # Content - Style Ratio
        gamma = run_args.gamma  # content/style ratio
        beta = run_args.beta  # Style weight
        alpha = gamma*beta  # Content weight
        theta = run_args.theta  # tv loss coef'

        # Total loss
        loss = tf.constant(alpha, dtype="float32")*content_loss + tf.constant(beta, dtype="float32")*style_loss +\
                                                                            tf.constant(theta, dtype="float32")*tv_loss

        # Learning Rate
        # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # global_step = tf.Variable(0)

        # Lower the learning rate by factor of 0.2 each 50 iterations
        LR = tf.train.exponential_decay(0.02, global_step, 2000, 0.6, staircase=True)
        # LR = tf.train.exponential_decay(100.0, global_step, 400, 0.6, staircase=True)

        # Optimizer
        optimizer_type = run_args.optimizer
        # optimizer = tf.train.GradientDescentOptimizer(LR).minimize(loss, global_step=global_step)
        if optimizer_type == 'adam':
            optimizer = tf.train.AdamOptimizer(LR).minimize(loss, global_step=global_step)
        elif optimizer_type == 'lbfgs':
            optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss, method='L-BFGS-B',
                        options={'maxiter': num_iterations})

        print('='*70 + 'INITIALIZATION COMPLETE' + '='*70)

        tf.global_variables_initializer().run()
        if optimizer_type == 'adam':
            for step in range(num_iterations):
                _, l, lrate = sess.run([optimizer, loss, LR])
                if step % 10 == 0:
                    print('Loss at step %d: %f  ;  Learning Rate: %f' % (step, l, lrate))
                    [l1, l2, l3] = sess.run([content_loss, style_loss, tv_loss])
                    print('C_loss: %f, S_loss: %f, tv_loss: %f' % (alpha*l1, beta*l2, theta*l3))
                if step % 100 == 0 and step != 0:
                    result = sess.run(result_img)
                    plt.figure()
                    plt.imshow(result)
                    misc.imsave(result_path+str(step)+'.jpg', result)
                    plt.show()

        elif optimizer_type == 'lbfgs':
            stylf.optimize_lbfgs(sess, optimizer, loss, result_img, result_path)

        # Save output file
        result = sess.run(result_img)
        stylf.show_images(content_image, style_image, result)
        misc.imsave(result_path, result)


if __name__ == '__main__':
    """

    """
    # Get default parameters
    params = get_default_params()

    parser = argparse.ArgumentParser(description='Main script for rendering a style transfer')

    parser.add_argument('--iterations', type=int,
                        default=params['num_iterations'],
                        help='Number of iterations. (default: %(default)s)')

    parser.add_argument('--out_Width', type=int,
                        default=params['out_width'],
                        help='width of output image. (default: %(default)s)')

    parser.add_argument('--content_path', type=str,
                        default=params['content_path'],
                        help='Path of content image.')

    parser.add_argument('--style_path', type=str,
                        default=params['style_path'],
                        help='Path name of style image.')

    parser.add_argument('--result_path', type=str,
                        default=params['result_path'],
                        help='Path of output image (with extension). If only a directory path is given, an automatic '
                             'file name will be generated')

    parser.add_argument('--noise_ratio', type=float,
                        default=params['noise_ratio'],
                        help='init image is noise_ratio*noise + (1-noise_ratio)*content_image. (default: %(default)s)')

    parser.add_argument('--gamma', type=float,
                        default=params['gamma'],
                        help='content/style ratio. (default: %(default)s)')

    parser.add_argument('--beta', type=float,
                        default=params['beta'],
                        help='style weight. (default: %(default)s)')

    parser.add_argument('--theta', type=float,
                        default=params['theta'],
                        help='tv loss coef. (default: %(default)s)')

    parser.add_argument('--optimizer', type=str,
                        default=params['optimizer_type'], choices=['lbfgs', 'adam'],
                        help='optimizer. (default: %(default)s)')

    parser.add_argument('--content_layers', type=str,
                        default=params['optimizer_type'], choices=['lbfgs', 'adam'],
                        help='optimizer. (default: %(default)s)')


    args = parser.parse_args()

    # Add selected layer names and weights Todo: Make this also configurable via command line, need to transform the input into a list
    args.content_layer_names = params['content_layer_names']
    args.style_layer_names = params['style_layer_names']
    args.w_content = params['w_content']
    args.w_style = params['w_style']

    main(args)






