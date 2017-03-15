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

def main(args):
    # Im_H = 224
    # Im_W = 224
    num_iterations = args.iterations

    Out_W = args.out_width
    # Out_H = math.floor(1365*0.2)

    cont_path = 'images\content\\'
    style_path = 'images\style\\'
    result_path = 'images\\result\\'
    # Load Content Image
    cont_fname = args.content_name
    cont_image = stylf.load_img(cont_path+cont_fname, width=Out_W)
    Out_H = cont_image.shape[0]
    # print(cont_image.shape)
    # Load Style Image
    style_fname = args.style_name
    style_image = stylf.load_img(style_path+style_fname, width=Out_W, height=Out_H)

    # Output file name
    if args.output_name is None:
        output_path = stylf.gen_output_path(cont_fname, style_fname, result_path)
    else:
        output_path = result_path+args.output_name
    print('Output path: '+output_path)
    # Generate white noise image for initialization
    noise_ratio = args.noise_ratio  # init image is noise_ratio*noise + (1-noise_ratio)*content_image
    init_img = stylf.gen_init_img(Out_H, Out_W, cont_image, noise_ratio)

    # Plot images
    stylf.show_images(cont_image, style_image, init_img)

    # Reshape to batch form
    cont_image_batch = np.expand_dims(cont_image, 0)
    style_image_batch = np.expand_dims(style_image, 0)
    init_img_batch = np.expand_dims(init_img, 0)


    # Prepare graph
    graph = tf.Graph()
    with graph.as_default():
        print('='*70 + 'INITIALIZING' + '='*70)
        # output image:
        # input_img = tf.placeholder("float",[1, Out_H, Out_W, 3])
        input_img = tf.Variable(init_img_batch, dtype="float32")
        # input_img = tf.Variable(tf.truncated_normal(cont_image_batch.shape, 0.5, 0.001), dtype="float32")
        output_img = tf.clip_by_value(tf.squeeze(input_img, [0]), 0, 1)

        # Initialize VGG nets for each image
        vgg_cont = vgg_clipped.Vgg19()
        vgg_style = vgg_clipped.Vgg19()
        vgg_ref = vgg_clipped.Vgg19()

        with tf.name_scope("cont_vgg"):
            vgg_cont.build(cont_image_batch)
        with tf.name_scope("style_vgg"):
            vgg_style.build(style_image_batch)
        with tf.name_scope("ref_vgg"):
            vgg_ref.build(input_img)
        # images = tf.placeholder("float", [1, Im_H, Im_W, 3])

    with tf.Session(graph=graph, config=tf.ConfigProto(log_device_placement=False)) as sess:

        # Get content and style representations
        cont_layers = [vgg_cont.conv1_1,vgg_cont.conv2_1,vgg_cont.conv3_1,vgg_cont.conv4_1,vgg_cont.conv5_1, vgg_cont.conv4_2]
        [cont1_1, cont2_1, cont3_1, cont4_1, cont5_1, cont4_2] = stylf.get_activations(sess, cont_layers)
        style_layers = [vgg_style.conv1_1,vgg_style.conv2_1,vgg_style.conv3_1,vgg_style.conv4_1,vgg_style.conv5_1]
        [style1_1, style2_1, style3_1, style4_1, style5_1] = stylf.get_activations(sess, style_layers)

        # Input activations
        input_activations = [vgg_ref.conv1_1, vgg_ref.conv2_1, vgg_ref.conv3_1, vgg_ref.conv4_1, vgg_ref.conv5_1, vgg_ref.conv4_2]

        # Content loss
        cont_activations = np.stack([cont1_1, cont2_1, cont3_1, cont4_1, cont5_1, cont4_2])
        w_content = np.array([0, 0, 0, 0, 0, 1]) * 1.0 # Weights for each layer
        w_content /= np.sum(w_content)
        L_content = []
        selected_layers = ''
        for idx, w in enumerate(w_content):
            if w != 0:
                L_content.append(tf.constant(w, dtype="float32") *
                                 stylf.l2_content_loss(cont_activations[idx], input_activations[idx]))
                selected_layers += ', '+cont_layers[idx].name#', conv1_'+str(idx+1)
        print('Content loss based on layers: ', selected_layers)
        content_loss = tf.add_n(L_content)  # sum weighted content losses

        # Style loss
        style_activations = np.stack([style1_1, style2_1, style3_1, style4_1, style5_1])
        # style_image_matrices = [stylf.gram_matrix(tens) for tens in [style1_1, style2_1, style3_1, style4_1, style5_1]]
        # input_style_matrices = [stylf.gram_matrix(tens) for tens in input_activations[:5]]

        w_style = np.array([1, 1, 1, 1, 1]) * 1.0  # Weights for each layer
        w_style /= np.sum(w_content)
        L_style = []
        selected_layers = ''
        for idx, w in enumerate(w_style):
            if w != 0:
                # norm_factor = np.array(input_activations[idx].get_shape().as_list())
                # print(norm_factor)
                # norm_factor = np.prod(norm_factor) ** 2.0

                L_style.append(tf.constant(w, dtype="float32") *
                                 stylf.l2_style_loss(style_activations[idx], input_activations[idx]))
                selected_layers += ', ' + style_layers[idx].name  # ', conv1_'+str(idx+1)
        print('Style loss based on layers: ', selected_layers)
        style_loss = tf.add_n(L_style)  # sum weighted content losses
        # content_loss = tf.constant(0, dtype="float32") #Temporary

        # Total Variation loss
        tv_loss = stylf.total_variation_loss(input_img)

        # Clear redundant VGGs for memory
        del vgg_style
        del vgg_cont

        # Content - Style Ratio
        gamma = args.gamma  # content/style ratio
        beta = args.beta  # Style weight
        alpha = gamma*beta  # Content weight
        theta = args.theta  # tv loss coef'
        # debug
        # alpha = 0.
        # beta = 1.
        # Total loss
        loss = tf.constant(alpha, dtype="float32")*content_loss + tf.constant(beta, dtype="float32")*style_loss + tf.constant(theta, dtype="float32")*tv_loss

        # Learning Rate
        # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # global_step = tf.Variable(0)

        # Lower the learning rate by factor of 0.2 each 50 iterations
        LR = tf.train.exponential_decay(0.02, global_step, 2000, 0.6, staircase=True)
        # LR = tf.train.exponential_decay(100.0, global_step, 400, 0.6, staircase=True)

        # Optimizer
        optimizer_type = args.optimizer
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
                    output = sess.run(output_img)
                    plt.figure()
                    plt.imshow(output)
                    misc.imsave(output_path+str(step)+'.jpg', output)
                    plt.show()

        elif optimizer_type == 'lbfgs':
            stylf.optimize_lbfgs(sess, optimizer, loss, output_img, output_path)

        # Save output file
        output = sess.run(output_img)
        stylf.show_images(cont_image, style_image, output)
        misc.imsave(output_path+'.jpg', output)


if __name__ == '__main__':
    """

    """

    # Arguments for internal use (not through command line):
    num_iterations_ = 1001
    Out_W_ = math.floor(200)  # width of output image

    cont_fname_ = 'Human_eye.jpg'
    style_fname_ = 'stars.jpg'
    noise_ratio_ = 0.9  # init image is noise_ratio*noise + (1-noise_ratio)*content_image
    # Content - Style Ratio
    gamma_ = 4e-2  # content/style ratio
    beta_ = 1.0  # Style weight
    theta_ = 2e1  # tv loss coef'

    optimizer_type_ = 'lbfgs'

    parser = argparse.ArgumentParser(description='Main script for rendering a style transfer')

    parser.add_argument('--iterations', type=int,
                        default=num_iterations_,
                        help='Number of iterations. (default: %(default)s)')

    parser.add_argument('--out_width', type=int,
                        default=Out_W_,
                        help='width of output image. (default: %(default)s)')

    parser.add_argument('--content_name', type=str,
                        default=cont_fname_,
                        help='File name of content image. image must be in images/content')

    parser.add_argument('--style_name', type=str,
                        default=style_fname_,
                        help='File name of style image. image must be in images/content')

    parser.add_argument('--output_name', type=str,
                        default=None,
                        help='File name of output image (no suffix). image will be saved in images/results')

    parser.add_argument('--noise_ratio', type=float,
                        default=noise_ratio_,
                        help='init image is noise_ratio*noise + (1-noise_ratio)*content_image. (default: %(default)s)')

    parser.add_argument('--gamma', type=float,
                        default=gamma_,
                        help='content/style ratio. (default: %(default)s)')

    parser.add_argument('--beta', type=float,
                        default=beta_,
                        help='style weight. (default: %(default)s)')

    parser.add_argument('--theta', type=float,
                        default=theta_,
                        help='tv loss coef. (default: %(default)s)')

    parser.add_argument('--optimizer', type=str,
                        default=optimizer_type_,choices=['lbfgs', 'adam'],
                        help='optimizer. (default: %(default)s)')

    args = parser.parse_args()

    main(args)






