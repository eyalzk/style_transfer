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
# Im_H = 224
# Im_W = 224
num_iterations = 1001

Out_W = math.floor(620*1.5)
Out_H = math.floor(350*1.5)

cont_path = 'images\content\\'
style_path = 'images\style\\'
result_path = 'images\\result\\'
# Load Content Image
cont_fname = 'night_king.jpg'
cont_image = stylf.load_img(cont_path+cont_fname, (Out_H, Out_W))

# Load Style Image
style_fname = 'stars.jpg'
style_image = stylf.load_img(style_path+style_fname, (Out_H, Out_W))

# Output file name
output_path = stylf.gen_output_path(cont_fname, style_fname, result_path)

# Generate white noise image for initialization
noise_ratio = 0.9  # init image is noise_ratio*noise + (1-noise_ratio)*content_image
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


# with tf.device('/gpu:0'):
i = 0

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
    gamma = 4e-2  # content/style ratio
    beta = 1.0  # Style weight
    alpha = gamma*beta  # Content weight
    theta = 2e1  # tv loss coef'
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
    optimizer_type = 'lbfgs'
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









