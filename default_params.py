import math


def get_default_params():
    params = {}

    # File Paths
    params['content_path'] = 'images\content\\bear.jpg'
    params['style_path'] = 'images\style\\bullets.jpg'
    params['result_path'] = 'images\\result\\bear_bullets\\'

    # Training process
    params['num_iterations'] = 1001
    params['out_width'] = math.floor(500)  # width of output image
    params['optimizer_type'] = 'lbfgs'  # 'lbfgs' or 'adam'

    # Layers to collect activations to
    params['content_layer_names'] = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1', 'conv4_2']
    params['w_content'] = [0, 0, 0, 0, 0, 1]  # weights to apply for each layer
    params['style_layer_names'] = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
    params['w_style'] = [1, 1, 1, 1, 1]  # weights to apply for each layer

    # Stylization parameters
    params['noise_ratio'] = 0.6  # init image is noise_ratio*noise + (1-noise_ratio)*content_image
    params['gamma'] = 5e-1  # content/style ratio
    params['beta'] = 1.0  # Style weight
    params['theta'] = 1e1  # tv loss coef'

    return params


