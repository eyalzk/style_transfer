# Style Transfer Tensorflow Implementation

Tensorflow implementation of the paper ["A Neural Algorithm of Artistic Style"](https://arxiv.org/abs/1508.06576). 

The algorithm renders an image that keeps the content of one reference image while copying the style of another. For example:




<img src="images/result/eagle3/eagle&psy21.jpg" width="790">





There's plenty of great explanations on the web regarding how this works so I won't go into detail here... 


## Dependencies:
* [Tensorflow](https://www.tensorflow.org/install/) (version > 1.0)
* [tensorflow-vgg](https://github.com/machrisaa/tensorflow-vgg)

## Usage:
Most parameters can be configured when running using command line, though some parameters can currently be configured only through default_params.py.

To run using command line:

`python main.py`

All parameters are optional and have default values:

`--iterations <number of iterations>`, `--out_width <width of output image>`, `--content_path <content file path>`, `--style_path <style file path>`, `--result_path <Path of output image (with extension). If only a directory path is given, an automatic file name will be generated>`, `--noise_ratio <init image is noise_ratio*noise + (1-noise_ratio)*content_image>`, `--gamma <content/style ratio>`, `--beta <style weight>`, `--theta <tv loss weight>`, `--optimizer <'adam' or 'lbfgs'>`.

## Acknowledgements 
For the trained VGG19 I have used the implementation of [mechrisaa](https://github.com/machrisaa/tensorflow-vgg).


### Photo credits:

1. [Eagle](https://www.flickr.com/photos/jacobmeredith/)
2. [Psychedelic art](http://wallpaperspack.info/?p=52239)
