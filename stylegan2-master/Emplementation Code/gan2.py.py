
Make sure to run this code on a GPU instance.  GPU is assumed.

First, map your G-Drive, this is where your GANs will be written to.

## Examining the Latent Vector

Setup to use TF 1.0 (as required by nVidia).
"""

# Commented out IPython magic to ensure Python compatibility.
# Run this for Google CoLab (use TensorFlow 1.x)
# %tensorflow_version 1.x
from google.colab import files

"""Next, clone StyleGAN2 from GitHub."""

!git clone https://github.com/NVlabs/stylegan2.git

"""Verify that StyleGAN has been cloned."""

!ls /content/stylegan2/

"""# Run StyleGAN2 From Python Code

Add the StyleGAN folder to Python so that you can import it.  The code below is based on code from NVidia. This actually generates your images.
"""

import sys
sys.path.insert(0, "/content/stylegan2")

import dnnlib

import argparse
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import re
import sys
import os

import pretrained_networks

#----------------------------------------------------------------------------

def expand_seed(seeds, vector_size):
  result = []

  for seed in seeds:
    rnd = np.random.RandomState(seed)
    result.append( rnd.randn(1, vector_size) ) 
  return result

def generate_images(Gs, seeds, truncation_psi):
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    if truncation_psi is not None:
        Gs_kwargs.truncation_psi = truncation_psi

    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d/%d ...' % (seed_idx, len(seeds)))
        rnd = np.random.RandomState(seed=1)
        tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
        images = Gs.run(seed, None, **Gs_kwargs) # [minibatch, height, width, channel]
        path = f"/content/tmp/image-{seed_idx}.png"
        PIL.Image.fromarray(images[0], 'RGB').save(path)

"""Specify the two seeds that we wish to morph betwen."""

sc = dnnlib.SubmitConfig()
sc.num_gpus = 1
sc.submit_target = dnnlib.SubmitTarget.LOCAL
sc.local.do_not_copy_source_files = True
sc.run_dir_root = "/content/drive/My Drive/projects/stylegan2"
sc.run_desc = 'generate-images'
network_pkl = 'gdrive:networks/stylegan2-ffhq-config-f.pkl'

print('Loading networks from "%s"...' % network_pkl)
_G, _D, Gs = pretrained_networks.load_networks(network_pkl)
vector_size = Gs.input_shape[1:][0]

vec = expand_seed( [100,860], vector_size) # Morph between these two seeds

print(vec[0].shape)

"""We can now do a morph over 300 steps."""

STEPS = 300
diff = vec[1] - vec[0]
step = diff / STEPS
current = vec[0].copy()

vec2 = []
for i in range(STEPS):
  vec2.append(current)
  current = current + step

temp_path = "/content/tmp"

# Create a temporary directory to hold video frames
try:  
  os.mkdir(temp_path)
except OSError:  
  print("Temp dir already exists.")

generate_images(Gs, vec2,truncation_psi=0.5)

"""We are now ready to build the video."""

! ffmpeg -r 30 -i /content/tmp/image-%d.png -vcodec mpeg4 -y /content/gan_morph.mp4

"""Download the video."""

files.download('/content/gan_morph.mp4')