import os, random
import numpy as np
import matplotlib.pyplot as plt
import cv2

# see this stack exchange for the RGF code details:
# https://mathematica.stackexchange.com/questions/4829/
# efficiently-generating-n-d-gaussian-random-fields/9951#9951

def fft_idx(n):
  a = list(range(0, int(n/2+1)))
  b = list(range(1, int(n/2)))
  b.reverse()
  b = [-i for i in b]
  return a + b
  
def gaussian_random_field(Pk = lambda k : k**-3.0, size = 100):
  def Pk2(kx, ky):
    if kx == 0 and ky == 0:
      return 0.0
    return np.sqrt(Pk(np.sqrt(kx**2 + ky**2)))

  noise = np.fft.fft2(np.random.normal(size = (size, size)))
  amplitude = np.zeros((size,size))

  for i, kx in enumerate(fft_idx(size)):
    for j, ky in enumerate(fft_idx(size)):            
      amplitude[i, j] = Pk2(kx, ky)
  return np.fft.ifft2(noise * amplitude)

def alpha_blend(img1, img2, mask):
    """ alphaBlend img1 and img 2 (of CV_8UC3) with mask (CV_8UC1 or CV_8UC3)
    """
    if mask.ndim==3 and mask.shape[-1] == 3:
        alpha = mask/255.0
    else:
        alpha = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)/255.0
    blended = cv2.convertScaleAbs(img1*(1-alpha) + img2*alpha)
    return blended


def generate_samples(
  images_dir = './data/source_images',
  output_dir = './data/synth',
  subset = 'train',
  samples = 1000,
  split = 0.2):
  
  # list images
  dist_images = list(filter(lambda x:subset + '_dist' in x, os.listdir(images_dir)))
  dist_images = [os.path.join(images_dir, image_id) for image_id in dist_images]
  forest_images = list(filter(lambda x:subset + '_forest' in x, os.listdir(images_dir)))
  forest_images = [os.path.join(images_dir, image_id) for image_id in forest_images]
  
  # after the split percentage of mixed samples
  # between forest and disturbance tiles is covered
  # create some split tiles which involve mixed forest
  # forest tiles, this to learn faulty stitching of
  # Agisoft Metashape
  forest_mix = samples - (samples * split)
  
  # shuffled list of increments
  ind = list(range(samples))
  random.shuffle(ind)
  
  # loop over all samples
  for i, v in enumerate(ind) :
    
    # compose filename prefixes
    composite_name = subset + '_' + str(i)
    
    # read forest image
    forest_image = forest_images[random.randint(0,len(forest_images)-1)]
    
    # depending on the counter select either a disturbance
    # or another forest tile to create a composite
    if v < forest_mix :
      dist_image = dist_images[random.randint(0,len(dist_images)-1)]
    else:
      dist_image = forest_images[random.randint(0,len(forest_images)-1)]
      
    # read random images
    forest = cv2.imread(forest_image)
    dist = cv2.imread(dist_image)
    row, col, z = dist.shape
    
    # create a field the size of the (square) image
    field = gaussian_random_field(Pk = lambda k: k**random.uniform(-10,-5), size=row)
    mask = field.real > 0
    
    # change export type depending on output
    if v < forest_mix :
      mask_export = mask.astype(np.uint8) + 1
    else:
      mask_export = np.ones(shape=mask.shape)
      
    # convert mask
    mask = mask.astype(np.uint8) * 255
  
    # set a random smoothing window size
    w = random.randint(15,55)
    if (w % 2) == 0 :
      w = w + 1
  
    # blur the mask for smooth edges
    mask = cv2.GaussianBlur(mask, (w,w), cv2.BORDER_DEFAULT)
  
    # blend two images
    blended = alpha_blend(forest, dist, mask)
  
    # write to disk
    cv2.imwrite(output_dir + '/images/' + composite_name + '.png',
    blended, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    
    
    cv2.imwrite(output_dir + '/labels/' + composite_name + '.png',
    mask_export, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

if __name__ == "__main__":
  
  # generate training samples (2000 of which 400 are forest + forest)
  generate_samples(
    subset = 'train',
    samples = 5000,
    split = 0.1,
    output_dir = "../data/synthetic_images/")
  
  # generate validation samples (200 of which 40 are forest + forest)
  generate_samples(
    subset = 'val',
    samples = 500,
    split = 0.1,
    output_dir = "../data/synthetic_images/")
  
  # generate testing samples (200 of which 40 are forest + forest)
  generate_samples(
    subset = 'test',
    samples = 500,
    split = 0.1,
    output_dir = "../data/synthetic_images/")
