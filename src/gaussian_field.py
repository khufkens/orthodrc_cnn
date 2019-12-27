import os, sys, time, random
import numpy as np
import matplotlib.pyplot as plt

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

r = random.uniform(-10,-5)
print(r)

out = gaussian_random_field(Pk = lambda k: k**r, size=256)
plt.figure()
plt.imshow(out.real, interpolation='none')
plt.show()
