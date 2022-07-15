import pywt
import pywt.data
import matplotlib.pyplot as plt
import numpy as np

# Load image
original = pywt.data.camera()

# Wavelet transform of image, and plot approximation and details
titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
coeffs2 = pywt.dwt2(original, 'bior1.3')
LL, (LH, HL, HH) = coeffs2
fig = plt.figure(figsize=(10, 10*4))
for i, a in enumerate([LL, LH, HL, HH]):
    ax = fig.add_subplot(4, 1, i + 1)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
plt.show()


undersample_rate = .5
n = original.shape[0] * original.shape[1]
original_undersampled = ( original.reshape(-1) \
    * np.random.permutation(
        np.concatenate(
            (np.ones( int( n * undersample_rate ) ),
             np.zeros( int( n * ( 1-undersample_rate )) ))
        )
    )
                        ).reshape(512,512)


fig,ax = plt.subplots(1,2,figsize=(20,10))
ax[0].imshow(original, cmap=plt.cm.gray)
ax[1].imshow(original_undersampled,cmap=plt.cm.gray)
plt.show()