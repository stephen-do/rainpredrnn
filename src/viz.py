import PIL
import numpy as np
from matplotlib import pyplot as plt
import os
from mpl_toolkits.axes_grid1 import ImageGrid

ids = next(os.walk('samples'))[2]
fig = plt.figure(figsize=(8, 4), constrained_layout=True)
count = 1
for id in ids:
    an_image = PIL.Image.open("samples/" + id)
    grayscale_image = an_image.convert("L")
    grayscale_array = np.asarray(grayscale_image)
    # Adds a subplot at the 1st position

    fig.add_subplot(2, 4, count)

    # showing image
    plt.imshow(grayscale_array, cmap='gray')
    plt.axis('off')
    hour = id.split('.')[0][9:]
    plt.title(hour[0:2] + ":" + hour[2:4] + ":" + hour[4:], fontdict={'fontsize': 12})

    if count in [4, 8]:
        plt.colorbar(label='rain level')
    count += 1
plt.show()
fig.savefig("samples.png")

# fig, axes = plt.subplots(nrows=2, ncols=4, constrained_layout=True)
# for ax in axes.flat:
#     an_image = PIL.Image.open("samples/PHA200623065004.RAWV7AW.png")
#     grayscale_image = an_image.convert("L")
#     grayscale_array = np.asarray(grayscale_image)
#     im = ax.imshow(grayscale_array, cmap='gray')
#
# fig.subplots_adjust(right=0.8)
# cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
# fig.colorbar(im, cax=cbar_ax)

plt.show()
fig.savefig("samples.png")