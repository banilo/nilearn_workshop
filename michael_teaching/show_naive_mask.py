mask = masker.mask_img_.get_data()
print(mask.shape)
plt.matshow(mask[20])

from nilearn.plotting import plot_roi
plot_roi(masker.mask_img_, bg_img=None)

