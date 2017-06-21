masker = NiftiMasker(mask_strategy='epi')
masker.fit(func_file)

mask = masker.mask_img_.get_data()
print(mask.shape)
plt.matshow(mask[20])

plot_roi(masker.mask_img_)

