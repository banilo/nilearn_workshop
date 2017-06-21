"""
Decoding with SpaceNet: face vs house object recognition
=========================================================

Here is a simple example of decoding with a SpaceNet prior (i.e Graph-Net,
TV-l1, etc.), reproducing the Haxby 2001 study on a face vs house
discrimination task.

See also the SpaceNet documentation: :ref:`space_net`.
"""

##############################################################################
# Load the Haxby dataset
from nilearn.datasets import fetch_haxby
data_files = fetch_haxby()

# Load Target labels
import numpy as np
labels = np.recfromcsv(data_files.session_target[0], delimiter=" ")


# Restrict to face and house conditions
target = labels['labels']
condition_mask = np.logical_or(target == "face", target == "house")

# Split data into train and test samples, using the chunks
condition_mask_train = np.logical_and(condition_mask, labels['chunks'] <= 6)
condition_mask_test = np.logical_and(condition_mask, labels['chunks'] > 6)

# Apply this sample mask to X (fMRI data) and y (behavioral labels)
# Because the data is in one single large 4D image, we need to use
# index_img to do the split easily
from nilearn.image import index_img
func_filenames = data_files.func[0]
X_train = index_img(func_filenames, condition_mask_train)
X_test = index_img(func_filenames, condition_mask_test)
y_train = target[condition_mask_train]
y_test = target[condition_mask_test]

# Compute the mean epi to be used for the background of the plotting
from nilearn.image import mean_img
background_img = mean_img(func_filenames)
background_img.to_filename("bg.nii.gz")

##############################################################################
# Fit SpaceNet with a Graph-Net penalty
from nilearn.decoding import SpaceNetClassifier

# Fit model on train data and predict on test data
decoder = SpaceNetClassifier(memory="nilearn_cache", penalty='graph-net', 
                             memory_level=2, screening_percentile=20.)
decoder.fit(X_train, y_train)
y_pred = decoder.predict(X_test)
accuracy = (y_pred == y_test).mean() * 100.
print("Graph-net classification accuracy : %g%%" % accuracy)

# Visualization
from nilearn.plotting import plot_stat_map, show
coef_img = decoder.coef_img_
plot_stat_map(coef_img, background_img,
              title="graph-net: accuracy %g%%" % accuracy,
              cut_coords=(-34, -16), display_mode="yz")

# Save the coefficients to a nifti file
coef_img.to_filename('haxby_graph-net_weights.nii')


# ##############################################################################
# # Now Fit SpaceNet with a TV-l1 penalty
# decoder = SpaceNetClassifier(memory="nilearn_cache", penalty='tv-l1',
#                              screening_percentile=100.)
# decoder.fit(X_train, y_train)
# y_pred = decoder.predict(X_test)
# accuracy = (y_pred == y_test).mean() * 100.
# print("TV-l1 classification accuracy : %g%%" % accuracy)

# # Visualization
# coef_img = decoder.coef_img_
# plot_stat_map(coef_img, background_img,
#               title="tv-l1: accuracy %g%%" % accuracy,
#               cut_coords=(-34, -16), display_mode="yz")

# # Save the coefficients to a nifti file
# coef_img.to_filename('haxby_tv-l1_weights.nii')


##############################################################################
# sklearn unstructured classifiers
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.base import clone
from sklearn2nilearn import SklearnEstimatorWrapper

high = decoder.alpha_grids_.max()
low = high * decoder.eps
C = np.reciprocal(np.logspace(np.log10(high), np.log10(low),
                              decoder.n_alphas))
masker = clone(decoder.masker_)
for estimator, params_grid in zip([SVC(kernel='linear'), RidgeClassifier(),
                                   LogisticRegression()],
                                  [dict(C=C), dict(C=C), dict(gamma=C)]):
    decoder_name = estimator.__class__.__name__.lower()
    decoder = SklearnEstimatorWrapper(GridSearchCV, masker=masker,
                                      estimator=estimator,
                                      param_grid=params_grid, n_jobs=2)
    decoder.fit(X_train, y_train)
    y_pred = decoder.predict(X_test)

    # Visualization
    accuracy = (y_pred == y_test).mean() * 100.
    coef_img = decoder.coef_img_
    plot_stat_map(coef_img, background_img,
                  title="%s: accuracy %g%%" % (decoder_name, accuracy),
                  cut_coords=(-34, -16), display_mode="yz")

    # Save the coefficients to a nifti file
    coef_img.to_filename('haxby_%s_weights.nii' % decoder_name)


show()
