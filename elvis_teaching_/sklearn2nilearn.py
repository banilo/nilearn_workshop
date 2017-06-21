# Wrapper for sklearn estimators. Useful for benchmarking structured vrs.
# unstructured estimators.


class SklearnEstimatorWrapper(object):
    """Lightweight nilearn wrapper for sklearn estimators.

    This is nothing but a sandbox for the underlying estimator,
    with the added feature of nifti-masking input data and unmasking of
    coefficients.

    Parameters
    ----------
    masker : NiftiMasker instance .
        The nifti masker object for doing the masking / unmasking.

    Attributes
    ----------
    coef_img_ : niimg instance
         The unmasked image of coefficients.

    """

    def __init__(self, estimator_cls, masker=None, **kwargs):
        self.estimator_cls = estimator_cls
        self.masker = masker
        self.estimator = estimator_cls(**kwargs)

        # steal the params of the underlying estimator
        for k, v in self.estimator.get_params().items():
            setattr(self, k, v)

    def get_params(self):
        """Params of underlying estimator, with the added coef_img_ attr
        of the raper"""
        params = self.estimator.get_params()
        params["masker"] = self.masker
        if hasattr(self, "coef_img_"):
            params["coef_img_"] = self.coef_img_
        return params

    def __repr__(self):
        return "Wrapped%s(%s)" % (
            self.estimator_cls.__name__,
            ", ".join(["%s=%s" % (k, v)
                       for k, v in self.get_params().items()]))

    def fit(self, X, y):
        """Mask input data, and then fit @ la sklearn."""
        if self.masker is not None:
            X = self.masker.fit_transform(X)
        self.estimator.fit(X, y)
        if self.masker is not None:
            if hasattr(self.estimator, "estimator"):
                coef = self.estimator.estimator.coef_
            else:
                coef = self.estimator.coef_
            self.coef_img_ = self.masker.inverse_transform(coef)
        return self.estimator

    def predict(self, X):
        """Mask input data, and then fit @ la sklearn."""
        if self.masker is not None:
            X = self.masker.transform(X)
        return self.estimator.predict(X)
