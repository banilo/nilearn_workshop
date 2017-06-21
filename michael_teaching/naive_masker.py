from nilearn.input_data import NiftiMasker
masker = NiftiMasker()
masker.fit(func_file)

