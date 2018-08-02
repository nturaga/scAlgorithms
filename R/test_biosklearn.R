## BiocSklearn testing in R

library(reticulate)

reticulate::py_install(c("sklearn","numpy","scipy","pandas","h5py"),
                       envname="BiocSklearn", method="virtualenv")

reticulate::use_virtualenv("BiocSklearn")

#reticulate::virtualenv_install("BiocSklearn", packages=c("numpy", "scipy","sklearn","pandas", "h5py"))

library(BiocSklearn)


SklearnEls() ## sklearn is a different package


## NMF

kidney_droplet_sce <- readRDS(file="kidney_droplet.rds")

nmf = SklearnEls()$skd$NMF

X = counts(kidney_droplet_sce)

model <- nmf(n_components=3L, init="random", random_state=0L)

W <- model$fit_transform(X)
H <- model$components_

## PCA

pca <- SklearnEls()$skd$PCA

