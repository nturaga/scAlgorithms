## Original function from Rtsne
runTSNE <- 
    function(object,
             ncomponents = 2,
             ntop = 500,
             feature_set = NULL,
             exprs_values = "logcounts",
             scale_features = TRUE,
             use_dimred = NULL,
             n_dimred = NULL,
             rand_seed = NULL,
             perplexity = min(50, floor(ncol(object)/5)),
             pca = TRUE,
             initial_dims = 50,
             ...) 
{
    if (!is.null(use_dimred)) {
        dr <- reducedDim(object, use_dimred)
        if (!is.null(n_dimred)) {
            dr <- dr[, seq_len(n_dimred), drop = FALSE]
        }
        vals <- dr
        pca <- FALSE
    }
    else {
        vals <- scater:::.get_highvar_mat(object, exprs_values = exprs_values, 
                                 ntop = ntop, feature_set = feature_set)
        vals <- scater:::.scale_columns(vals, scale = scale_features)
        initial_dims <- min(initial_dims, ncol(object))
    }
    if (!is.null(rand_seed)) {
        set.seed(rand_seed)
    }
    vals <- as.matrix(vals)
    
    ## tsne_biocsklearn
    tsne_out <- tsne_BiocSklearn(vals,
                                 initial_dims,
                                 init = 'pca',
                                 perplexity = perplexity,
                                 dims = ncomponents)

    reducedDim(object, "TSNE") <- tsne_out
    return(object)
}


##tsne_out <- Rtsne_tensorflow(vals, initial_dims = initial_dims, 
##    pca = pca, perplexity = perplexity, dims = ncomponents, 
##    check_duplicates = FALSE, ...)

#' Rtsne implementation in BiocSklearn
tsne_BiocSklearn <-
    function(vals, initial_dims,
             perplexity, dims,
             check_duplicates = FALSE, ...)
{
    ## Get model from BiocSklearn
    sklearn_manifold <- reticulate::import("sklearn.manifold")
    sklearn_tsne <- sklearn_manifold$TSNE
    res <-
        sklearn_tsne(n_components = as.integer(dims), perplexity = perplexity,
                     init=initial_dims)$fit_transform(vals)
    res
}


## TODO:
## library(microbenchmark)
## microbenchmark::time
## compare, BiocSklearn_tsne, Rtsne, tensorflow_tsne

## ## Test
library(BiocSklearn)
library(scater)
library(Rtsne)
sce <- readRDS("data/tsne_sce.Rds")

debug(runTSNE)

runTSNE(sce, n_components=2L)

## ## Test tsne with small matrix
## X <- apply(X=matrix(0, 4, 3), MARGIN=c(1,2), FUN=function(x) sample(c(0,1), 1))
## X_embedded = sklearn_tsne(n_components=2L)$fit_transform(X)
## X_embedded.shape
