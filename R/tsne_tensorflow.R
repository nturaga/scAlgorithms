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
    tsne_out <- tsne_tensorflow(vals,
                                initial_dims,
                                init = 'pca',
                                perplexity = perplexity,
                                dims = ncomponents)
    
    reducedDim(object, "TSNE") <- tsne_out
    return(object)
}


## Compute the perplexity and the P-row for a specific value of the
##  precision of a Gaussian distribution.
Hbeta <-
    function(D, beta = tf$constant(1.0, dtype = "float64"))
{
    ## Compute P-row and corresponding perplexity
    P <- tf$exp(-D * beta)
    sumP <- tf$reduce_sum(P)
    sum_D_P <- tf$reduce_sum(tf$multiply(D, P))
    H <- tf$add(
        tf$log(sumP),
        tf$multiply(beta, (sum_D_P/ sumP))
    )
    P <- tf$truediv(P, sumP)             
    c(H, P)
}

## Test a random HBeta
beta <- tf$constant(1.0, dtype="float64")
Hbeta(aa)


    
## Performs a binary search to get P-values in such a way that each
##  conditional Gaussian has the same perplexity.    
x2p <-
    function(X,
             tol = tf$constant(1e-5, dtype="float64"),
             perplexity = tf$constant(30.0, dtype="float64"))
{
    message("Computing pairwise distances ...")
    n <- dim(X)[1]
    d <- dim(X)[2]

    sum_X <- tf$reduce_sum(tf$square(X), 1)
    X_T <- tf$matrix_transpose(X)
    D <- tf$add(tf$matrix_transpose(tf$add(-2 * tf$tensordot(X, X_T), sum_X)), sum_X)
    P <- matrix(0, n, n)
    beta <- matrix(1, n, 1)
    logU <- tf$log(perplexity)

    ## loop over data points
    for (i in seq_len(n)) {

        ## Print progress
        if (i %% 500 == 0) {
            message("Computing P-values for point", i, "of", n)
        }

        ## Compute the Gaussian kernel and entropy for the current precision
        betamin <- -Inf
        betamax <- Inf
        ## TODO: take note of ranges here
        Di <- D[i, c(seq_along(0,i), c(i+1, n))]

        res <- Hbeta(Di, beta[i])
        H <- res[1]
        thisP <- res[2]

        ## Evaluate whether the perplexity is within tolerance
        Hdiff <- tf$subtract(H, logU)
        tries <- 0

        while (tf.abs(Hdiff) > tol && tries < 50) {

            ## If not, increase or decrease precision
            if (Hdiff > 0) {
                betamin <- beta[i]
                if (betamax == Inf || betamax == -Inf)
                    beta[i] <- beta[i] * 2.0
                else
                    beta[i] = (beta[i] + betamax) / 2.0    
            } else {
                betamax <- beta[i]
                if (betamin == Inf || betamin == -Inf)
                    beta[i] <- beta[i] / 2.0
                else
                    beta[i] <- (beta[i] + betamin) / 2.0
            }

            ## recomupte the values
            res <- Hbeta(Di, beta[i])
            H <- res[1]
            thisP <- res[2]
            Hdiff <- tf.subtract(H,logU)
            tries = tries + 1   
        }
        
        P[i, c(seq_along(0,i), c(i+1, n))] <- thisP
    }
    ## Return final P matrix
    P
}


##  Runs PCA on the NxD array X in order to reduce its dimensionality to
##  no_dims dimensions.
pca <-
    function(X, no_dims = 50L)
{
    n <- dim(X)[1]
    d <- dim(X)[2]

    X <- tf$subtract(X,
                     tf$tile(tf$reduce_mean(X,0L),
                             multiples=tuple(n,1L))
                     )
    X_T <- tf$matrix_transpose(X)
    eig <- tf$self_adjoint_eig(tf$tensordot(X_T, X))
    l <- eig[1] ## Eigenvalues
    M <- eig[2] ## Eigenvectors
    Y <- tf$tensordot(X, M[, seq_along(1,no_dims)])
    Y
}


#' Rtsne implemenataion in tensorflow
#' @param X matrix; Data matrix
#' @param initial_dims: integer; the number of dimensions that should be retained
#'          in the initial PCA step (default: 50)

#' @param perplexity numeric; Perplexity parameter
#' @param pca logical; Whether an initial PCA step should be performed
#'      (default: TRUE)
#' @param dims integer; Output dimensionality (default: 2)
#' @param check_duplicates logical; Checks whether duplicates are present. It is
#'          best to make sure there are no duplicates present and set
#'          this option to FALSE, especially for large datasets (default:
#'          TRUE)
tsne_tensorflow <-
    function(X, no_dims = 2,
             initial_dims = 50,
             perplexity = perplexity,
             pca = pca,
             check_duplicates = FALSE)
{
    ## Check if tensorflow module is available
    if !is(tf, "python.builtin.module") {
        stop("tensorflow module 'tf' not available")
    }

    ## Check inputs
    stopifnot(
        is(no_dims, "numeric") 
    )

    ## Initialize variables
    X <- pca(X, initial_dims) ## TODO: convert to real number??
    dim <- tf$shape(X)
    n <- dim[1]
    d <- dim[2]

    max_iter <- 1000
    initial_momentum <- 0.5
    final_momentum <- 0.8
    eta <- 500
    min_gain <- 0.01
    Y <- matrix(rnorm(n), n ,no_dims) ## TODO: is this correct?

    dY <- matrix(0, n, no_dims)
    iY <- matrix(0, n, no_dims)
    gains <- matrix(1, n, no_dims)

    # Compute P-values
    P <- x2p(X, 1e-5, perplexity)
    P <- tf$add(P, tf$matrix_transpose(P))
    P <- tf$truediv(P, tf$reduce_sum(P))
    P <- tf$multiply(P, 4.0) ## early exaggeration
    P <- tf$maximum(P, 1e-12)

    ## Run iterations
    for (iter in seq_len(max_iter)) {

        ## Compute pairwise affinities
        sum_Y <- tf$add(tf$square(Y), 1)
        num <- -2.0 * tf$tensordot(Y, tf$matrix_transpose(Y))
        num <- 1.0 / (1.0 + tf$add(tf$matrix_transpose(tf$add(num, sum_Y)), sum_Y))
        ## FIXME: This is suspect
        num[seq_len(n), seq_len(n)] <- 0.0
        Q <- tf$truediv(num, tf$sum(num))
        Q <- tf$maximum(Q, 1e-12)

        ## Compute gradient
        PQ <- P - Q
        
        ## FIXME: This is suspect
        for (i in seq_len(n)) {
            dY[i, ] <- tf$sum(tf$matrix_transpose(tf$tile(PQ[, i] * num[, i], (no_dims, 1))) * (Y[i, ] - Y), 0)
        }
        
        ## Perform the update
        if (iter < 20) {
            momentum <- initial_momentum
        } else {
            momentum <- final_momentum
        }
        gains <- (gains + 0.2) * ((dY > 0.0) != (iY > 0.0)) + (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] <- min_gain
        iY <- momentum * iY - eta * (gains * dY)
        Y <- Y + iY
        Y <- Y - tf$tile(tf$reduce_mean(Y, 0), tuple(n, 1L))

        # Compute current value of cost function
        if ((iter + 1) %% 10 == 0) {
            C <- tf$reduce_sum(P * tf$log(P / Q))
            sprintf("Iteration %d: error is %f", iter + 1, C)
        }
        # Stop lying about P-values
        if (iter == 100) {
            P <- P / 4.0
        }
        
    # Return solution
    return Y
}
