library(tensorflow)

nmf <-
    function(V, rank, algo='mu', learning_rate = 0.01)
{
    V <- tf$constant(V,dtype=tf$float32)

    scale = 2 * np$sqrt(V.mean() / rank)
    initializer = tf$random_uniform_initializer(maxval=scale)
    
    H <- tf$get_variable("H", [rank, shape[1]],
                              initializer=initializer)
    W <- tf$get_variable(name="W", shape=[shape[0], rank],
                              initializer=initializer)
    
    if (algo == "mu") {
        build_mu_algorithm()
    }
    else if (algo == "grad") {
        build_grad_algorithm()
    }
    else {
        stop("The attribute algo must be in {'mu', 'grad'}")
    }
}


