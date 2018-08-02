library(tensorflow)

matrix1 <- tf$constant(matrix(c(3.0, 3.0), nrow = 1, ncol = 2))

matrix2 <- tf$constant(matrix(c(3.0, 3.0), nrow = 2, ncol = 1))


product <- tf$matmul(matrix1, matrix2)

## dimensions
product$shape

## name
product$name

## tensorflow object type
product$graph

## Lazy eval the data
sess <- tf$Session()
result <- sess$run(product)
print(result)
sess$close()

## Different way of closing session
with(tf$Session() %as% sess, {
  result = sess$run(product)
  print(result)
})


