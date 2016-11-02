library(tensorflow)

# Read dataset
datasets <- tf$contrib$learn$datasets
mnist <- datasets$mnist$read_data_sets("MNIST-data", one_hot = TRUE)

# Parameters
nInput <- 28L
nSteps <- 28L
nHidden <- 128L
nClasses <- 10L
batchSize <- 128L

# X and Y placeholders
x <- tf$placeholder(tf$float32, shape(NULL, nSteps, nInput))
y <- tf$placeholder(tf$float32, shape(NULL, nClasses))

# Reshaping
xX <- tf$transpose(x, shape(1L, 0L, 2L))
xX <- tf$reshape(xX, shape(-1L, nInput))
xX <- tf$split(0L, nSteps, xX)

# Weigths and bias
weights <- tf$Variable(tf$random_normal(shape(nHidden, nClasses)))
bias <- tf$Variable(tf$random_normal(shape(nClasses)))

# Prepare Cells and RNN based on these cells
lstmCell <- tf$nn$rnn_cell$BasicLSTMCell(nHidden, forget_bias=1.0, state_is_tuple=FALSE)
# Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
result <- tf$nn$rnn(lstmCell, xX, dtype=tf$float32)

# Matmul
lastCell <- length(result[1][[1]])
# Prediction is done by last layer in the RNN
pred <- tf$matmul(result[1][[1]][[lastCell-1]], weights) + bias
cost <- tf$reduce_mean(tf$nn$softmax_cross_entropy_with_logits(pred, y))
optimizer <- tf$train$AdamOptimizer(learning_rate=0.001)$minimize(cost)

# Evaluate model
correct_pred <- tf$equal(tf$argmax(pred,1L), tf$argmax(y,1L))
accuracy <- tf$reduce_mean(tf$cast(correct_pred, tf$float32))

# Init session
sess <- tf$Session()
# Initializing the variables
sess$run(tf$initialize_all_variables())

# Loop training
for (step in 1:500) {
  batches <- mnist$train$next_batch(128L)
  batch_xs <- sess$run(tf$reshape(batches[[1]], shape(batchSize, nSteps, nInput))) # Reshape
  batch_ys <- batches[[2]]
  # Run optimization op (backprop)
  sess$run(optimizer, feed_dict = dict(x = batch_xs, y= batch_ys))
  if (step %% 10 == 0) {
    # Get accuracy 
    acc <- sess$run(accuracy, feed_dict = dict(x = batch_xs, y= batch_ys))
    # Calculate batch loss
    loss <- sess$run(cost, feed_dict = dict(x = batch_xs, y= batch_ys))
    print(paste("Acc:",round(acc,3)," loss:",round(loss,3)," step(",step,")"))
  }
}


# Make a prediction
idToTest <- 62
img <- mnist$test$images[idToTest,]
# save as matrix for plot
imgMatrix <- matrix(img, ncol = 28, nrow = 28)
# reshape for prediction
img <- sess$run(tf$reshape(img, shape(1, nSteps, nInput))) 

# Prediction
sess$run(tf$argmax(pred, 1L), feed_dict = dict(x = img))
# Plot image
image(t(imgMatrix), col  = gray((0:32)/32))


