## A single Layer Network on the Hitters Data

library(ISLR2)
Gitters <- na.omit(Hitters)
n <- nrow(Gitters)
set.seed(13)
ntest <- trunc(n / 3)
testid <- sample(1:n, ntest)

# Fit a linear regression model
lfit <- lm(Salary ~ ., data = Gitters[-testid, ])
lpred <- predict(lfit, Gitters[testid, ])

# For with command, first argument is the dataframe, second 
# is an expression that refer to elements of the dataframe by name
with(Gitters[testid, ], mean(abs(lpred - Salary)))

# Extract x and y for glmnet function.
# model.matrix produces the same matrix that was used by lm, -1 omits the 
# intercept. This function then automatically converts factors to dummy
# variables.

# scale standardize the data to have mean zero and variance one.
x <- scale(model.matrix(Salary ~ . - 1, data = Gitters))
y <- Gitters$Salary


library(glmnet)
# Note the metric is mae, which stands for mean absolute error
cvfit <- cv.glmnet(x[-testid, ], y[-testid], type.measure = "mae")
cpred <- predict(cvfit, x[testid, ], s = "lambda.min")
mean(abs(y[testid] - cpred))


library(keras)
# So far we have created a vanilla model with keras_model_sequential function.
# The %>% pipe operator passes the previous term as the first argument
# to the next function and return its result. 
# This pipe operator allows us to specify the layers of a neural network
# in a readable form. 
modnn <- keras_model_sequential() %>%
    layer_dense(units = 50, activation = "relu", 
              input_shape = ncol(x)) %>%
    layer_dropout(rate = 0.4) %>% 
    layer_dense(units = 1)

# This neural network has a single hidden layer with 50 hidden units, and 
# a ReLU activation function. 
# It then addes a dropout layer, in which a random 40% of the 50 units
# from the previous layers are set to zero. 
# 

x <- model.matrix(Salary ~ . - 1, data = Gitters) %>% scale()

# We minimize the squared-error loss and the algorithm
# tracks the mean absolute error on the training data
modnn %>% compile(loss = "mse", 
                  optimizer = optimizer_rmsprop(),
                  metrics = list("mean_absolute_error")
)

# compile does not really change the R object modnn, instead
# it communicate these specifications to the corresponding 
# python instant of this model that has been created along this way.

# Two important parameters: epochs and batch_size
# batch_size indicates the sample size use for each step of 
# stochastic gradient descent.
# one epoch means going through the n data point once
# Thus, for this case, we have 176 data points, and thus, an epoch
# would consist of 176 / 32 = 5.5 SGD steps.

# validation_data: track the progress of the model fitting, not used 
# for fitting.
history <- modnn %>% fit(
  x[-testid, ], y[-testid], epochs = 1500, batch_size=32,
  validation_data = list(x[testid, ], y[testid])
)

library(ggplot2)  # a library that makes beautiful plots.
plot(history)

# The results might vary due to the usage of SGD.
npred <- predict(modnn, x[testid, ])
mean(abs(y[testid] - npred))

# A multilayer Network on the MNIST Digit Data
mnist <- dataset_mnist() # load the MNIST data
x_train <- mnist$train$x
g_train <- mnist$train$y
x_test <- mnist$test$x
g_test <- mnist$test$y

dim(x_train)

dim(x_test)

x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))

# One-hot encoding the class labels
y_train <- to_categorical(g_train, 10)
y_test <- to_categorical(g_test, 10)

# Scale the data
x_train <- x_train / 255
x_test <- x_test / 255


modelnn <- keras_model_sequential()
modelnn %>% 
  layer_dense(units = 256, activation = "relu", 
               input_shape = c(784)) %>%
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = "relu") %>%
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 10, activation = "softmax")

# The first hidden layer goes from 784 input units to a hidden layer of 256
# units.
# Then we add a dropout layer
# Then we add another hidden layer and another dropout layer
# For the final layer, we specify the activation function to be softmax
# for the 10-class classification problem. 

summary(modelnn)

modelnn %>% compile(loss = "categorical_crossentropy",
                    optimizer = optimizer_rmsprop(), metrics = c("accuracy"))

system.time(
  history <- modelnn %>% 
    fit(x_train, y_train, epochs = 30, batch_size=128, validation_split = 0.2)
)

# Here we specify that 20% of the data being used as validation. 

plot(history, smooth = FALSE)

accuracy <- function(pred, truth)
  mean(drop(as.numeric(pred)) == drop(truth))

modelnn %>% predict(x_test) %>% k_argmax() %>% accuracy(g_test)

# We can also fit a multiclass logistic regression with keras!
# And it even runs faster than glmnet.
modellr <- keras_model_sequential() %>% 
  layer_dense(input_shape = 784, units = 10, activation = "softmax")

summary(modellr)

modellr %>% compile(loss = "categorical_crossentropy",
                    optimizer = optimizer_rmsprop(), metrics = c("accuracy"))
modellr %>% fit(x_train, y_train, epochs = 30,
                batch_size = 128, validation_split = 0.2)
modellr %>% predict(x_test) %>% k_argmax() %>% accuracy(g_test)
