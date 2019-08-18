# Simple Neural Network
This is a simple Neural Network with one hidden layer and Backpropagation algorithm for image classification. As data source, the [MNIST](http://yann.lecun.com/exdb/mnist) database is used. This database contains 60.000 labeled examples of handwritten digits for training and 10.000 non-labeled examples for testing the Neural Network.

**Goal:**
The goal of the Neural Network is to detect digits within non-labeled image data by learning from labeled examples.
![Goal](https://github.com/marceldorner/simple-neural-network/blob/master/goal.png)

The source code is based on an example of the book [
Neuronale Netze selbst programmieren](https://www.oreilly.de/landing/12892.php). To execute the Python script, just type `python neural_network.py` in a console.

**Example output:**
```
Training Neural Network with 60000 records ...
Testing Neural Network with 10000 records ...

Overall performance = 0.9523

Press ENTER to exit...
```
--> 95,23 % of all 10.000 non-labeled examples for testing where classified correctly by the Neural Network!


During development you may want to use `mnist_train_100.csv` and `mnist_test_10.csv` instead of `mnist_train.csv` and `mnist_test.csv`. This is a subset with only 100 labeled examples of handwritten digits for training and 10 non-labeled examples for testing the Neural Network. Using the subset will reduce the execution time but will also result in poor performance due to insufficient training data.
