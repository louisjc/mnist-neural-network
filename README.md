# Python Neural Network - Handwritten digits classification

This project is a simple Python script which implements and trains a 2 layer neural network classifying handwritten digits using the [MNIST database](http://yann.lecun.com/exdb/mnist/) for both training and testing.

## Usage

This script requires **Python 3**. By default, the script trains a NN with 300 hiddens units until convergence.

```bash
pip install scipy numpy
python neural_net.py
```

## License

The code is liscensed under the MIT lisence. See `LICENSE.md`.

Yann LeCun and Corinna Cortes hold the copyright of [MNIST database](http://yann.lecun.com/exdb/mnist/), which is a derivative work from original NIST datasets. [MNIST database](http://yann.lecun.com/exdb/mnist/) is made available under the terms of the [Creative Commons Attribution-Share Alike 3.0](http://creativecommons.org/licenses/by-sa/3.0/) license.

The files `data_testing` and `data_training` contain the MNIST database saved using Python pickle. They are licensed under [Creative Commons Attribution-Share Alike 3.0](http://creativecommons.org/licenses/by-sa/3.0/).
