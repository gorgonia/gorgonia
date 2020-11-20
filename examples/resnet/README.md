# ConvNet [![experimental](http://badges.github.io/stability-badges/dist/experimental.svg)](http://github.com/badges/stability-badges) #

This is an example of a 3-layer ConvNet built with Gorgonia. It uses the MNIST data as an example. The MNIST data is not provided, and MUST be put in `../testdata`.

This example currently works but is very slow, pending a change in transpose algorithms.

A step by step tutorial is exposed on to the Gorgonia website: [https://gorgonia.org/tutorials/mnist/](https://gorgonia.org/tutorials/mnist/)

# Disclaimer #

This example comes with a basic implementation of a ConvNet. It does not contain:

* Model serialization (you cannot save your models)
* Checkpointing.
* Multithreaded batching (training is highly serial).

