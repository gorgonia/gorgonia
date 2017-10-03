# CharRNN [![experimental](http://badges.github.io/stability-badges/dist/experimental.svg)](http://github.com/badges/stability-badges) #

This is a simple example fo how to build a charRNN with Gorgonia. 

This example currently works but is very slow, due to the new way the tensors are structured. Fixes to this example will be made in the next version (v0.8.0)

# Disclaimer #

This example comes with a basic implementation of CharRNN. It does not contain:

* Model serialization (you cannot save your models)
* Checkpointing.
* Multithreaded batching (training the CharRNN is highly serial).
