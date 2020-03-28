![Logo](https://raw.githubusercontent.com/gorgonia/gorgonia/master/media/Logo_horizontal_small.png)

[![GoDoc](https://godoc.org/gorgonia.org/gorgonia?status.svg)](https://godoc.org/gorgonia.org/gorgonia) [![GitHub version](https://badge.fury.io/gh/gorgonia%2Fgorgonia.svg)](https://badge.fury.io/gh/gorgonia%2Fgorgonia) [![Build Status](https://travis-ci.org/gorgonia/gorgonia.svg?branch=master)](https://travis-ci.org/gorgonia/gorgonia) [![Coverage Status](https://coveralls.io/repos/github/gorgonia/gorgonia/badge.svg?branch=master)](https://coveralls.io/github/gorgonia/gorgonia?branch=master) [![Go Report Card](https://goreportcard.com/badge/gorgonia.org/gorgonia)](https://goreportcard.com/report/gorgonia.org/gorgonia) [![unstable](http://badges.github.io/stability-badges/dist/unstable.svg)](http://github.com/badges/stability-badges)

#

Gorgonia is a library that helps facilitate machine learning in Go. Write and evaluate mathematical equations involving multidimensional arrays easily. If this sounds like [Theano](http://deeplearning.net/software/theano/) or [TensorFlow](https://www.tensorflow.org/), it's because the idea is quite similar. Specifically, the library is pretty low-level, like Theano, but has higher goals like Tensorflow.

Gorgonia:

* Can perform automatic differentiation
* Can perform symbolic differentiation
* Can perform gradient descent optimizations
* Can perform numerical stabilization
* Provides a number of convenience functions to help create neural networks
* Is fairly quick (comparable to Theano and Tensorflow's speed)
* Supports CUDA/GPGPU computation (OpenCL not yet supported, send a pull request)
* Will support distributed computing

# Goals #

The primary goal for Gorgonia is to be a *highly performant* machine learning/graph computation-based library that can scale across multiple machines. It should bring the appeal of Go (simple compilation and deployment process) to the ML world. It's a long way from there currently, however, the baby steps are already there.

The secondary goal for Gorgonia is to provide a platform for exploration for non-standard deep-learning and neural network related things. This includes things like neo-hebbian learning, corner-cutting algorithms, evolutionary algorithms and the like.

# Why Use Gorgonia? #

The main reason to use Gorgonia is developer comfort. If you're using a Go stack extensively, now you have access to the ability to create production-ready machine learning systems in an environment that you are already familiar and comfortable with.

ML/AI at large is usually split into two stages: the experimental stage where one builds various models, test and retest; and the deployed state where a model after being tested and played with, is deployed. This necessitate different roles like data scientist and data engineer.

Typically the two phases have different tools: Python ([PyTorch](http://pytorch.org/), etc) is commonly used for the experimental stage, and then the model is rewritten in some more performant language like C++ (using [dlib](http://dlib.net/ml.html), [mlpack](http://mlpack.org) etc). Of course, nowadays the gap is closing and people frequently share the tools between them. Tensorflow is one such tool that bridges the gap.

Gorgonia aims to do the same, but for the Go environment. Gorgonia is currently fairly performant - its speeds are comparable to PyTorch's and Tensorflow's  CPU implementations. GPU implementations are a bit finnicky to compare due to the heavy cgo tax, but rest assured that this is an area of active improvement.

# Getting started

## Installation #

The package is go-gettable: `go get -u gorgonia.org/gorgonia`.

Gorgonia is compatible with go modules.

## Documentation

Up-to-date documentation, references and tutorials are present on the official Gorgonia website at [https://gorgonia.org](https://gorgonia.org).

## Keeping Updated 

Gorgonia's project has a [Slack channel on gopherslack](https://gophers.slack.com/messages/gorgonia/), as well as a [Twitter account](https://twitter.com/gorgoniaML). Official updates and announcements will be posted to those two sites.

## Usage 

Gorgonia works by creating a computation graph, and then executing it. Think of it as a programming language, but is limited to mathematical functions, and has no branching capability (no if/then or loops). In fact this is the dominant paradigm that the user should be used to thinking about. The computation graph is an [AST](http://en.wikipedia.org/wiki/Abstract_syntax_tree).

Microsoft's [CNTK](https://github.com/Microsoft/CNTK), with its BrainScript, is perhaps the best at exemplifying the idea that building of a computation graph and running of the computation graphs are different things, and that the user should be in different modes of thoughts when going about them.

Whilst Gorgonia's implementation doesn't enforce the separation of thought as far as CNTK's BrainScript does, the syntax does help a little bit.

Here's an example - say you want to define a math expression `z = x + y`. Here's how you'd do it:

[embedmd]:# (example_basic_test.go)
```go
package gorgonia_test

import (
	"fmt"
	"log"

	. "gorgonia.org/gorgonia"
)

// Basic example of representing mathematical equations as graphs.
//
// In this example, we want to represent the following equation
//		z = x + y
func Example_basic() {
	g := NewGraph()

	var x, y, z *Node
	var err error

	// define the expression
	x = NewScalar(g, Float64, WithName("x"))
	y = NewScalar(g, Float64, WithName("y"))
	if z, err = Add(x, y); err != nil {
		log.Fatal(err)
	}

	// create a VM to run the program on
	machine := NewTapeMachine(g)
	defer machine.Close()

	// set initial values then run
	Let(x, 2.0)
	Let(y, 2.5)
	if err = machine.RunAll(); err != nil {
		log.Fatal(err)
	}

	fmt.Printf("%v", z.Value())
	// Output: 4.5
}
```

You might note that it's a little more verbose than other packages of similar nature. For example, instead of compiling to a callable function, Gorgonia specifically compiles into a `*program` which requires a `*TapeMachine` to run. It also requires manual a `Let(...)` call.

The author would like to contend that this is a Good Thing - to shift one's thinking to a machine-based thinking. It helps a lot in figuring out where things might go wrong.

Additionally, there are no support for branching - that is to say there are no conditionals (if/else) or loops. The aim is not to build a Turing-complete computer.

---
More examples are present in the `example` subfolder of the project, and step-by-step tutorials are present on the [main website](https://gorgonia.org/tutorials/)

## Using CUDA ##

Gorgonia comes with CUDA support out of the box.
Please see the reference documentation about how cuda works on [the Gorgonia.org](https://gorgonia.org/reference/cuda/) website, or jump to the [tutorial](https://gorgonia.org/tutorials/mnist-cuda/).

# About Gorgonia's development process

## Versioning ##

We use [semver 2.0.0](http://semver.org/) for our versioning. Before 1.0, Gorgonia's APIs are expected to change quite a bit. API is defined by the exported functions, variables and methods. For the developers' sanity, there are minor differences to semver that we will apply prior to version 1.0. They are enumerated below:

* The MINOR number will be incremented every time there is a deleterious break in API. This means any deletion, or any change in function signature or interface methods will lead to a change in MINOR number. 
* Additive changes will NOT change the MINOR version number prior to version 1.0. This means that if new functionality were added that does not break the way you use Gorgonia, there will not be an increment in the MINOR version. There will be an increment in the PATCH version.

### API Stability #
Gorgonia's API is as of right now, not considered stable. It will be stable from version 1.0 forwards.


## Go Version Support ##

Gorgonia supports 2 versions below the Master branch of Go. This means Gorgonia will support the current released version of Go, and up to 4 previous versions - providing something doesn't break. Where possible a shim will be provided (for things like new `sort` APIs or `math/bits` which came out in Go 1.9).

The current version of Go is 1.13.1. The earliest version Gorgonia supports is Go 1.11.x but Gonum supports only 1.12+. Therefore, the minimum Go version to run the master branch is Go > 1.12.

## Hardware and OS supported ##

Gorgonia runs on :
- linux/AMD64
- linux/ARM7
- linux/ARM64
- win32/AMD64
- darwin/AMD64
- freeBSD/AMD64

If you have tested gorgonia on other platform, please update this list.

## Hardware acceleration

Gorgonia use some pure assembler instructions to accelerate somes mathematical operations. Unfortunately, only amd64 is supported.


# Contributing #

Obviously since you are most probably reading this on Github, Github will form the major part of the workflow for contributing to this package.

See also: [CONTRIBUTING.md](CONTRIBUTING.md)


## Contributors and Significant Contributors ##
All contributions are welcome. However, there is a new class of contributor, called Significant Contributors.

A Significant Contributor is one who has shown *deep understanding* of how the library works and/or its environs.  Here are examples of what constitutes a Significant Contribution:

* Wrote significant amounts of documentation pertaining to **why**/the mechanics of particular functions/methods and how the different parts affect one another
* Wrote code, and tests around the more intricately connected parts of Gorgonia
* Wrote code and tests, and have at least 5 pull requests accepted
* Provided expert analysis on parts of the package (for example, you may be a floating point operations expert who optimized one function)
* Answered at least 10 support questions.

Significant Contributors list will be updated once a month (if anyone even uses Gorgonia that is).

# How To Get Support #
The best way of support right now is to open a [ticket on Github](https://github.com/gorgonia/gorgonia/issues/new).

# Frequently Asked Questions #

### Why are there seemingly random `runtime.GC()` calls in the tests? ###

The answer to this is simple - the design of the package uses CUDA in a particular way: specifically, a CUDA device and context is tied to a `VM`, instead of at the package level. This means for every `VM` created, a different CUDA context is created per device per `VM`. This way all the operations will play nicely with other applications that may be using CUDA (this needs to be stress-tested, however).

The CUDA contexts are only destroyed when the `VM` gets garbage collected (with the help of a finalizer function). In the tests, about 100 `VM`s get created, and garbage collection for the most part can be considered random. This leads to cases where the GPU runs out of memory as there are too many contexts being used.

Therefore at the end of any tests that may use GPU, a `runtime.GC()` call is made to force garbage collection, freeing GPU memories.

In production, one is unlikely to start that many `VM`s, therefore it's not really a problem. If there is, open a ticket on Github, and we'll look into adding a `Finish()` method for the `VM`s.


# Licence #

Gorgonia is licenced under a variant of Apache 2.0. It's for all intents and purposes the same as the Apache 2.0 Licence, with the exception of not being able to commercially profit directly from the package unless you're a Significant Contributor (for example, providing commercial support for the package). It's perfectly fine to profit directly from a derivative of Gorgonia (for example, if you use Gorgonia as a library in your product)

Everyone is still allowed to use Gorgonia for commercial purposes (example: using it in a software for your business).

## Dependencies ##

There are very few dependencies that Gorgonia uses - and they're all pretty stable, so as of now there isn't a need for vendoring tools. These are the list of external packages that Gorgonia calls, ranked in order of reliance that this package has (subpackages are omitted):

|Package|Used For|Vitality|Notes|Licence|
|-------|--------|--------|-----|-------|
|[gonum/graph](https://github.com/gonum/gonum/tree/master/graph)| Sorting `*ExprGraph`| Vital. Removal means Gorgonia will not work | Development of Gorgonia is committed to keeping up with the most updated version|[gonum license](https://github.com/gonum/license) (MIT/BSD-like)|
|[gonum/blas](https://github.com/gonum/gonum/tree/master/blas)|Tensor subpackage linear algebra operations|Vital. Removal means Gorgonial will not work|Development of Gorgonia is committed to keeping up with the most updated version|[gonum license](https://github.com/gonum/license) (MIT/BSD-like)|
|[cu](https://gorgonia.org/cu)| CUDA drivers | Needed for CUDA operations | Same maintainer as Gorgonia | MIT/BSD-like|
|[math32](https://github.com/chewxy/math32)|`float32` operations|Can be replaced by `float32(math.XXX(float64(x)))`|Same maintainer as Gorgonia, same API as the built in `math` package|MIT/BSD-like|
|[hm](https://github.com/chewxy/hm)|Type system for Gorgonia|Gorgonia's graphs are pretty tightly coupled with the type system | Same maintainer as Gorgonia | MIT/BSD-like|
|[vecf64](https://gorgonia.org/vecf64)| optimized `[]float64` operations | Can be generated in the `tensor/genlib` package. However, plenty of optimizations have been made/will be made | Same maintainer as Gorgonia | MIT/BSD-like|
|[vecf32](https://gorgonia.org/vecf32)| optimized `[]float32` operations | Can be generated in the `tensor/genlib` package. However, plenty of optimizations have been made/will be made | Same maintainer as Gorgonia | MIT/BSD-like|
|[set](https://github.com/xtgo/set)|Various set operations|Can be easily replaced|Stable API for the past 1 year|[set licence](https://github.com/xtgo/set/blob/master/LICENSE) (MIT/BSD-like)|
|[gographviz](https://github.com/awalterschulze/gographviz)|Used for printing graphs|Graph printing is only vital to debugging. Gorgonia can survive without, but with a major (but arguably nonvital) feature loss|Last update 12th April 2017|[gographviz licence](https://github.com/awalterschulze/gographviz/blob/master/LICENSE) (Apache 2.0)|
|[rng](https://github.com/leesper/go_rng)|Used to implement helper functions to generate initial weights|Can be replaced fairly easily. Gorgonia can do without the convenience functions too||[rng licence](https://github.com/leesper/go_rng/blob/master/LICENSE) (Apache 2.0)|
|[errors](https://github.com/pkg/errors)|Error wrapping|Gorgonia won't die without it. In fact Gorgonia has also used [goerrors/errors](https://github.com/go-errors/errors) in the past.|Stable API for the past 6 months|[errors licence](https://github.com/pkg/errors/blob/master/LICENSE) (MIT/BSD-like)|
|[gonum/mat](http://github.com/gonum/gonum)|Compatibility between `Tensor` and Gonum's Matrix|Development of Gorgonia is committed to keeping up with the most updated version||[gonum license](https://github.com/gonum/license) (MIT/BSD-like)|
|[testify/assert](https://github.com/stretchr/testify)|Testing|Can do without but will be a massive pain in the ass to test||[testify licence](https://github.com/stretchr/testify/blob/master/LICENSE) (MIT/BSD-like)|


## Various Other Copyright Notices ##

These are the packages and libraries which inspired and were adapted from in the process of writing Gorgonia (the Go packages that were used were already declared above):

| Source | How it's Used | Licence |
|------|---|-------|
| Numpy  | Inspired large portions. Directly adapted algorithms for a few methods (explicitly labelled in the docs) | MIT/BSD-like. [Numpy Licence](https://github.com/numpy/numpy/blob/master/LICENSE.txt) |
| Theano | Inspired large portions. (Unsure: number of directly adapted algorithms) | MIT/BSD-like [Theano's licence](http://deeplearning.net/software/theano/LICENSE.html) |
| Caffe | `im2col` and `col2im` directly taken from Caffe. Convolution algorithms inspired by the original Caffee methods | [Caffe Licence](https://github.com/BVLC/caffe/blob/master/LICENSE)
