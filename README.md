#Gorgonia [![GoDoc](https://godoc.org/github.com/chewxy/gorgonia?status.svg)](https://godoc.org/github.com/chewxy/gorgonia) [![Build Status](https://travis-ci.org/chewxy/gorgonia.svg?branch=master)](https://travis-ci.org/chewxy/gorgonia)#

Gorgonia is a library that helps facilitate machine learning in Go. Write and evaluate mathematical equations involving multidimensional arrays easily. If this sounds like [Theano](http://deeplearning.net/software/theano/) or [TensorFlow](https://www.tensorflow.org/), it's because the idea is quite similar. Specifically, the library is pretty low-level, like Theano, but has higher goals like Tensorflow.

Gorgonia:

* Can perform automatic differentiation
* Can perform symbolic differentiation
* Can perform gradient descent optimizations
* Can perform numerical stabilization
* Provides a number of convenience functions to help create neural networks
* Is fairly quick (comparable to Theano and Tensorflow's speed)
* Will support GPU/CUDA
* Will support distributed computing

#Why Use Gorgonia?#

The main reason to use Gorgonia is developer comfort. If you're using a Go stack extensively, now you have access to the ability to create production-ready machine learning systems in an environment that you are already familiar and comfortable with. 

ML/AI at large is usually split into two stages: the experimental stage where one builds various models, test and retest; and the deployed state where a model after being tested and played with, is deployed. This necessitate different roles like data scientist and data engineer.

Typically the two phases have different tools: Python/Lua (using Theano, [Torch](http://torch.ch/), etc) is commonly used for the experimental stage, and then the model is rewritten in some more performant language like C++ (using [dlib](http://dlib.net/ml.html), [mlpack](http://mlpack.org) etc). Of course, nowadays the gap is closing and people frequently share the tools between them. Tensorflow is one such tool that bridges the gap.

Gorgonia aims to do the same, but for the Go environment. Gorgonia is currently fairly performant - its speeds are comparable to Theano's and Tensorflow's (official benchmarks haven't yet been done because of an existing CUDA bug in Gorgonia; and also the implementations may differ slightly so an exact like-for-like model is hard to compare).

#Installation #

The package is go-gettable: `go get -u github.com/chewxy/gorgonia`. 

There are very few dependencies that Gorgonia uses - and they're all pretty stable, so as of now, there isn't a need for vendoring tools. These are the list of external packages that Gorgonia calls, ranked in order of reliance that this package has (subpackages are omitted):

|Package|Used For|Vitality|Notes|Licence|
|-------|--------|--------|-----|-------|
|[gonum/graph](http://github.com/gonum/graph)| Sorting `*ExprGraph`| Vital. Removal means Gorgonia will not work | Development of Gorgonia is committed to keeping up with the most updated version|[gonum license](https://github.com/gonum/license) (MIT/BSD-like)|
|[gonum/blas](http://github.com/gonum/blas)|Tensor subpackage linear algebra operations|Vital. Removal means Gorgonial will not work|Development of Gorgonia is committed to keeping up with the most updated version|[gonum license](https://github.com/gonum/license) (MIT/BSD-like)|
|[math32](http://github.com/chewxy/math32)|`float32` operations|Can be replaced by `float32(math.XXX(float64(x)))`|Same maintainer as Gorgonia, same API as the built in `math` package|MIT/BSD-like|
|[set](https://github.com/xtgo/set)|Various set operations|Can be easily replaced|Stable API for the past 1 year|[set licence](https://github.com/xtgo/set/blob/master/LICENSE) (MIT/BSD-like)|
|[gographviz](https://github.com/awalterschulze/gographviz)|Used for printing graphs|Graph printing is only vital to debugging. Gorgonia can survive without, but with a major (but arguably nonvital) feature loss|Stable API for the past 1 year|[gographviz licence](https://github.com/awalterschulze/gographviz/blob/master/LICENSE) (Apache 2.0)|
|[rng](https://github.com/leesper/go_rng)|Used to implement helper functions to generate initial weights|Can be replaced fairly easily. Gorgonia can do without the convenience functions too||[rng licence](https://github.com/leesper/go_rng/blob/master/LICENSE) (Apache 2.0)|
|[errors](https://github.com/pkg/errors)|Error wrapping|>Gorgonia won't die without it. In fact Gorgonia has also used [goerrors/errors](https://github.com/go-errors/errors) in the past.|Stable API for the past 6 months|[errors licence](https://github.com/pkg/errors/blob/master/LICENSE) (MIT/BSD-like)|
|[gonum/matrix](http://github.com/gonum/matrix)|Compatibility between `Tensor` and Gonum's Matrix|Development of Gorgonia is committed to keeping up with the most updated version|[gonum license](https://github.com/gonum/license) (MIT/BSD-like)|
|[testify/assert](https://github.com/stretchr/testify)|Testing|Can do without but will be a massive pain in the ass to test||[testify licence](https://github.com/stretchr/testify/blob/master/LICENSE) (MIT/BSD-like)|
|[diffmatchpatch](https://github.com/sergi/go-diff)|Generating the individual `Tensor` types|Can do without ||[MIT/BSD-like](https://github.com/sergi/go-diff/blob/master/LICENSE.txt), [Apache 2.0](https://github.com/sergi/go-diff/blob/master/APACHE-LICENSE-2.0.txt)|



#Usage#

Gorgonia works by creating a computation graph, and then executing it. Think of it as a programming language, but is limited to mathematical functions. In fact this is the dominant paradigm that the user should be used to thinking about. The computation graph is an [AST](http://en.wikipedia.org/wiki/Abstract_syntax_tree). 

Microsoft's [CNTK](https://github.com/Microsoft/CNTK), with its BrainScript, is perhaps the best at exemplifying the idea that building of a computation graph and running of the computation graphs are different thihngs, and that the user should be in different modes of thoughts when going about them. 

Whilst Gorgonia's implementation doesn't enforce the separation of thought as far as CNTK's BrainScript does, the syntax does help a little bit.

Here's an example - say you want to define a math expression `z = x + y`. Here's how you'd do it:

```go
import (
	"fmt"
	"log"

	. "github.com/chewxy/gorgonia"
)

func main() {
	g := NewGraph()

	var x, y, z *Node
	var err error

	// define the expression
	x = NewScalar(g, Float64, WithName("x"))
	y = NewScalar(g, Float64, WithName("y"))
	if z, err = Add(x, y); err != nil {
		log.Fatal(err)
	}

	// compile into a program
	prog, locMap, err := Compile(g)
	if err != nil {
		log.Fatal(err)
	}

	// create a VM to run the program on
	machine := NewTapeMachine(prog, locMap)

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

###VMs###

There are two VMs in the current version of Gorgonia:

* `TapeMachine`
* `LispMachine`

They function differently and take different inputs. The `TapeMachine` is useful for executing expressions that are generally static (that is to say the computation graph does not change). Due to its static nature, the `TapeMachine` is good for running expressions that are compiled-once-run-many-times (such as linear regression, SVM and the like).

The `LispMachine` on the other hand was designed to take a graph as an input, and executes directly on the nodes of the graph. If the graph change, simply create a new lightweight `LispMachine` to execute it on. The `LispMachine` is suitable for tasks such as creating recurrent neural networks without a fixed size.

Prior to release of Gorgonia, there was a third VM - a stack based VM that is similar to `TapeMachine` but deals with artificial gradients better. It may see light of day again, once this author has managed to fix all the kinks.

##Differentiation##

Gorgonia performs both symbolic and automatic differentiation. There are subtle differences between the two processes. The author has found that it's best to think of it this way - Automatic differentiation is differentiation that happens at runtime, concurrently with the execution of the graph, while symbolic differentiation is differentiation that happens during the compilation phase. 

Runtime of course, refers to the execution of the expression graph, not the program's actual runtime.

With the introduction to the two VMs, it's easy to see how Gorgonia can perform both symbolic and automatic differentiation. Using the same example as above, the reader should note that there was no differentiation done. Instead, let's try with a `LispMachine`:

```go
import (
	"fmt"
	"log"

	. "github.com/chewxy/gorgonia"
)

func main() {
	g := NewGraph()

	var x, y, z *Node
	var err error

	// define the expression
	x = NewScalar(g, Float64, WithName("x"))
	y = NewScalar(g, Float64, WithName("y"))
	if z, err = Add(x, y); err != nil {
		log.Fatal(err)
	}

	// set initial values then run
	Let(x, 2.0)
	Let(y, 2.5)

	// by default, LispMachine performs forward mode and backwards mode execution
	m := NewLispMachine(g)
	if err = m.RunAll(); err != nil {
		log.Fatal(err)
	}

	fmt.Printf("z: %v", z.Value())

	if xgrad, err := x.Grad(); err != nil {
		fmt.Printf("dz/dx: %v", xgrad)
	}

	if ygrad, err := y.Grad(); err != nil {
		fmt.Printf("dz/dy: %v", ygrad)
	}

	// Output:
	// z: 4.5
	// dz/dx: 1
	// dz/dy: 1
}
```

Of course, Gorgonia also supports the more traditional symbolic differentiation like in Theano:

```go
	g := NewGraph()

	var x, y, z *Node
	var err error

	// define the expression
	x = NewScalar(g, Float64, WithName("x"))
	y = NewScalar(g, Float64, WithName("y"))
	if z, err = Add(x, y); err != nil {
		log.Fatal(err)
	}

	// symbolically differentiate z with regards to x and y
	// this adds the gradient nodes to the graph g
	var Grads Nodes
	if grads, err = Grad(z, x, y); err != nil {
		log.Fatal(err)
	}

	// compile into a program
	prog, locMap, err := Compile(g)
	if err != nil {
		log.Fatal(err)
	}

	// create a VM to run the program on
	machine := NewTapeMachine(prog, locMap)

	// set initial values then run
	Let(x, 2.0)
	Let(y, 2.5)
	if err = machine.RunAll(); err != nil {
		log.Fatal(err)
	}

	fmt.Printf("%v", z.Value())
	if xgrad, err := x.Grad(); err != nil {
		fmt.Printf("dz/dx: %v", xgrad)
	}

	if ygrad, err := y.Grad(); err != nil {
		fmt.Printf("dz/dy: %v", ygrad)
	}

	// Output:
	// z: 4.5
	// dz/dx: 1 | 1
	// dz/dy: 1 | 1
```

Currently Gorgonia only performs backwards mode automatic differentiation (aka backpropagation), although one may observe the vestiges of an older version which supported forwards mode differentiation in the existence of `*dualValue`. It may return in the future.

##Graph##

A lot has been said about a computation graph or an expression graph. But what is it exactly? Think of it as an AST for the math expression that you want. Here's the graph for the examples (but with a vector and a scalar addition instead) above:

![graph1](https://raw.githubusercontent.com/chewxy/gorgonia/master/media/exprGraph_example1.png)

By the way, Gorgonia comes with nice-ish graph printing abilities. Here's an example of a graph of the equation `y = x²` and its derivation:

![graph1](https://raw.githubusercontent.com/chewxy/gorgonia/master/media/exprGraph_example2.png)

To read the graph is easy. The expression builds from bottom up, while the derivations build from top down. This way the derivative of each node is roughly on the same level. 

Red-outlined nodes indicate that it's a root node. Green outlined nodes indicate that they're a leaf node. Nodes with a yellow background indicate that it's an input node. The dotted arrows indicate which node is the gradient node for the pointed-to node.

Concretely, it says that `c42011e840` (`dy/dx`) is the gradient node of the input `c42011e000` (which is `x`).

###Node Rendering###

A Node is rendered thusly:

<table>
<tr><td>ID</td><td>node name :: type</td></tr>
<tr><td>OP*</td><td>op name :: type</td></tr>
<tr><td colspan="2">shape</td></tr>
<tr><td colspan="2">compilation metadata</td></tr>
<tr><td>Value†</td><td>Gradient</td></tr>
</table>

###Additional Notes###

* If it's an input node, then the Op row will not show up.
* If there are no Values bound to the node, it will show up as NIL. However, when there are values and gradients, it will try to as best as possible display the values bound to the node.



#API Stability#
Gorgonia's API is as of right now, not considered stable. It will be stable from version 1.0 forwards.

1.0 is defined by when the test coverage hits 90%, and the relevant `Tensor` methods have been completed.

#Roadmap#

Here are the goals for Gorgonia, sorted by importance 

- [ ] 90+% test coverage. Current coverage is 50% for Gorgonia and 75% for the Tensor packages.
- [ ] More advanced operations (like `einsum`). The current Tensor operators are pretty primitive.
- [ ] TravisCI for this package.
- [ ] Clean out the tests. The tests were the results of many years of accumulation. It'd be nice to refactor them out nicely.
- [ ] Improve performance especially re: allocation, minimize impact of type system.
- [ ] Improve Op extensibility by exposing/changing the Op interface to be all exported, and not a mix of exported and unexported methods (Alternatively, create a `Compose` Op type for extensibility). This way everyone can make their own custom `Op`s.
- [ ] Refactor the CuBLAS package as well as the Blase package.
- [ ] Distributed computing. The ability to spread jobs out across multiple machines and communicating with each other has been attempted at least 3 times, but failed each time.
- [ ] Better documentation on why certain decisions were made, and the design of Gorgonia in gneral.
- [ ] Higher order derivative optimization algorithms (LBFGS comes to mind)
- [ ] Derivative-free optimization algorithms

#Goals#
The primary goal for Gorgonia is to be a *highly performant* machine learning/graph computation-based library that can scale across multiple machines. It should bring the appeal of Go (simple compilation and deployment process) to the ML world. It's a long way from there currently, however, the baby steps are already there.

The secondary goal for Gorgonia is to provide a platform for exploration for non-standard deep-learning and neural network related things. This includes things like neo-hebbian learning, corner-cutting algorithms, evolutionary algorithms and the like. 


#Contributing#

Obviously since you are most probably reading this on Github, Github will form the major part of the workflow for contributing to this package.

See also: CONTRIBUTING.md


##Contributors and Significant Contributors##
All contributions are welcome. However, there is a new class of contributor, called Significant Contributors. 

A Significant Contributor is one who has shown *deep understanding* of how the library works and/or its environs.  Here are examples of what constitutes a Significant Contribution:

* Wrote significant amounts of documentation pertaining to **why**/the mechanics of particular functions/methods and how the different parts affect one another
* Wrote code, and tests around the more intricately connected parts of Gorgonia
* Wrote code and tests, and have at least 5 pull requests accepted
* Provided expert analysis on parts of the package (for example, you may be a floating point operations expert who optimized one function)
* Answered at least 10 support questions.

Significant Contributors list will be updated once a month (if anyone even uses Gorgonia that is).

#How To Get Support#
The best way of support right now is to open a ticket on Github.


#Licence#

Gorgonia is licenced under a variant of Apache 2.0. It's for all intents and purposes the same as the Apache 2.0 Licence, with the exception of not being able to commercially profit directly from the package unless you're a Significant Contributor (for example, providing commercial support for the package). It's perfectly fine to profit directly from a derivative of Gorgonia (for example, if you use Gorgonia as a library in your product)

Everyone is still allowed to use Gorgonia for commercial purposes (example: using it in a software for your business).
