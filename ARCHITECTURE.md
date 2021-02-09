# Architecture #

This file describes the architecture of Gorgonia. Use this file as a guide to contributing to Gorgonia.

# Subsystems View: Overview #

Gorgonia consists of a few parts:

- [A system to manage and manipulate mathematical expressions](#expr).
- [A system to perform backpropagation](#backprop).
- [A set of systems to evaluate mathematical expressions](#eval).
- [A set of systems to perform gradient descent](#solver).
- [A set of "utility" systems that support the above](#utils).

Each of these parts have their own sub-parts. Let's explore.

<a name="expr"/>
## Mathematical Expressions ##

Instead of tediously explaining what a mathematical expression, we'll use examples and rely on the reader's ability to perform induction.

This is an example of a mathematical expression:

![\sigma(\mathbf{W}'\mathbf{x} + \mathbf{b})](https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Csigma%28%5Cmathbf%7BW%7D%27%5Cmathbf%7Bx%7D+%2B+%5Cmathbf%7B1%7D%29)

A mathematical expression is contained in an `*ExprGraph`. Each component of a mathematical is stored in a `*Node`.

The usual definitions of a mathematical expression would break down the above example into:

| Component | Type (Component Name) |
|---|---|
| ![\sigma](https://render.githubusercontent.com/render/math?math=%5Clarge+%5Ctextstyle+%5Csigma) | function |
| ![\mathbf{W}](https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+%5Cmathbf%7BW%7D) | variable |
| !['](https://render.githubusercontent.com/render/math?math=%5Clarge+%5Ctextstyle+%27) | operator |
| ![\mathbf{x}](https://render.githubusercontent.com/render/math?math=%5Clarge+%5Ctextstyle+%5Cmathbf%7Bx%7D) | variable |
| ![+](https://render.githubusercontent.com/render/math?math=%5Clarge+%5Ctextstyle+%2B) | operator |
| ![\mathbf{1}](https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+%5Cmathbf%7B1%7D) | constant |

In Gorgonia, these various component types are all represented as a `*Node`. The `*Node` data type provides methods for finding out what "type" of node it is.

The `*Node` data type is a fairly broad data type - it is used in many different contexts:

- Manipulating mathematical expressions.
- Storing the link between a term and its derivatives (and vice versa).
- Storing the results of computations.

### The Graph ###

The `*ExprGraph` data type stores an entire expression. As hinted, a mathematical expression is a directed acyclic graph (DAG), usually in form of a tree. The reason for preferring a DAG over a tree is that a DAG automatically optimizes operations. So `3*(x+y) + 4*(x+y)` will only execute `(x+y)` once.

Further, having an entire expression in a graph also allows for easier reference when it comes to backpropagation. From this point forwards,  "mathematical expression" may be used interchangably with "graph".

### Ops and Functions ###

An operator is a notational shortcut representing an operation. A function is an abstract notion of a map from one set of values to another set. The actual act of going from one set to another set is an operation.

Hence for simplicity's sake, there is no difference between an operator and a function in Gorgonia. They are all treated as functions, but named `Op`. The `Op`s in Gorgonia may be found in files starting with `op_`.

The definition of the `Op` interface may be found in `op.go`.

From this point, "`Op`" may be interchangably used with "function".


### Variables, Weights and Constants ###

In most deep learning frameworks, there is a separation of notions between weights and variables. Often, variables are what the user/programmer may set. So in the example above, only ![\mathbf{x}](https://render.githubusercontent.com/render/math?math=%5Clarge+%5Ctextstyle+%5Cmathbf%7Bx%7D) is a variable, while ![\mathbf{W}](https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+%5Cmathbf%7BW%7D) is a generic tensor/node.

In Gorgonia, there is no such separation. Mathematically speaking, ![\mathbf{W}](https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+%5Cmathbf%7BW%7D) and ![\mathbf{x}](https://render.githubusercontent.com/render/math?math=%5Clarge+%5Ctextstyle+%5Cmathbf%7Bx%7D) are both variables in that they both do not have any values assigned to them (at least until the user assigns a value).

We adopt the terms used by any high school calculus textbook: a variable is defined by the fact that its derivative varies with the values assigned to it. Conversely, the derivative of a constant will always be 0.

Constants are implemented in Gorgonia as an `Op` denoting a value as a constant.

<a name="backprop"/>
## Backpropagation ##

One of the core abilites of Gorgonia is the ability to do compute partial derivatives. This is done in two ways in Gorgonia:

- Symbolic Differentiation
- Forwards Mode Automatic Differentiation
- Reverse Mode Automatic Differentiation (FUTURE)

Some functions are differentiable, and some functions are not. Gorgonia handles both kinds.

### Symbollic Differentiation ###

Symbolic differentiation is done by manipulating the graph. Consider the following expression:

![c = a \times b](https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+c+%3D+a+%5Ctimes+b)

Here, ![\times](https://render.githubusercontent.com/render/math?math=%5Ctextstyle+%5Ctimes) is an `Op`, while ![c](https://render.githubusercontent.com/render/math?math=%5Ctextstyle+c) is the dependent variable; and ![a](https://render.githubusercontent.com/render/math?math=%5Ctextstyle+a) or ![b](https://render.githubusercontent.com/render/math?math=%5Ctextstyle+b) are independent variables. All four components are represented as `*Node` in a `*ExprGraph`.

The partial derivatives are defined as follows:

![\begin{aligned}
\frac{\partial c}{\partial a} &= b\\
\frac{\partial c}{\partial b} &= a\\
\end{aligned}](https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cbegin%7Baligned%7D%0A%5Cfrac%7B%5Cpartial+c%7D%7B%5Cpartial+a%7D+%26%3D+b%5C%5C%0A%5Cfrac%7B%5Cpartial+c%7D%7B%5Cpartial+b%7D+%26%3D+a%5C%5C%0A%5Cend%7Baligned%7D)

Hence when we perform a symbolic differentiation, we're adding new `*Node`s to the graph, each representing a partial derivative of ![c](https://render.githubusercontent.com/render/math?math=%5Ctextstyle+c) with regards to ![a](https://render.githubusercontent.com/render/math?math=%5Ctextstyle+a) or ![b](https://render.githubusercontent.com/render/math?math=%5Ctextstyle+b).

An `Op` that supports symbolic differentiation must implement `SDOp`.

### Automatic Differentiation ###

Automatic differentiation does not modify the graph. Instead, differentiation is done on values. This is handled by the evaluating VM (see section on evaluation).

To aid in automatic differentiation, a `dualValue` type is also used. A `*dualValue` is exactly what it suggests: a value that contains two values - usually the value and a gradient.

An `Op` that supports automatic differentiation must implement `ADOp`.

<a name="eval"/>
## Evaluation of Mathematical Expressions ##

A mathematical expression is useless by itself. Here the word "useless" is meant literally. By itself, a mathematical expression does nothing. However, the expression may be evaluated to get values out of the expression.

The expression `1 + 2` does nothing. However, when we evaluate it, we get `3` as a result. `3` is a value. So are `1` and `2`. Specifically, `1` and `2` are constant values.

A value in Gorgonia is defined by the `Value` interface. It is defined in `values.go`.

To evaluate the graph, Gorgonia uses `VM`s (virtual machine). There are three main `VM`s:

* `*tapeMachine`
* `*lispMachine`
* `*goMachine`

The names of the `VM`s are suggestive of their operational semantics. `*tapeMachine` acts like a Turing machine with a finite tape. `*lispMachine` acts like a [Lisp machine](https://en.wikipedia.org/wiki/Lisp_machine). `*goMachine` acts like everything is concurrent process.

In order to evaluate using a `*tapeMachine`, the mathematical expression needs to be first compiled into a program that runs on the `*tapeMachine`. `*goMachine` and `*lispMachine` runs off the graph directly, and both these machines support automatic differentiation.

### `*tapeMachine` ###

### `*lispMachine` ###

### `*goMachine` ###

<a name="solver" />
## Gradient Descent ##

Gorgonia comes equipped with gradient descent functionalities. The main abstract data type that defines a gradient descent algorithm is the `Solver`. There are multiple `Solver`s implemented in Gorgonia.
All `Solver`s rely on a `ValueGrad`. A `ValueGrad` is anything that can provide a value and a gradient (also itself a value).

<a name="utils" />
## Other Subsystems ##

### Type System ###

Gorgonia is heavily reliant on a type system (emphasis on _a_, not _the_). The type system of the mathematical expressions is a [traditional Hindley-Milner style type system](https://en.wikipedia.org/wiki/Hindley%E2%80%93Milner) - it is powered by the [hm](https://github.com/chewxy/hm) library. It allows for the type of a new node to be inferred. However, it also means that `Op` implementors must contend with it.

Here's a quick primer. Like all the previous definitions, it'll be defined by examples instead of rigorous inductive definitions.

- `Float64` is a type. Specifically, it's a type constant: it has a name, and it describes what a bunch of bytes is supposed to represent.
- `a` is a type variable. It may be replaced by other types when inference is being done.
- `Matrix a` is a type. It is a type scheme (also occasionally called a polytype).
- `a → b` is a function type from type `a` to  type `b`. What are `a` and `b`? They're type variables, so they can be anything. This is also called a function's type signature.

The following table is a "translation" from Gorgonia's type system to the closest equivalent in Go's type system:

| Gorgonia Type System Term | Go Type System Term | Notes |
|---|---|---|
| `Float64` | `float64` | Most types in Go are type constants |
| `a` | `interface{}` | This is not an entirely accurate analogy. An `interface{}` is resolved at runtime, while `a` is resolved at compile time (of the mathematical expression, not of the Go program). |
| `Vector a` | `[]T` | The `T` represents any data type, which is what `a` is. The difference is `T` has to be a concrete data type at the time of programming, while `a` doesn't have to be. |
| `a → b` | `func(a T) U` | `T` and `U` are meta variables that the programmer has to fill in at the time of programming |

An `Op` has to return a `hm.Type` in order to fulfil the interface. Most often, you will want to return a function type.

#### An Example: Addition ####
Let's say we're creating an addition `Op`. What inputs do additions take? An addition function is a function that takes two numbers of the same type (let's call this type `a`) and returns the results which is also of the same type. So, an addition `Op` would have the type `a → a → a` or `(a, a) → a`.

The difference between `a → a → a` and `(a, a) → a` is subtle. Let us  translate this type signature into a Go type signature for analogy's sake to help with understandability:

| Gorgonia Type Signature | Go Type Signature | Notes |
|---|---|---|
| `a → a → a` | `func add(a interface{}) func(a interface{}) interface{}` | Also known as "curried" function |
| `(a, a) → a` | `func add(a, b interface{}) interface{}` | |

While it's natural to gravitate towards `(a, a) → a`, Gorgonia strongly prefers `a → a → a`. Why? Because when an `Op` is defined with only one input and one output, it makes it easier to optimize the graph.

Having defined an addition `Op` with signature `a → a → a`, we can now have a look at what Gorgonia does with that information.

The primary thing that the type system is useful for is unification. Our addition `Op` has a type signature `a → a → a`. Now, let's say we want to create a `*Node` representing an addition between `x` adn `y`, which are also both `*Node`s. The type of `x` is the first argument, and has a type `Float64`. The first step is to replace the type variable in the first parameter of the type signature of the addition `Op`. This is better represented below:

```
   a   → a → a
   ↑
Float64
```

Having replaced `a` with `Float64`, we find the remainder of the type signature to have been transformed into `Float64 → Float64`.

Let's say the type of `y`, the second argument is `Float32`. The next step is to repeat the same steps above: replace the type variables in the first parameter of the type signature with `Float32`.

```
Float64 → Float64
   ↑
Float32
```

This replacement cannot happen for two reasons:

1. There is no type variable in the function type signature.
2. `Float32` is not `Float64`.

Hence an error occurs. Following from this we can see that the addition `Op` is both generic and restrictive at the same time. The very same `Op` allows `Matrix a → Matrix a → Matrix a` to be a legal definition, while `Matrix a → Matrix b → Matrix a` to immediately cause an error.

#### Another Example: Scale ####

Now, let's consider another `Op`: a scaling function. For simplicity's sake, let's say the scale `Op` only operates on vectors. We can define a type signature as follows: `Vector a → a → Vector a`. An analogous Go type signature would be `func(floats []float64, scalar float64) []float64`. Except the scale `Op` can work on any data type. If the first argument is `Vector Float64` then the remainder function will have the signature `Float64 → Vector Float64`.

#### Where Is The Type System Used? ####

The type system powers the creation of new `*Node`. `ApplyOp` is the function that takes an `Op`, and the children `*Node` and returns a new `*Node` representing the `Op`.


<summary>

<details> Quick Recipes </details>

### Create a type constant ###

The only "kind" of type constant we will really use with Gorgonia is the `tensor.Dtype`, which itself is just a wrapper around a `reflect.Type`.

```
T := reflect.TypeOf(v)
C := tensor.Dtype{T}
```

### Create a 3 dimensional Tensor Type of any underlying datatype ###

Gorgonia comes with functions that allow you to define a Tensor type.

```
of := hm.TypeVariable('a')
dims := 3
T := &gorgonia.TensorType{
	Dims: dims,
	Of: of,
}
```

### Create a Matrix Type of Float64s ###

```
T := &gorgonia.TensorType {
	Dims: 2,
	Of: tensor.Float64,
}
```

### Create a Function Type `a → b` ###

```
a := hm.TypeVariable('a')
b := hm.TypeVariable('b')
T := hm.NewFnType(a, b)
```

</summary>

# File-Based View #

Another way to get around this repository is via the files. The files are quite well named (barring a few files whose names came from a particularly childish developer, @chewxy).

## API Files ##

The majority of APIs of Gorgonia can be found in

* `api_gen.go`
* `operations.go`
* `gorgonia.go`

## Op Files ##

All files pertaining to the implementation of `Op` can be found in files starting with `op_`


# How Gorgonia is Developed #

There are large parts of Gorgonia that are machine generated. TODO
