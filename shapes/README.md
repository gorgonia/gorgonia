# About

Package `shapes` provides the machinery for dealing with the metainformation of shapes.

# Why a shape package?

The shape package defines a syntax and semantics that describes shapes of values.

The goal is:

- to describe the notion of shape;
- to implement various automation to work with shapes (example a a tool that verify if shapes of different inputs are compatible to be used with one operator)

# What is a Shape? #

A shape is a piece of meta-information that tells you something about a tensor.

A rank-2 tensor is also commonly known as a matrix (mathematicians and physicists, please bear with the inaccuracies and informalities).

## Example ##

Let's consider the following matrix:

```
⎡1  2  3⎤
⎣4  5  6⎦
```

We can describe this matrix by saying "it's a matrix with 2 rows and 3 columns". We can write this as (2, 3).

`(2, 3)` is the shape of the matrix.

This is a very convenient way of describing N-dimensiional tensors. We don't have to give the dimensions names like "layer", "row" or "column". We would rapidly run out of names! Instead, we just index them by their number. So in a 3 dimensional shape, instead of saying "layer", we say "dimension 0".  In a 2 dimensional shape, "row" would be "dimension 0".


## Components of a Shape ##

Given a shape, let's explore the components of the shape:

```
 +--+--+--------- size
 v  v  v
(2, 3, 4) <------ shape
 ^̄  ^̄  ^̄
 +--+--+--------- dimensions
```

Each "slot" in a shape is called a dimension. The number of dimensions in a shape is called a rank (though confusingly enough, in the Gorgonia family of libraries, it's called `.Dims()`). There are 3 slots in the example, so it's a 3 dimensional shape. Each number in a slot is called the size of the dimension. For example, in dimension 1, the size is 3. We say that dimension 1 is of size 3.

To use the traditional named dimensions - recall in this instance, that dimension 1 is "rows". So we say there are 3 rows.

# Shape Expr #


The primary data structure that the package provides is the shape expression. A shape expression is given by the following [BNF](https://en.wikipedia.org/wiki/Backus–Naur_form)

```text
<shape>    ::= <unit>| "("<integer>",)" | "("<integer>","<shape>")" |
               "("<shape>","<integer>")" | "("<variable>",)" |
               <binaryOperationExpression> | <nameOfT>

<integer>  ::= <digit> [ <integer> ]
<digit>    ::= 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
<unit>     ::= "()"
<variable>  ::= ...
<binaryOperationExpression>  ::= "("<variable> <binop> <integer>",)" |
                       "("<variable> <binop> <variable>",) |
                       "("<integer> <binop> <variable>",)"
<binop>   ::= + | *

<expression> ::= <shape> | I n <expression> | D <expression |
                 K <expression> | Σ <expression> | Π <expression> |
                 <variable> | <expression → <expression> |
                 (<expression> → <expression>) @ <expression> // TODO
T ::= (I n E,) | (D E,) | (Σ E,) | (Π E,) // TODO
```

<details>
<summary>A compact BNF </summary>

The original design for the language of shapes was written in a compact BNF. This was expanded by Olivier Wulveryck into something that traditional computer scientists are more familiar with.

Here, the original compact BNF is preserved.


```
E ::= S | I n E | D E | K E | Σ E | Π E | a | E -> E | (E -> E) @ E
S ::= () | (n,) | (n, S) | (S, n) | (a,) | B | T
T ::= (I n E,) | (D E,) | (Σ E,) | (Π E,)
B ::=  (a O n,) | (a O b,) | (n O a,)
O ::= + | ×
n,m ::= ℕ
a,b ::= variables
```
</details>


The BNF might be brief, but it is  dense, so let's break it down.

## Primitive Shape Construction ##


```text
<shape>    ::= <unit>| "("<integer>",)" | "("<integer>","<shape>")" |
               "("<shape>","<integer>")" | "("<variable>",)" |
               <binaryOperationExpr> | <nameOfT>
```

What this snippet says is: a shape (`<shape>`) is one of: the four possible definitions of a shape:

1. `<unit>` - This is the shape of scalar values. `()` is pronounced "unit".
2. `"("<integer>",)"` - `<integer>` is any positive number. Example of a shape: `(10,)`.
3. `"("<integer>","<shape>")"` - an `<interer>` folowwed by another `<shape>`. Example: `(8, (10,))`.
4. `"("<shape>","<integer>")"`  - a `<shape>` followed by another `<integer>`. Example: `((8, (10,)), 20)`.


From the snippet, we have generated 4 examples of shapes: `()`, `(10,)`, `(8, (10,))` and `((8, (10,)), 20)`. This isn't how we normally write shapes. We would normally write them as `(10,)`, `(8, 10)` and `(8, 10, 20)`. So what's the difference?

In fact, the following are equivalent;

* `(8, (10,))` = `(8, 10)`
* `((8, (10,)), 20)`  = `((8, 10), 20)` = `(8, (10, 20))` = `(8, 10, 20)`

What this says is that the primitive shape construction is [associative](https://en.wikipedia.org/wiki/Associative_property). This is useful as we can now omit the additional parentheses.

### On <unit> ###

The unit `()` is particularly interesting. In a while we'll see why it's called a "unit".

For now, given the associativity rules, what would `(10, ())` be equivalent to? One possible answer is `(10, )` - afterall, we're just removing the internal parentheses. Another possible answer is `(10, 1)`.

 TODO write more.


## Introducing Variables ##

Now, let us focus, once again on line 2 of the BNF, but on the latter parts:

```
<shape> ::= ... | (<variable>,) | ...
```

What this says is that you can create a <shape>  using a variable. I'll use `x` and `y` for real variables.

Example: `(x,)` is a shape. Combining this rule with the rules from above, we can see that `(x, y)` is also a valid shape. So is `(10, x)` or `(x, 10)`

To recap using the examples we have seen so far, these are exammples of valid shapes:

* `(x,)`
* `(x, 10)`
* `(10, x)`
* `(10, 20, x)`
* `(10, x, 20)`

## Introducing Binary Operations ##

In the following snippet (still on line 2 of the BNF), this is introduced:

```
<shape> ::= ... | <binaryOperationExpression> | ...
```

And `<binaryOperationExpression>` is defined in the following line as:

```
<binaryOperationExpr>  ::= "("<variable> <binop> <integer>",)" |
                       "("<variable> <binop> <variable>",) |
                       "("<integer> <binop> <variable>",)"
```

What this says is any valid `<binaryOperationExpr>` is also a valid `<shape>`.

`<binop>` is defined as:

```
<binop>   ::= + | *
```
That is, a valid mathematical operation is a addition or a multiplication.

So what line 3 of the BNF says is that these are valid shapes:

* `(x + 10,)`
* `(x + y,)`
* `(10 + x,)`


An astute reader will observe that `"("<integer> <binop> <integer>")"` (example: `(10 × 20,)`) isn't allowed. The reasoning is clear - if you know ahead of time what the resulting shape will be, don't put a mathematical expression in. Note that though there is a restriction in the fomal language, this is not enforced by the package machinery.

## Recap 1 ##

To recap, these are all examples valid shapes:

* `()`
* `(10,)`
* `(10, 20)`
* `(x,)`
* `(x, 10)`
* `(10, x)`
* `(x, y)`
* `(x+20,)`
* `(x×y,)`
* `(x+20, 10)`
* `(10, x×y)`

We wil leave the last part of the definition of  `S` (`S ::= ... | T`) to after we've introduced the notion of expressions.



# Why So Complicated? #
Why introduce all these complicated things? We introduce these complicated things because we want to do things with shape expressions.


# Enumeration of Common Functions #

* `Add : a → a → a`
* `Apply : a → (() → ()) → a`
* `Random: Void → a`
* `Concat (D a = D b): a → () → b → c`
* `Conv1D : ??? `
* `Conv2D : ??? `
* `GlovalAveragePool : ???`
* `KeepDims (D a = D b): a → (a → b) → b`
* `MaxPool1D: ???`
* `MaxPool2D: ???`
* `MatMul: (a, b) → (b, c) → (a, c)`
* `MatVecMul: (a, b) → (b,) → (a,)`
* `Norm: a → ()`
* `Reshape : a → b → b`
* `Slice (D b = m-n+1): a → (I n a → I m a) → b`
* `Sum: a → ()` all axes
* `Sum[along = x] (D a = (D b) + 1): a → b`
* `Sum[along = x] (I x b = 1): a → b`
* `Transpose : ???`
