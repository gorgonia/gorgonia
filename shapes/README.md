package shapes provides the machinery for dealing with the metainformation of shapes.

# Shape Expr #

The primary data structure that the package provides is the shape expression. A shape expression is given by the following BNF

```
E ::= S | I n E | D E | K E | Σ E | Π E | a | E -> E | (E -> E) @ E
S ::= () | (n,) | (n, S) | (S, n) | (a,) | B | T
T ::= (I n E,) | (D E,) | (Σ E,) | (Π E,)
B ::=  (a O n,) | (a O b,) | (n O a,)
O ::= + | ×
n,m ::= ℕ
a,b ::= variables
```

The BNF might be brief, but it is  dense, so let's break it down.

## Primitive Shape Construction ##

We'll start with the following snippet (line 2), which we call the primitive shape construction:

```
S ::= () | (n,) | (n, S) | (S, n) | ...
```

What this snippet says is: a shape (S) is one of: the four possible definitions of a shape:

1. `()` - This is the shape of scalar values. `()` is pronounced "unit".
2. `(n,)` - `n` is any positive number, as specified in line 6 - `n,m ::= ℕ`. Example of a shape: `(10,)`.
3. `(n, S)` - a `n` followed by another `S`. Example: `(8, (10,))`.
4. `(S, n)` - a `S` followed by another `n`. Example: `((8, (10,)), 20)`.

From the snippet, we have generated 4 examples of shapes: `(), `(10,)`, `(8, (10,))` and `((8, (10,)), 20)`. This isn't how we normally write shapes. We would normally write them as `(10,)`, `(8, 10)` and `(8, 10, 20)`.

In fact, the following are equivalent;

* `(8, (10,))` = `(8, 10)`
* `((8, (10,)), 20)`  = `((8, 10), 20)` = `(8, (10, 20))`   `(8, 10, 20)`

What this says is that the primitive shape construction is [associative](https://en.wikipedia.org/wiki/Associative_property). This is useful as we can now omit the additional parentheses.

### On Unit ###

The unit `()` is particularly interesting. In a while we'll see why it's called a "unit".

For now, given the associativity rules, what would `(10, ())` be equivalent to? One possible answer is `(10, )` - afterall, we're just removing the internal parentheses. Another possible answer is `(10, 1)`.

 TODO write more.


## Introducing Variables ##

Now, let us focus, once again on line 2 of the BNF, but on the latter parts:

```
S ::= ... | (a,) | ...
```

What this says is that you can create a shape (S) using a variable. `a` and `b` are metavariables. I'll use `x` and `y` for real variables.

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
S ::= ... | B | ...
```

And `B` is defined in the following line as:

```
B ::=  (a O n,) | (a O b,) | (n O a,)
```

What this says is any valid `B` is also a valid `S`. So what is a `B`? It's a binary operation expression.

`O` is defined as:

```
O ::= + | ×
```
That is, a valid mathematical operation is a addition or a multiplication.

So what line 3 of the BNF says is that these are valid shapes:

* `(x + 10,)`
* `(x + y,)`
* `(10 + x,)`


An astute reader will observe that `(n O m)` (example: `(10 × 20,)`) isn't allowed. The reasoning is clear - if you know ahead of time what the resulting shape will be, don't put a mathematical expression in. Note that though there is a restriction in the fomal language, this is not enforced by the package machinery.

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
