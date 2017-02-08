# Package Tensor #
Package `tensor` is a package that provides efficient, generic (by some definitions of generic) n-dimensional arrays in Go. Also in this package are functions and methods that are used commonly in arithmetic, comparison and linear algebra operations.

The main purpose of this package is to support the operations required by [Gorgonia](https://github.com/chewxy/gorgonia).

## Introduction ##
In the data analysis world, [Numpy](http://http://www.numpy.org/) and [Matlab](https://www.mathworks.com/products/matlab.html) currently reign supreme. Both tools rely heavily on having performant n-dimensional arrays, or tensors. **There is an obvious need for multidimensional arrays in Go**. 

While slices are cool, a large majority of scientific and numeric computing work relies heavily on matrices (two-dimensional arrays), three dimensional arrays and so on. In Go, the typical way of getting multidimensional arrays is to use something like `[][]T`. Applications that are more math heavy may opt to use the very excellent Gonum [`matrix` package](https://github.com/gonum/matrix). What then if we want to go beyond having a `float64` matrix? What if we wanted a 3-dimensional `float32` array?

It comes to reason then there should be a data structure that handles these things. The `tensor` package fits in that niche. 

### Basic Concepts: Tensor ###
A tensor is a multidimensional array. 

With slices, there are usage patterns that are repeated enough that warrant abstraction - `append`, `len`, `cap`, `range` are abstrations used to manipulate and query slices. Additionally slicing operations (`a[:1]` for example) are also abstractions provided by the language. Andrew Gerrand wrote a very good write up on [Go's slice usage and internals](https://blog.golang.org/go-slices-usage-and-internals). 

Tensors come with their own set of usage patterns and abstractions. Most of these have analogues in slices, enumerated below (do note that certain slice operation will have more than one tensor analogue - this is due to the number of options available):

| Slice Operation | Tensor Operation |
|:---------------:|:----------------:|
| `len(a)`        | `T.Shape()`      |
| `cap(a)`        | `T.DataSize()`   |
| `a[:]`          | `T.Slice(...)`   |
| `a[0]`          | `T.At(x,y)`      |
| `append(a, ...)`| `T.Stack(...)`, `T.Concat(...)`   |
| `copy(dest, src)`| `T.CopyTo(dest)`, `tensor.Copy(dest, src)` |
| `for _, v := range a` | `for i, err := iterator.Next(); err == nil; i, err = iterator.Next()` | 

Some operations for a tensor does not have direct analogues to slice operations. However, they stem from the same idea, and can be considered a superset of all operations common to slices. They're enumerated below:

| Tensor Operation | Basic idea in slices |
|:----------------:|:--------------------:|
|`T.Strides()`     | The stride of a slice will always be one element |
|`T.Dims()`        | The dimensions of a slice will always be one |
|`T.Size()`        | The size of a slice will always be its length |
|`T.Dtype()`       | The type of a slice is always known at compile time |
|`T.Reshape()`     | Given the shape of a slice is static, you can't really reshape a slice |
|`T.T(...)` / `T.Transpose()` / `T.UT()` | No equivalent with slices |


## The Types of Tensors ##

As of the current revision of this package, only dense tensors are supported. Support for sparse matrix (in form of a sparse column matrix and dictionary of keys matrix) will be coming shortly.


### Dense Tensors ###

The `*Dense` tensor is the primary tensor and is represented by a singular flat array, regardless of dimensions.

## Generic Features ##

Example:

```go 

x := New(WithBacking([]string{"hello", "world", "hello", "world"}), WithShape(2,2))
x = New(WithBacking([]int{1,2,3,4}), WithShape(2,2))
```

The above code will not cause a compile error, because the structure holding the underlying array (of `string`s and then of `int`s) is a `*Dense`. 

One could argue that this sidesteps the compiler's type checking system, deferring it to runtime (which a number of people consider dangerous). However, tools are being developed to type check these things, and until Go does support typechecked generics, unfortunately this will be the way it has to be.


Currently, the tensor package supports limited type of genericity - limited to a tensor of any primitive type. 

## How This Package is Developed ##


## Things Knowingly Untested For ##
- Inverse tests fail, and have been disabled (not generated)
- Bugs involving changing from int/uint type to float type and back
		- Identity tests for Pow

### Edge Cases: ###

Due to use of `testing/quick`, a number of edge cases were found, and primarily are caused by loss of accuracy. Handling these edge cases is deferred to the user of this package, hence all the edge cases are enumerated here:

1.  In `Pow` related functions, there are loss of accuracy issues
	This fails due to loss of accuracy from conversion:

	```go
		// identity property of exponentiation: a ^ 1 ==  a
		a := New(WithBacking([]int(1,2,3)))
		b, _ := a.PowOf(1) // or a.Pow(New(WithBacking([]{1,1,1})))
		t := a.ElemEq(b) // []bool{false, false, false}
	```

2. Large number float operations - inverse of Vector-Scalar ops have not been generated because tests to handle the correctness of weird cases haven't been written

TODO: 

* Identity optimizations for op
* Zero value optimizations
* fix SVD tests
* fix Random() - super dodgy