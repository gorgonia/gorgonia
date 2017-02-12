# Package Tensor #
Package `tensor` is a package that provides efficient, generic (by some definitions of generic) n-dimensional arrays in Go. Also in this package are functions and methods that are used commonly in arithmetic, comparison and linear algebra operations.

The main purpose of this package is to support the operations required by [Gorgonia](https://github.com/chewxy/gorgonia).

## Introduction ##
In the data analysis world, [Numpy](http://http://www.numpy.org/) and [Matlab](https://www.mathworks.com/products/matlab.html) currently reign supreme. Both tools rely heavily on having performant n-dimensional arrays, or tensors. **There is an obvious need for multidimensional arrays in Go**. 

While slices are cool, a large majority of scientific and numeric computing work relies heavily on matrices (two-dimensional arrays), three dimensional arrays and so on. In Go, the typical way of getting multidimensional arrays is to use something like `[][]T`. Applications that are more math heavy may opt to use the very excellent Gonum [`matrix` package](https://github.com/gonum/matrix). What then if we want to go beyond having a `float64` matrix? What if we wanted a 3-dimensional `float32` array?

It comes to reason then there should be a data structure that handles these things. The `tensor` package fits in that niche. 

### Basic Idea: Tensor ###
A tensor is a multidimensional array. It's like a slice, but works in multiple dimensions.

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

The `*Dense` tensor is the primary tensor and is represented by a singular flat array, regardless of dimensions. See the [Design of `*Dense`](#design-of-dense) section for more information. It can hold any data type.

### Compressed Sparse Column Matrix ###

Coming soon

### Compressed Sparse Row Matrix ###

Coming soon

## Usage ##

To create a matrix with package `tensor` is easy:

```go
// Creating a (2,2) matrix:
a := New(WithShape(2, 2), WithBacking([]int{1, 2, 3, 4}))
fmt.Printf("a:\n%v\n", a)

// Output:
// a:
// ⎡1  2⎤
// ⎣3  4⎦
//
```

To create a 3-Tensor is just as easy - just put the correct shape and you're good to go:

```go 
// Creating a (2,3,4) 3-Tensor
b := New(WithBacking(Range(Float32, 0, 24)), WithShape(2, 3, 4))
fmt.Printf("b:\n%1.1f\n", b)

// Output:
// b:
// ⎡ 0.0   1.0   2.0   3.0⎤
// ⎢ 4.0   5.0   6.0   7.0⎥
// ⎣ 8.0   9.0  10.0  11.0⎦
```

Accessing and Setting data is fairly easy (be warned, this is the inefficient way if you want to do a batch access/setting):

```go
// Accessing data:
b := New(WithBacking(Range(Float32, 0, 24)), WithShape(2, 3, 4))
x, _ := b.At(0, 1, 2) // in Numpy syntax: b[0,1,2]
fmt.Printf("x: %v\n", x)

// Setting data
b.SetAt(float32(1000), 0, 1, 2)
fmt.Printf("b:\n%v", b)

// Output:
// x: 6
// b:
// ⎡   0     1     2     3⎤
// ⎢   4     5  1000     7⎥
// ⎣   8     9    10    11⎦

// ⎡  12    13    14    15⎤
// ⎢  16    17    18    19⎥
// ⎣  20    21    22    23⎦
```

Bear in mind to pass in data of the correct type. This example will cause a panic:

```go
// Accessing data:
b := New(WithBacking(Range(Float32, 0, 24)), WithShape(2, 3, 4))
x, _ := b.At(0, 1, 2) // in Numpy syntax: b[0,1,2]
fmt.Printf("x: %v\n", x)

// Setting data
b.SetAt(1000, 0, 1, 2)
fmt.Printf("b:\n%v", b)
```


## Design of `*Dense` ##

The design of the `*Dense` tensor is quite simple in concept. However, let's start with something more familiar. This is a visual representation of a slice in Go (taken from rsc's excellent blog post on [Go data structures](https://research.swtch.com/godata)):

![slice](https://raw.githubusercontent.com/chewxy/gorgonia/master/tensor/media/slice.png)

The data structure for `*Dense` is similar, but a lot more complex. Much of the complexity comes from the need to do accounting work on the data structure as well as preserving references to memory locations. This is how the `*Dense` is defined:

```go 
type Dense struct {
	*AP
	data unsafe.Pointer
	hdr *reflect.SliceHeader
	t Dtype

	// other fields elided for simplicity's sake
}
```

And here's a visual representation of the `*Dense`.

![dense](https://raw.githubusercontent.com/chewxy/gorgonia/master/tensor/media/dense.png)

`*Dense` draws its inspiration from Go's slice. Underlying it all is a flat array, and access to elements are controlled by `*AP`. Where a Go is able to store its metadata in a 3-word stucture (obiviating the need to allocate memory), a `*Dense` unfortunately needs to allocate some memory. The majority of the data is stored in the `*AP` structure, which contains metadata such as shape, stride, and methods for accessing the array.

The `*Dense` tensor stores the address of the first element in the `data` field as an `unsafe.Pointer` mainly to ensure that the underlying data does not get garbage collected. The `hdr` field of the `*Dense` tensor is there to provide a quick and easy way to translate back into a slice for operations that use familiar slice semantics, of which much of the operations are dependent upon.

By default, `*Dense` operations try to use the language builtin slice operations by casting the `hdr` field into a slice. However, to accomodate a larger subset of types, the `*Dense` operations have a fallback to using pointer arithmetic to iterate through the slices for other types with non-primitive kinds (yes, you CAN do pointer arithmetic in Go. It's slow and unsafe). The result is slower operations for types with non-primitive kinds.

### Memory Allocation###
`New()` functions as expected - it returns a pointer of `*Dense` to a array of zeroed memory. The underlying array is allocated, depending on what `ConsOpt` is passed in. With `New()`, `ConsOpt`s are used to determine the exact nature of the `*Dense`. It's a bit icky (I'd have preferred everything to have been known statically at compile time), but it works. Let's look at some examples:

``` go
x := New(Of(Float64), WithShape(2,2))
```

This will allocate a `float64` array of size 4.

```go
x := New(WithShape(2,2))
```

This will panic - because the function doesn't know what to allocate - it only knows to allocate an array of *something* for the size of 4.

```go
x := New(WithBacking([]int{1,2,3,4}))
```

This will NOT fail, because the array has already been allocated (the `*Dense` reuses the same backing array as the slice passed in). Its shape will be set to `(4)`.


### Other failed designs ###

One particular failed design of which I am interested in revisiting is direcly inspired by the slice design. Instead of having to allocate `*Dense`, a `Dense` data structure comprising of fewer words are used. In particular, the `*AP` would be reduced to being a 2 word structure. However, there are limitations arising from that - the number of dimensions allowed would be a maximum of 2 (a 64-bit uint can be split into 2 32-bit uints, representing shape). That would defeat the purpose of the design of this package, which is to go beyond two dimensions. Eventually the data structure built up to roughly around the same number of words as the current definition. 

Given the Go runtime isn't generally guaranteed to stay the same, it would have been a better decision to stick with passing around pointers.

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
Much of the code in this package is generated. The code to generate them is in the directory `genlib`. 

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