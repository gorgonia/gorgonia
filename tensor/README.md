# Package Tensor #
Package `tensor` is a package that provides efficient, generic n-dimensional arrays in Go. Also in this package are functions and methods that are used commonly in arithmetic, comparison and linear algebra operations.

## Introduction ##

## The Types of Tensors ##

## Generic Features ##

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