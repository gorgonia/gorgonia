## Known Bugs ##
Tests: 
	- Inverse tests fail, and have been disabled (not generated)
	- Bugs involving changing from int/uint type to float type and back
		- Identity tests for Pow

Edge Cases:
	
This fails due to loss of accuracy from conversion:

```go
// identity property of exponentiation: a ^ 1 ==  a
a := New(WithBacking([]int(1,2,3)))
b, _ := a.PowOf(1) // or a.Pow(New(WithBacking([]{1,1,1})))
t := a.ElemEq(b) // false
```

Large number float operations - inverse of Vector-Scalar ops have not been generated because tests to handle the correctness of weird cases haven't been written