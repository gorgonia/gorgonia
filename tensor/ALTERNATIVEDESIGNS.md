# Alternateive Designs #

This document holds the alternative designs for the various tensor data structures that had been tried in the past and why they didn't make it to the final designs. That doesn't mean that the current design is the best. It just means that the authors may not have gone far enough with these other designs.


## Single interface, multiple packages ##

In this design, there is a single interface for dense tensors, which is rather similar to the one that is currently there right now:

```
type Tensor interface {
	Shape() Shape
	Strides() []int
	Dtype() Dtype
	Dims() int
	Size() int
	DataSize() int

	// Basic operations all tensors must support
	Slice(...Slice) (Tensor, error)
	At(...int) (interface{}, error)
	SetAt(v interface{}, coord ...int) error
	Reshape(...int) error
	T(axes ...int) error
	UT()
	Transpose() error // Transpose actually moves the data
	Apply(fn interface{}, opts ...FuncOpt) (Tensor, error)
}
```

The idea is then to have subpackages for each type that would implement the `Tensor` like such:

```
// in tensor/f32
type Tensor struct {
	
}
// implements tensor.Tensor

// in tensor/f64
type Tensor struct {
	
}
// implements tensor.Tensor
```

Additionally there are interfaces which defined operational types:

```
type Adder interface {
	Add(other Tensor) (Tensor, error)
}

type Number interface {
	Adder
	Suber
	Muler
	Diver
}

type Real interface {
	Number
	Tanher
	Exper
}

type Complex interface {
	Real
}
```

And there are functions which operated on the `Tensor`s:

```
func Add(a, b Tensor) (Tensor, error){
	if adder, ok := a.(Adder); ok {
		return a.Add(other)
	}
	return nil, errors.New("Cannot Add: Not an Adder")
}
```


### Pros ###

It is very idiomatic Go, and no reflection was used. It is an ideal model of an abstract data type. 

### Cons ###

1. Having all packages import a common "tensor/types" (which holds `*AP`, `Shape` and `Slice` definitions).
2. It'd be ideal to keep all the packages in sync in terms of the methods and functions that the subpackages export. In reality that turns out to be more difficult than expected. 
3. Performance issues in hot loops: In a number of hot loops, the amount of `runtime.assertI2I2` ended up taking up a large portion of the cycles.
4. Performance issues wrt allocation of objects. Instead of a single pool, every sub pacakge would have to implement its own object pool and manage it.
5. There was a central registry of `Dtype`s, and a variant of the SQL driver pattern was used (you had to `import _ "github.com/chewxy/gorgonia/tensor/f32" to register the `Float32` Dtype). This is ugly. 
6. Cross package requirements: for `Argmax` and `Argmin` related functions, it'd be nice to be able to return a `Tensor` of `int`. That meant having `tensor/i` as a core dependency in the rest of the packages. 

#### Workarounds ####

* `Slice` is a interface. All packages that implement `tensor.Tensor` *coulc* implement their own `Slice`. But that'd be a lot of repeat work. 
* `AP` and `Shape` could be made interfaces, but for the latter it means dropping the ability to loop through the shape dimensions.
* Keeping the packages in sync could be solved with code generation programs, but if we were to do that, we might as well merge everything into one package

### Notes for revisits ###

This idea is nice. I'd personally love to revisit (and do from time to time). If we were to revisit this idea, there would have to be some changes, which I will suggest here:

1. Make `Transpose` and `T` functions that work on `Tensor` instead of making it a `Tensor`-defining method. This would be done the same way as `Stack` and `RollAxis` and `Concat`. 
2. Perhaps re-weight the importance of having a inplace transpose. The in-place transpose was the result of dealing with a very large matrix when my machine didn't have enough memory. It's generally slower than reallocating a new backing array anyway.


# One struct, multiple backing interfaces #

In this design, we abstract away the backing array into a interface. So we'd have this:

```
type Tensor struct {
	*AP

	t Dtype
	data Array
}

type Array interface {
	Len() int
	Cap() int
	Get(int) interface{}
	Set(int, interface{}) error
	Map(fn interface{}) error
}
```

And we'd have these types which implemented the `Array` interface: 

```
type Ints []int
type F32s []float64
type F64s []float32

// and so on and so forth, and each would implement Array
```

### Pros ###

* Multiple subpackages only when necessary (external, "unhandled" dtypes )
* Shared definition of `*AP`, `Shape`, `Dtype` (no more use of a common package)
* Clean package structure - easier to generate code for

### Cons ###

* Difficult to implement other tensor types (sparse for example)
* VERY VERY slow

The slowness was caused by excessive calls from `runtime.convT2E` when using `Get` and `Set` methods which for primitive types cause plenty of allocations on the heap. It was unacceptably slow for any deep learning work.

#### Workarounds ####

Type switch on known data types, and use slower methods for out-of-bounds data types that do not have specializations on it. This led to ugly unwieldly code, and also changes the pressure from `runtime.convT2E` to `runtime.assertI2I2`, which while performs better than having to allocate primitive values on the heap, still led to a lot of unnecessary cycles being spent on it. 

# Reflection + Pointers + Interfaces #

This was the design that was reigning before the refactor at #127. 

The idea is to combine parts of the first attempt and second attempt and fill up the remaining missing bits with the use of reflections.