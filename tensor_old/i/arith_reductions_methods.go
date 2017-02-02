package tensori

import "github.com/chewxy/gorgonia/tensor/types"

/*
This file contains code that deals with the reduction of a Tensor by axis.


All of the code in this file is structured in such a way that they're embarassingly parallel.
This message will serve as a reminder until all the code in this file which are embarassingly parallel
has been parallelized

List of functions parallalized:
	<crickets>
*/

// Reduce takes a function, a default value and reduces the axis using the function.
func (t *Tensor) Reduce(f func(a, b int) int, def int, axis int) (retVal *Tensor, err error) {
	if axis >= t.Dims() {
		err = types.DimMismatchErr(axis, t.Dims())
		return
	}

	var newShape types.Shape
	for i, s := range t.Shape() {
		if i == axis {
			continue
		}
		newShape = append(newShape, s)
	}

	retVal = NewTensor(WithShape(newShape...))

	lastAxis := t.Dims() - 1
	switch axis {
	case 0:
		size := t.Shape()[axis]
		split := len(t.data) / size
		copy(retVal.data[0:split], t.data[0:split])

		start := split
		for i := 0; i < size-1; i++ {
			for j := 0; j < split; j++ {
				retVal.data[j] = f(retVal.data[j], t.data[start+j])
			}
			start += split
		}
	case lastAxis:
		size := t.Shape()[axis]
		var at int
		for start := 0; start <= len(t.data)-size; start += size {
			r := reduce(f, def, t.data[start:start+size]...)
			retVal.data[at] = r
			at++
		}
	default:
		/*
			A visual explanation for the following algorithm:
			Say you have a (2,3,2,3)-shaped tensor. It looks something like that:

				0  1  2		18 19 20
				3  4  5		21 22 23

				6  7  8		24 25 26
				9 10 11		27 28 29

				12 13 14	30 31 32
				15 16 17	33 34 35

			We'll consider only the first layer (0 - 17), since the same actions can be repeated upon the second layer

			Let's say we want to reduce axis 2. The resulting shape would be (2,3,3) (it's as simple as removing the second axis from the shape).
			This is how the matrix is laid out in the strided slice:

			t.data:
				0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15  16  17
				+   +   +   +   +   +   +   +   +   +    +   +  +   +   +    +   +   +
				|   |   |   |   |   |   |   |   |   |    |   |  |   |   |    |   |   |
				|   |   |   |   |   |   |   |   |   |    |   |  |   |   |    |   |   |
				+---------------------+-+-----------------------+   |   |    |   |   |
				    |   |   |   |   | |     |   |   |    |   |      |   |    |   |   |
				    +--------------------+--+-----------------------+   |    |   |   |
				        |   |   |   | |  |      |   |    |   |          |    |   |   |
				        +-----------------------------------------------+    |   |   |
				            |   |   | |  |      |   |    |   |               |   |   |
				            |   |   | +  +      +   |    |   |               |   |   |
			res.data index  |   |   | 0  1      2   |    |   |               |   |   |
				            |   |   |               |    |   |               |   |   |
				            +----------------------------+-+-----------------+   |   |
				                |   |               |      | |                   |   |
				                +------------------------------------------------+---+
				                    |               |      | |                       |
				                    +------------------------+-----+-----------------+
				                                    |      |       |
				                                    |      |       |
				                                    +      +       +
			res.data indes                          3      4       5

			It's a little difficult to see, but elements (0, 6, 12) from t.data will be written to index 0 of the reduced strided array. This is the listing:
				reduce (t[0], t[6], t[12]) -> res[0]
				reduce (t[1], t[7], t[13]) -> res[1]
				reduce (t[2], t[8], t[14]) -> res[2]
				...

			These are the basic rules:
				size of axis to be reduced  = number of elements to be reduced
				stride of axis to be reduced = how many to skip innerStart
				newStride[0] = expected number of groups within a layer

			The main idea is then this - we loop through the resulting array, and for each index, we find the elements of the original array that is supposed to fit in
			there, and then we reduce it. It is quite self explanatory.
		*/

		size := t.Shape()[axis]
		oStride := t.Strides()[0]
		stride := t.Strides()[axis]
		expected := retVal.Strides()[0]
		for i := 0; i < t.Shape()[0]; i++ {
			// this loop can be parallelized!
			start := i * oStride
			data := t.data[start : start+oStride]
			var innerStart, strideTrack int
			for j := 0; j < expected; j++ {
				for k := 0; k < size; k++ {
					readFrom := innerStart + k*stride
					writeTo := i*expected + j
					retVal.data[writeTo] = f(retVal.data[writeTo], data[readFrom])
				}
				strideTrack++
				if strideTrack >= stride {
					strideTrack = 0
					innerStart += stride
				}
				innerStart++
			}
		}

	}

	return
}

func (t *Tensor) reduce(axis int, zeroFn func(a, b []int), oneFn func([]int) int, defFn func(a, b int) int) (retVal *Tensor) {
	if t.IsScalar() {
		return t
	}

	var newShape types.Shape
	for i, s := range t.Shape() {
		if i == axis {
			continue
		}
		newShape = append(newShape, s)
	}
	retVal = NewTensor(WithShape(newShape...))
	size := t.Shape()[axis]
	switch axis {
	case 0:
		// most efficient
		split := len(t.data) / size
		copy(retVal.data[0:split], t.data[0:split])

		start := split
		for i := 0; i < size-1; i++ {
			zeroFn(retVal.data, t.data[start:start+split])
			start += split
		}
	case len(t.Shape()) - 1:
		// second most efficient
		var at int
		for start := 0; start <= len(t.data)-size; start += size {
			s := oneFn(t.data[start : start+size])
			retVal.data[at] = s
			at++
		}
	default:
		outerSize := t.Shape()[0]
		outerStride := t.Strides()[0]
		stride := t.Strides()[axis]
		expected := retVal.Strides()[0]

		for i := 0; i < outerSize; i++ {
			start := i * outerStride
			data := t.data[start : start+outerStride]
			var innerStart, strideTrack int
			for j := 0; j < expected; j++ {
				for k := 0; k < size; k++ {
					readFrom := innerStart + k*stride
					writeTo := i*expected + j
					a := retVal.data[writeTo]
					b := data[readFrom]
					if k == 0 {
						retVal.data[writeTo] = b
					} else {
						retVal.data[writeTo] = defFn(a, b)
					}
				}
				strideTrack++
				if strideTrack >= stride {
					strideTrack = 0
					innerStart += stride
				}
				innerStart++
			}
		}
	}
	return
}

// Sum sums up the elements of the Tensor along the given axes.
// If multiple axes are given, then this method will sum the Tensor according the the order of the axes provided
func (t *Tensor) Sum(along ...int) (retVal *Tensor, err error) {
	monotonic, incr1 := types.IsMonotonicInts(along) // if both are true, then it means all axes are accounted for, then it'll return a scalar value
	if (monotonic && incr1 && len(along) == t.Dims()) || len(along) == 0 {
		ret := sum(t.data)
		retVal = NewTensor(AsScalar(ret))
		return
	}
	retVal = t
	prev := -1
	dims := len(retVal.Shape())
	for _, axis := range along {
		if prev == -1 {
			prev = axis
		}
		if axis > prev {
			axis--
		}

		if axis >= dims {
			err = types.DimMismatchErr(axis, retVal.Dims())
			return
		}

		retVal = retVal.sum(axis)
	}
	return
}

// sum does work of summing
func (t *Tensor) sum(axis int) (retVal *Tensor) {
	return t.reduce(axis, vecAdd, sum, add)
}

// Max returns the max of the elements of the tensor along the given axes.
// If multiple axes are given, then this method will return the max of the Tensor according the the order of the axes provided
func (t *Tensor) Max(along ...int) (retVal *Tensor, err error) {
	monotonic, incr1 := types.IsMonotonicInts(along) // if both are true, then it means all axes are accounted for, then it'll return a scalar value
	if (monotonic && incr1 && len(along) == t.Dims()) || len(along) == 0 {
		ret := sliceMax(t.data)
		retVal = NewTensor(AsScalar(ret))
		return
	}
	retVal = t
	prev := -1
	dims := len(retVal.Shape())
	for _, axis := range along {
		if prev == -1 {
			prev = axis
		}
		if axis > prev {
			axis--
		}

		if axis >= dims {
			err = types.DimMismatchErr(axis, retVal.Dims())
			return
		}

		retVal = retVal.max(axis)
	}
	return
}

func (t *Tensor) max(axis int) (retVal *Tensor) {
	return t.reduce(axis, vecMax, sliceMax, max)
}

// Min returns the max of the elements of the tensor along the given axes.
// If multiple axes are given, then this method will return the min of the Tensor according the the order of the axes provided
func (t *Tensor) Min(along ...int) (retVal *Tensor, err error) {
	monotonic, incr1 := types.IsMonotonicInts(along) // if both are true, then it means all axes are accounted for, then it'll return a scalar value
	if (monotonic && incr1 && len(along) == t.Dims()) || len(along) == 0 {
		ret := sliceMin(t.data)
		retVal = NewTensor(AsScalar(ret))
		return
	}
	retVal = t
	prev := -1
	dims := len(retVal.Shape())

	for _, axis := range along {
		if prev == -1 {
			prev = axis
		}
		if axis > prev {
			axis--
		}

		if axis >= dims {
			err = types.DimMismatchErr(axis, retVal.Dims())
			return
		}

		retVal = retVal.min(axis)
	}
	return
}

func (t *Tensor) min(axis int) (retVal *Tensor) {
	return t.reduce(axis, vecMin, sliceMin, min)
}
