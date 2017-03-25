package tensor

import (
	"fmt"

	"github.com/pkg/errors"
)

// An AP is an access pattern. It tells the various ndarrays how to access their data through the use of strides
// Through the AP, there are several definitions of things, most notably there are two very specific "special cases":
//		Scalar has Dims() of 0. However, its shape can take several forms:
//			- (1, 1)
//			- (1)
//		Vector has Dims() of 1, but its shape can take several forms:
//			- (x, 1)
//			- (1, x)
//			- (x)
//		Matrix has Dims() of 2. This is the most basic form. The len(shape) has to be equal to 2 as well
//		ndarray has Dims() of n.
// The AP also supports masked array interation through slice maskedStrides. It's length should be the
// same as strides (corresponding to data), in fact they can refer to the same underlying int array.
// Having it as separate from strides however allows for more flexibililty,such as masking entire dimensions
// with one value.
type AP struct {
	shape       Shape // len(shape) is the operational definition of the dimensions
	strides     []int // strides is usually calculated from shape
	fin         bool  // is this struct change-proof?
	maskStrides []int //strides of mask. Should be same length as strides.

	// future stuff
	// triangle byte // up = 0xf0;  down = 0x0f; symmetric = 0xff; not a triangle = 0x00
}

// NewAP creates a new AP, given the shape and strides. An optional argument, maskStrides, applies
// to  masked arrays. maskStrides can be either []bool, or []int, but must be the same length as
// shape. If bool, it indicates which dimensions of shape can be masked. For example, given shape
// [2,2,2] and maskStrides [true,true,false], the AP calculates mask strides internally as [2,1,0].
// If argument maskStrides is []int, then it is directly used as the AP's maskStrides value.
func NewAP(shape Shape, strides []int, argStrides ...interface{}) *AP {
	ap := &AP{
		shape:   shape,
		strides: strides,
		fin:     true,
	}

	if len(argStrides) > 0 {
		ap.SetMaskStrides(argStrides[0])
	}
	return ap
}

// SetShape is for very specific times when modifying the AP is necessary, such as reshaping and doing I/O related stuff
//
// Caveats:
//
// - SetShape will recalculate the strides.
//
// - If the AP is locked, nothing will happen
func (ap *AP) SetShape(s ...int) {
	if !ap.fin {
		// scalars are a special case, we don't want to remove it completely
		if len(s) == 0 {
			ap.shape = ap.shape[:0]
			ap.strides = ap.strides[:0]
			ap.maskStrides = ap.maskStrides[:0]
			return
		}

		if ap.shape != nil {
			// ReturnInts(ap.shape)
			ap.shape = nil
		}
		if ap.strides != nil {
			// ReturnInts(ap.strides)
			ap.strides = nil
		}
		ap.shape = Shape(s).Clone()
		ap.strides = ap.shape.calcStrides()

		if ap.IsMasked() {
			b := BorrowBools(len(s))
			for i := 0; i < len(s); i++ {
				if ap.maskStrides[i] > 0 {
					b[i] = true
				}
			}
			ap.maskStrides = ap.shape.calcStridesWithMask(b)
			ReturnBools(b)
		}
	}
}

//SetMaskStrides is for setting mask stride
func (ap *AP) SetMaskStrides(s interface{}) {
	var maskStrides []int
	switch m := s.(type) {
	case []int:
		if len(m) == len(ap.shape) || ap.IsColVec() || ap.IsRowVec() {
			maskStrides = m
		} else {
			panic("maskStrides slice differs in length from shape")
		}
	case []bool:
		var useMask = false
		for i := range m {
			useMask = useMask || m[i]
		}
		if !useMask {
			return
		}
		if len(m) == len(ap.shape) || ap.IsColVec() || ap.IsRowVec() {
			maskStrides = ap.shape.calcStridesWithMask(m)
		} else {
			panic("maskStrides slice differs in length from shape")
		}
	default:
		panic("maskStrides only supports []int and []bool types")
	}
	if cap(ap.maskStrides) >= len(maskStrides) {
		copy(ap.maskStrides, maskStrides)
		ReturnInts(maskStrides)
		ap.maskStrides = ap.maskStrides[:len(ap.shape)]
	} else {
		ap.maskStrides = maskStrides
	}
	//To do: decide whether to keep length equality restriction, as well as panic.
}

// locking and unlocking is used to ensure that the shape and stride doesn't change (it's not really safe though, as a direct mutation of the strides/shape would still mutate it, but at least the dimensions cannot change)
func (ap *AP) lock()   { ap.fin = true }
func (ap *AP) unlock() { ap.fin = false }

// Shape returns the shape of the AP
func (ap *AP) Shape() Shape { return ap.shape }

// Strides returns the strides of the AP
func (ap *AP) Strides() []int { return ap.strides }

// MaskStrides returns the mask strides of the AP
func (ap *AP) MaskStrides() []int { return ap.maskStrides }

// Dims returns the dimensions of the shape in the AP
func (ap *AP) Dims() int { return ap.shape.Dims() }

// Size returns the expected array size of the shape
func (ap *AP) Size() int { return ap.shape.TotalSize() }

// LogicalSize returns the expected array size of the shape
func (ap *AP) LogicalSize() int {
	if ap.IsRowVec() {
		return ap.shape[1] * ap.strides[0]
	}
	if ap.IsColVec() {
		return ap.shape[0] * ap.strides[0]
	}
	if ap.IsVector() {
		return ap.shape[0] * ap.strides[0]
	}
	for i := range ap.shape {
		if ap.strides[i] > 0 {
			return ap.strides[i] * ap.shape[i]
		}
	}
	return 0
}

// MaskSize returns the expected array size of the mask. This may be less than data size
func (ap *AP) MaskSize() int {
	if len(ap.maskStrides) < 1 {
		return 0
	}
	if ap.IsScalar() {
		return 1 //Unlikely that this line can be reached, look into.
	}
	var tap = AP{shape: ap.shape, strides: ap.maskStrides}
	return tap.LogicalSize()
}

// String implements fmt.Stringer and runtime.Stringer
func (ap *AP) String() string { return fmt.Sprintf("%v", ap) }

// Format implements fmt.Formatter
func (ap *AP) Format(state fmt.State, c rune) {
	if ap.IsMasked() {
		fmt.Fprintf(state, "Shape: %v, Stride: %v, MaskStride: %v, Lock: %t", ap.shape, ap.strides, ap.maskStrides, ap.fin)
	} else {
		fmt.Fprintf(state, "Shape: %v, Stride: %v, Lock: %t", ap.shape, ap.strides, ap.fin)
	}
}

// IsVector returns whether the access pattern falls into one of three possible definitions of vectors:
//		vanilla vector (not a row or a col)
//		column vector
//		row vector
func (ap *AP) IsVector() bool { return ap.shape.IsVector() }

// IsColVec returns true when the access pattern has the shape (x, 1)
func (ap *AP) IsColVec() bool { return ap.shape.IsColVec() }

// IsRowVec returns true when the access pattern has the shape (1, x)
func (ap *AP) IsRowVec() bool { return ap.shape.IsRowVec() }

// IsScalar returns true if the access pattern indicates it's a scalar value
func (ap *AP) IsScalar() bool { return ap.shape.IsScalar() }

// IsMasked returns true if the access pattern indicates it is a masked value
func (ap *AP) IsMasked() bool {
	if ap.IsColVec() || ap.IsRowVec() {
		return len(ap.maskStrides) == 1
	}
	return len(ap.maskStrides) > 0 && (len(ap.shape) == len(ap.maskStrides))
}

// IsMatrix returns true if it's a matrix. This is mostly a convenience method. RowVec and ColVecs are also considered matrices
func (ap *AP) IsMatrix() bool { return len(ap.shape) == 2 }

// Clone clones the *AP. Clearly.
func (ap *AP) Clone() (retVal *AP) {
	retVal = BorrowAP(len(ap.shape))
	copy(retVal.shape, ap.shape)
	copy(retVal.strides, ap.strides)
	if cap(retVal.maskStrides) < len(ap.maskStrides) {
		retVal.maskStrides = make([]int, len(ap.maskStrides))
	}
	copy(retVal.maskStrides[:len(ap.maskStrides)], ap.maskStrides)

	// handle vectors
	retVal.shape = retVal.shape[:len(ap.shape)]
	retVal.strides = retVal.strides[:len(ap.strides)]
	retVal.maskStrides = retVal.maskStrides[:len(ap.maskStrides)]
	retVal.fin = ap.fin
	return
}

// C returns true if the access pattern is C-contiguous array
func (ap *AP) C() bool {
	return ap.strides[len(ap.strides)-1] == 1
}

// F returns true if the access pattern is Fortran contiguous array
func (ap *AP) F() bool {
	return ap.strides[0] == 1
}

// S returns the metadata of the sliced tensor.
func (ap *AP) S(size int, slices ...Slice) (newAP *AP, ndStart, ndEnd int, err error) {
	if len(slices) > len(ap.shape) {
		// error
		err = errors.Errorf(dimMismatch, len(ap.shape), len(slices))
		return
	}
	ndEnd = size

	newShape := ap.shape.Clone()    // the new shape
	dims := ap.Dims()               // reported dimensions
	newStrides := make([]int, dims) // the new strides

	for i := 0; i < dims; i++ {
		var sl Slice
		if i <= len(slices)-1 {
			sl = slices[i]
		}

		size := ap.shape[i]

		var stride int
		if ap.IsVector() {
			// handles non-vanilla vectors
			stride = ap.strides[0]
		} else {
			stride = ap.strides[i]
		}

		var start, end, step int
		if start, end, step, err = SliceDetails(sl, size); err != nil {
			return
		}

		// a slice where start == end is []
		ndStart = ndStart + start*stride
		ndEnd = ndEnd - (size-end)*stride
		if step > 0 {
			newShape[i] = (end - start) / step
			newStrides[i] = stride * step

			//fix
			if newShape[i] <= 0 {
				newShape[i] = 1
			}
		} else {
			newShape[i] = (end - start)
			newStrides[i] = stride
		}
	}

	if ndEnd-ndStart == 1 {
		// scalars are a special case
		newAP = new(AP)
		newAP.SetShape() // make it a Scalar
		newAP.lock()
	} else {

		// drop any dimension with size 1, except the last dimension
		for d := 0; d < dims; d++ {
			if newShape[d] == 1 /*&& d != t.dims-1  && dims > 2*/ {
				newShape = append(newShape[:d], newShape[d+1:]...)
				newStrides = append(newStrides[:d], newStrides[d+1:]...)
				d--
				dims--
			}
		}

		//fix up strides
		if newShape.IsColVec() {
			newStrides = []int{newStrides[0]}
		}

		newAP = NewAP(newShape, newStrides)
	}
	return
}

// T returns the transposed metadata based on the given input
func (ap *AP) T(axes ...int) (retVal *AP, a []int, err error) {
	if ap.IsMasked() {
		retVal, a, err = ap.maskedT(axes...)
	} else {
		retVal, a, err = ap.plainT(axes...)
	}
	return
}

// plainT is the underlying T implementation
func (ap *AP) plainT(axes ...int) (retVal *AP, a []int, err error) {
	// prep axes
	if len(axes) > 0 && len(axes) != ap.Dims() {
		err = errors.Errorf(dimMismatch, ap.Dims(), len(axes))
		return
	}

	dims := len(ap.shape)
	if len(axes) == 0 || axes == nil {
		axes = make([]int, dims)
		for i := 0; i < dims; i++ {
			axes[i] = dims - 1 - i
		}
	}
	a = axes

	// if axes is 0, 1, 2, 3... then no op
	if monotonic, incr1 := IsMonotonicInts(axes); monotonic && incr1 && axes[0] == 0 {
		err = noopError{}
		return
	}

	currentShape := ap.shape
	currentStride := ap.strides
	shape := make(Shape, len(currentShape))
	strides := make([]int, len(currentStride))

	switch {
	case ap.IsScalar():
		return
	case ap.IsVector():
		if axes[0] == 0 {
			return
		}
		copy(strides, currentStride)
		shape[0], shape[1] = currentShape[1], currentShape[0]
	default:
		copy(shape, currentShape)
		copy(strides, currentStride)
		err = UnsafePermute(axes, shape, strides)
		if err != nil {
			err = handleNoOp(err)
		}
	}

	retVal = BorrowAP(len(shape))
	copy(retVal.shape, shape)
	copy(retVal.strides, strides)

	if ap.IsVector() {
		retVal.strides = retVal.strides[:1]
	}

	return
}

// maskedT returns the transposed metadata based on the given input for a masked tensor
func (ap *AP) maskedT(axes ...int) (retVal *AP, a []int, err error) {
	var maskAP *AP
	tempAP := ap.Clone()
	retVal, a, err = tempAP.plainT(axes...)
	tempAP.strides = tempAP.maskStrides
	maskAP, _, _ = tempAP.plainT(axes...)
	if cap(retVal.maskStrides) < len(maskAP.strides) {
		retVal.maskStrides = make([]int, len(maskAP.strides))
	}
	copy(retVal.maskStrides[:len(maskAP.strides)], maskAP.strides)
	return
}

// TransposeIndex returns the new index given the old index
func TransposeIndex(i int, oldShape, pattern, oldStrides, newStrides []int) int {
	oldCoord, err := Itol(i, oldShape, oldStrides)
	if err != nil {
		panic(err) // or return error?
	}
	/*
		coordss, _ := Permute(pattern, oldCoord)
		coords := coordss[0]
		index, _ := Ltoi(newShape, strides, coords...)
	*/

	// The above is the "conceptual" algorithm.
	// Too many checks above slows things down, so the below is the "optimized" edition
	var index int
	for i, axis := range pattern {
		index += oldCoord[axis] * newStrides[i]
	}
	return index
}

// UntransposeIndex returns the old index given the new index
func UntransposeIndex(i int, oldShape, pattern, oldStrides, newStrides []int) int {
	newPattern := make([]int, len(pattern))
	for i, p := range pattern {
		newPattern[p] = i
	}
	return TransposeIndex(i, oldShape, newPattern, oldStrides, newStrides)
}

// BroadcastStrides handles broadcasting from different shapes.
//
// Deprecated: this function will be unexported
func BroadcastStrides(destShape, srcShape Shape, destStrides, srcStrides []int) (retVal []int, err error) {
	dims := len(destShape)
	start := dims - len(srcShape)

	if destShape.IsVector() && srcShape.IsVector() {
		return []int{srcStrides[0]}, nil
	}

	if start < 0 {
		//error
		err = errors.Errorf(dimMismatch, dims, len(srcShape))
		return
	}

	retVal = BorrowInts(len(destStrides))
	for i := dims - 1; i >= start; i-- {
		s := srcShape[i-start]
		switch {
		case s == 1:
			retVal[i] = 0
		case s != destShape[i]:
			// error
			err = errors.Errorf("Cannot broadcast from %v to %v", srcShape, destShape)
			return
		default:
			retVal[i] = srcStrides[i-start]
		}
	}
	for i := 0; i < start; i++ {
		retVal[i] = 0
	}
	return
}
