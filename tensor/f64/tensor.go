package tensorf64

import (
	"fmt"

	"github.com/chewxy/gorgonia/tensor/types"
)

type Tensor struct {
	*types.AP
	data []float64

	// backup AP. When a transpose is done, the old *AP is backed up here, for easy untransposes
	old           *types.AP
	transposeWith []int

	// if viewOf != nil, then this *Tensor is a view.
	viewOf *Tensor
}

// a consOpt is a tensor construction option
type consOpt func(*Tensor)

// NewTensor creates a new Float64 *Tensor
func NewTensor(opts ...consOpt) *Tensor {
	t := new(Tensor)
	t.AP = new(types.AP)

	for _, opt := range opts {
		opt(t)
	}
	t.fix()
	// TODO: sanity check
	if err := t.sanity(); err != nil {
		panic(err)
	}
	return t
}

// newBorrowedTensor tries to borrow from the tensor pool. It isn't zeroed!
func newBorrowedTensor(size int, opts ...consOpt) *Tensor {
	t := BorrowTensor(size)

	for _, opt := range opts {
		opt(t)
	}

	t.fix()
	if err := t.sanity(); err != nil {
		panic(err)
	}
	return t
}

func newTensor(size int) *Tensor {
	t := new(Tensor)
	t.AP = new(types.AP)
	t.setShape(size)
	t.data = make([]float64, size)
	return t
}

// Ones create a ndarray of the given shape, and fills it with 1.0
func Ones(shape ...int) *Tensor {
	if len(shape) == 0 {
		one := float64(1) //@DEFAULTONE
		return NewTensor(AsScalar(one))
	}

	t := BorrowTensor(types.Shape(shape).TotalSize())
	for i := range t.data {
		t.data[i] = float64(1) //@DEFAULTONE
	}

	t.setShape(shape...)
	return t
}

// Zeroes create a ndarray of a given shape and fills it with float64(0) (which is Go's default value)
// It's here mainly as a convenience function
func Zeroes(shape ...int) *Tensor {
	t := BorrowTensor(types.Shape(shape).TotalSize())
	t.setShape(shape...)
	t.Zero()
	return t
}

// I creates the identity matrix (usually a square) matrix with 1s across the diagonals, and zeroes elsewhere, like so:
//		Matrix(4,4)
// 		⎡1  0  0  0⎤
// 		⎢0  1  0  0⎥
// 		⎢0  0  1  0⎥
// 		⎣0  0  0  1⎦
// While technically an identity matrix is a square matrix, in attempt to keep feature parity with Numpy,
// the I() function allows you to create non square matrices, as well as an index to start the diagonals.
//
// For example:
//		T = I(4, 4, 1)
// Yields:
//		⎡0  1  0  0⎤
//		⎢0  0  1  0⎥
//		⎢0  0  0  1⎥
//		⎣0  0  0  0⎦
//
// The index k can also be a negative number:
// 		T = I(4, 4, -1)
// Yields:
// 		⎡0  0  0  0⎤
// 		⎢1  0  0  0⎥
// 		⎢0  1  0  0⎥
// 		⎣0  0  1  0⎦
func I(r, c, k int) (retVal *Tensor) {
	retVal = NewTensor(WithShape(r, c))
	if k >= c {
		return
	}

	i := k
	if k < 0 {
		i = (-k) * c
	}

	var s *Tensor
	var err error
	end := c - k
	if end > r {
		s, err = retVal.Slice(nil)
	} else {
		s, err = retVal.Slice(rs{0, end, 0})
	}
	defer ReturnTensor(s)

	if err != nil {
		panic(err)
	}

	var nexts []int
	iter := types.NewFlatIterator(s.AP)
	nexts, err = iter.Slice(rs{i, s.Size(), c + 1})

	for _, v := range nexts {
		s.data[v] = float64(1) //@DEFAULTONE
	}
	return
}

// WithBacking is a construction option for NewTensor
// Use it as such:
//		backing := []float64{1,2,3,4}
// 		t := NewTensor(WithBacking(backing))
// It can be used with other construction options like WithShape
func WithBacking(a []float64) consOpt {
	f := func(t *Tensor) {
		t.data = a
	}
	return f
}

// WithShape is a construction option for NewNDArray - it creates the ndarray in the required shape
func WithShape(dims ...int) consOpt {
	f := func(t *Tensor) {
		t.setShape(dims...)
	}
	return consOpt(f)
}

// AsScalar is a construction option for representing a scalar value as an ndarray
func AsScalar(s float64) consOpt {
	f := func(t *Tensor) {
		t.setShape()
		t.data = []float64{s}
	}
	return f
}

func (t *Tensor) setShape(s ...int) {
	t.Unlock()
	t.SetShape(s...)
	t.Lock()
	return
}

func (t *Tensor) fix() {
	if t.Shape() == nil {
		if t.data == nil {
			return
		}
		// otherwise, the shape is automatically a [n,1]
		rows := len(t.data)
		if rows == 1 {
			t.SetShape() // it's a scalar!
		} else {
			t.SetShape(rows) // it's a vector (unknown whether column or row)
		}
	}

	if t.data == nil {
		size := t.Shape().TotalSize()
		t.data = make([]float64, size)
	}
	t.Lock() // don't put this in a defer - if t.data == nil and t.Shape() == nil. then leave it unlocked
}

// sanity is a function that sanity checks that a tensor is correct.
func (t *Tensor) sanity() error {
	if t.AP != nil && t.Shape() == nil && t.data == nil {
		return types.EmptyTensorError()
	}

	size := len(t.data)
	expected := t.Size()
	if t.viewOf == nil && size != expected && !t.IsScalar() {
		return types.NewError(types.ShapeMismatch, "Expected backing data to have %d elements from shape %v. Got %d instead", expected, t.Shape(), size)
	}
	// TODO: sanity check for views
	return nil
}

func (t *Tensor) oshape() types.Shape {
	if t.old != nil {
		return t.old.Shape()
	}
	return t.Shape()
}

func (t *Tensor) ostrides() []int {
	if t.old != nil {
		return t.old.Strides()
	}
	return t.Strides()
}

func (t *Tensor) Dtype() types.Dtype { return types.Float64 }
func (t *Tensor) Size() int          { return t.Shape().TotalSize() }
func (t *Tensor) DataSize() int      { return len(t.data) }

func (t *Tensor) Reshape(dims ...int) error {
	t.Unlock()
	t.SetShape(dims...)
	t.Lock()
	return t.sanity()
}

func (t *Tensor) Zero() {
	for i := range t.data {
		t.data[i] = float64(0) //@DEFAULTZERO
	}
}

// ScalarValue() returns the scalar value of a *Tensor,
// IF and ONLY IF it's a Tensor representation of a scalar value.
// This is required because operations like a (vec · vec) would return a scalar value.
// I didn't want to return interface{} for all the API methods, so the next best solution is to
// wrap the scalar value in a *Tensor
func (t *Tensor) ScalarValue() interface{} {
	if !t.IsScalar() {
		panic(fmt.Sprintf("ScalarValue only works when the Tensor is a representation of a scalar value. The value of the tensor is %v", t))
	}

	return t.data[0]
}

func (t *Tensor) Eq(other types.Tensor) bool {
	if ot, ok := other.(*Tensor); ok {
		if ot == t {
			return true
		}

		if len(ot.data) != len(t.data) {
			return false
		}

		for i, v := range t.data {
			if ot.data[i] != v {
				return false
			}
		}

		if !t.Shape().Eq(ot.Shape()) {
			return false
		}
		//TODO: MORE METADATA CHECKS!

		return true
	}
	return false
}

func (t *Tensor) Clone() *Tensor {
	retVal := new(Tensor)
	retVal.AP = t.AP.Clone()
	if t.old != nil {
		retVal.old = t.old.Clone()
	}

	newdata := make([]float64, len(t.data))
	copy(newdata, t.data)
	retVal.data = newdata
	retVal.Lock()
	return retVal
}

func (t *Tensor) borrowClone() *Tensor {
	retVal := BorrowTensor(len(t.data))
	types.ReturnAP(retVal.AP)
	retVal.AP = t.AP.Clone()

	if t.old != nil {
		retVal.old = t.old.Clone()
	}

	newdata := make([]float64, len(t.data))
	copy(newdata, t.data)
	retVal.data = newdata
	retVal.Lock()
	return retVal
}

func (t *Tensor) IsView() bool {
	return t.viewOf != nil
}

/* Misc public API */
func (t *Tensor) Data() interface{} { return t.data }
func (t *Tensor) Old() *types.AP    { return t.old }

/* Other Data types */

// rs is a struct representing a ranged slice: [start:end:step]
type rs struct {
	start, end, step int
}

// makeRS creates a ranged slice. It takes an optional step param.
func makeRS(start, end int, opts ...int) rs {
	step := 1
	if len(opts) > 0 {
		step = opts[0]
	}
	return rs{
		start: start,
		end:   end,
		step:  step,
	}
}

func (s rs) Start() int { return s.start }
func (s rs) End() int   { return s.end }
func (s rs) Step() int  { return s.step }

// ss is a single slice, representing this: [start:start+1:0]
type ss int

func (s ss) Start() int { return int(s) }
func (s ss) End() int   { return int(s) + 1 }
func (s ss) Step() int  { return 0 }
