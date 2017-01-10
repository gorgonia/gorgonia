package tensor

import (
	"fmt"

	"github.com/pkg/errors"

	"github.com/chewxy/gorgonia/tensor/types"
)

type Dense struct {
	*AP
	data Array
	t    Dtype

	// backup AP. When a transpose is done, the old *AP is backed up here, for easy untransposes
	old           *AP
	transposeWith []int

	// if viewOf != nil, then this *Dense is a view.
	viewOf *Dense
}

// recycledDense gets from the pool
func recycledDense(dt Dtype, shape Shape, opts ...ConsOpt) *Dense {
	var d *Dense
	if t, ok := dt.(dtype); ok {
		d = borrowDense(t, shape.TotalSize())
	} else {
		d = newDense(dt, shape.TotalSize())
	}

	for _, opt := range opts {
		opt(d)
	}
	d.setShape(shape...)

	d.fix()
	if err := d.sanity(); err != nil {
		panic(err)
	}

	return d
}

func newDense(dt Dtype, size int) *Dense {
	d := new(Dense)
	d.t = dt
	d.data = makeArray(dt, size)
	d.AP = new(AP)
	d.setShape(size)
	return d
}

func (t *Dense) Info() *AP         { return t.AP }
func (t *Dense) Dtype() Dtype      { return t.t }
func (t *Dense) Data() interface{} { return t.data.Data() }

func (t *Dense) DataSize() int {
	switch t.t {
	case Float64:
		return len(t.float64s())
	case Float32:
		return len(t.float32s())
	case Int:
		return len(t.ints())
	case Int64:
		return len(t.int64s())
	case Int32:
		return len(t.int32s())
	case Byte:
		return len(t.bytes())
	case Bool:
		return len(t.bools())
	default:
		panic("Unhandled")
	}
}

// Reshape reshapes a *Dense. If the tensors need to be materialized (either it's a view or transpose), it will be materialized before the reshape happens
func (t *Dense) Reshape(dims ...int) error {
	if t.viewOf != nil {
		return errors.Errorf(methodNYI, "Reshape", "views")
	}

	if t.old != nil {
		t.Transpose()
	}

	return t.reshape(dims...)
}

func (t *Dense) reshape(dims ...int) error {
	t.setShape(dims...)
	return t.sanity()
}

// Zero zeroes a *Dense.
func (t *Dense) Zero() {
	t.data.Zero()
}

func (t *Dense) SetAll(val interface{}) error {
	if val == 1 {
		if o, ok := t.data.(Oner); ok {
			o.One()
			return nil
		}
	}
	return t.data.Memset(val)
}

// ScalarValue() returns the scalar value of a *Tensor,
// IF and ONLY IF it's a Tensor representation of a scalar value.
// This is required because operations like a (vec · vec) would return a scalar value.
// I didn't want to return interface{} for all the API methods, so the next best solution is to
// wrap the scalar value in a *Tensor
func (t *Dense) ScalarValue() interface{} {
	if !t.IsScalar() {
		panic(fmt.Sprintf("ScalarValue only works when the Tensor is a representation of a scalar value. The value of the tensor is %v", t))
	}

	return t.data.Get(0)
}

//  IsView indicates if the Tensor is a view of another (typically from slicing)
func (t *Dense) IsView() bool {
	return t.viewOf != nil
}

// IsMaterializeable() indicates if the Tensor is materializable - if it has either gone through some transforms or slicing
func (t *Dense) IsMaterializable() bool {
	return t.viewOf != nil || t.old != nil
}

// Eq checks that two types.Tensor are equal. If the shapes are the same, but the strides are not the same, it's will still be considered the same
func (t *Dense) Eq(other interface{}) bool {
	if ot, ok := other.(*Dense); ok {
		if ot == t {
			return true
		}

		if ot.data.Len() != t.data.Len() {
			return false
		}

		if !t.Shape().Eq(ot.Shape()) {
			return false
		}

		if !t.data.Eq(ot.data) {
			return false
		}
		//TODO: MORE METADATA CHECKS!

		return true
	}
	return false
}

/* utility functions to get stuff */

func (t *Dense) float64s() []float64 { return t.data.Data().([]float64) }
func (t *Dense) float32s() []float32 { return t.data.Data().([]float32) }
func (t *Dense) ints() []int         { return t.data.Data().([]int) }
func (t *Dense) int64s() []int64     { return t.data.Data().([]int64) }
func (t *Dense) int32s() []int32     { return t.data.Data().([]int32) }
func (t *Dense) bytes() []byte       { return t.data.Data().([]byte) }
func (t *Dense) bools() []bool       { return t.data.Data().([]bool) }

/* Utility functions */

func (t *Dense) setShape(s ...int) {
	t.Unlock()
	t.SetShape(s...)
	t.Lock()
	return
}

func (t *Dense) fix() {
	if t.AP == nil {
		return
	}

	switch {
	case t.Shape() == nil && t.data != nil:
		size := t.data.Len()
		if size == 1 {
			t.SetShape() // scalar
		} else {
			t.SetShape(size) // vector
		}
	case t.data == nil && t.t != nil:
		size := t.Shape().TotalSize()
		t.data = makeArray(t.t, size)
	case t.t == nil && t.data != nil:
		var err error
		t.t, err = typeOf(t.data)
		if err != nil {
			panic(err)
		}
	}
	t.Lock() // don't put this in a defer - if t.data == nil and t.Shape() == nil. then leave it unlocked
}

// sanity is a function that sanity checks that a tensor is correct.
func (t *Dense) sanity() error {
	if t.AP != nil && t.Shape() == nil && t.data == nil {
		return errors.New(emptyTensor)
	}

	size := t.data.Len()
	expected := t.Size()
	if t.viewOf == nil && size != expected && !t.IsScalar() {
		return types.NewError(types.ShapeMismatch, "Expected backing data to have %d elements from shape %v. Got %d instead", expected, t.Shape(), size)
	}
	// TODO: sanity check for views
	return nil
}

func (t *Dense) oshape() Shape {
	if t.old != nil {
		return t.old.Shape()
	}
	return t.Shape()
}

func (t *Dense) ostrides() []int {
	if t.old != nil {
		return t.old.Strides()
	}
	return t.Strides()
}
