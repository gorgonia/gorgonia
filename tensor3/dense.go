package tensor

import (
	"fmt"
	"reflect"
	"unsafe"

	"github.com/chewxy/gorgonia/tensor/types"
	"github.com/pkg/errors"
)

type Dense struct {
	*AP

	data unsafe.Pointer       // Unsafe.Pointer is required to keep the pointer of the first element of the slice, to prevent the slice from being GC'd
	hdr  *reflect.SliceHeader // we keep a separate SliceHeader because it'd be easier to cast into a slice when doing get ops
	v    interface{}          // we keep a reference to the underlying slice
	t    Dtype                // the element type

	// backup AP. When a transpose is done, the old *AP is backed up here, for easy untransposes
	old           *AP
	transposeWith []int

	// if viewOf != nil, then this *Dense is a view.
	viewOf *Dense
}

func recycledDense(dt Dtype, shape Shape, opts ...ConsOpt) (retVal *Dense) {
	retVal = recycledDenseNoFix(dt, shape, opts...)
	retVal.fix()
	if err := retVal.sanity(); err != nil {
		panic(err)
	}
	return
}

func recycledDenseNoFix(dt Dtype, shape Shape, opts ...ConsOpt) (retVal *Dense) {
	if isSimpleKind(dt.Kind()) {
		retVal = borrowDense(dt, shape.TotalSize())
	} else {
		retVal = newDense(dt, shape.TotalSize())
	}

	for _, opt := range opts {
		opt(retVal)
	}
	retVal.setShape(shape...)
	return
}

func newDense(dt Dtype, size int) *Dense {
	d := new(Dense)
	d.t = dt
	d.AP = new(AP)
	d.setShape(size)
	d.fix()
	if err := d.sanity(); err != nil {
		panic(err)
	}
	return d
}

func (t *Dense) fromSlice(x interface{}) {
	xt := reflect.TypeOf(x)
	if xt.Kind() != reflect.Slice {
		panic("Not a slice")
	}
	xt = xt.Elem()

	xv := reflect.ValueOf(x)
	ptr := xv.Pointer()
	uptr := unsafe.Pointer(ptr)

	hdr := &reflect.SliceHeader{
		Data: ptr,
		Len:  xv.Len(),
		Cap:  xv.Cap(),
	}
	t.data = uptr
	t.v = x
	t.t = Dtype{xt}
	t.hdr = hdr
}

func (t *Dense) Info() *AP    { return t.AP }
func (t *Dense) Dtype() Dtype { return t.t }
func (t *Dense) Data() interface{} {
	if t.IsScalar() {
		return t.get(0)
	}
	return t.v
}
func (t *Dense) DataSize() int {
	if t.IsScalar() {
		return 0
	}
	return t.hdr.Len
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

// // Zero zeroes a *Dense.
// func (t *Dense) Zero() {
// 	// t.data.Zero()
// }

// ScalarValue() returns the scalar value of a *Tensor,
// IF and ONLY IF it's a Tensor representation of a scalar value.
// This is required because operations like a (vec · vec) would return a scalar value.
// I didn't want to return interface{} for all the API methods, so the next best solution is to
// wrap the scalar value in a *Tensor
func (t *Dense) ScalarValue() interface{} {
	if !t.IsScalar() {
		panic(fmt.Sprintf("ScalarValue only works when the Tensor is a representation of a scalar value. The value of the tensor is %v", t))
	}

	return t.get(0)
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

		if ot.len() != t.len() {
			return false
		}

		if !t.Shape().Eq(ot.Shape()) {
			return false
		}

		if t.data != ot.data {
			return false
		}

		return true
	}
	return false
}

// Clone clones a *Dense. It creates a copy of the data, and the underlying array will be allocated
func (t *Dense) Clone() interface{} {
	retVal := recycledDense(t.t, t.Shape().Clone())
	ReturnAP(retVal.AP)
	retVal.AP = t.AP.Clone()

	if t.old != nil {
		retVal.old = t.old.Clone()
	}

	copyDense(retVal, t)
	retVal.Lock()
	return retVal
}

func (t *Dense) cap() int { return t.hdr.Cap }
func (t *Dense) len() int { return t.hdr.Len } // exactly the same as DataSize

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
		size := t.hdr.Len
		if size == 1 {
			t.SetShape() // scalar
		} else {
			t.SetShape(size) // vector
		}
	case t.data == nil && t.t != Dtype{}:
		size := t.Shape().TotalSize()
		t.makeArray(size)
	}
	t.Lock() // don't put this in a defer - if t.data == nil and t.Shape() == nil. then leave it unlocked
}

// sanity is a function that sanity checks that a tensor is correct.
func (t *Dense) sanity() error {
	if t.AP != nil && t.Shape() == nil && t.data == nil {
		return errors.New(emptyTensor)
	}

	size := t.hdr.Len
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

// Shallow clone clones the *Dense without making a copy of the underlying array
func (t *Dense) shallowClone() *Dense {
	retVal := new(Dense)
	retVal.AP = t.AP.Clone()
	retVal.data = t.data
	retVal.v = t.v
	retVal.t = t.t
	retVal.hdr = &reflect.SliceHeader{
		Data: t.hdr.Data,
		Len:  t.hdr.Len,
		Cap:  t.hdr.Cap,
	}
	return retVal
}
