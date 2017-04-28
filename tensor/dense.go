package tensor

import (
	"fmt"
	"reflect"
	"unsafe"

	"github.com/pkg/errors"
)

type denseFlag byte

const (
	manuallyManagedMem denseFlag = 1 << iota
)

// Dense represents a dense tensor - this is the most common form of tensors. It can be used to represent vectors, matrices.. etc
type Dense struct {
	*AP

	flag denseFlag

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

// NewDense creates a new *Dense. It tries its best to get from the tensor pool.
func NewDense(dt Dtype, shape Shape, opts ...ConsOpt) *Dense {
	return recycledDense(dt, shape, opts...)
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
	size := shape.TotalSize()
	if shape.IsScalar() {
		size = 1
	}
	if isSimpleKind(dt.Kind()) {
		retVal = borrowDense(dt, size)
	} else {
		retVal = newDense(dt, size)
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

// Info returns the accesspattern which explains how the data in the underlying array is accessed. This is mostly used for debugging.
func (t *Dense) Info() *AP { return t.AP }

// Dtype returns the data type of the *Dense tensor.
func (t *Dense) Dtype() Dtype { return t.t }

// Data returns the underlying array. If the *Dense represents a scalar value, the scalar value is returned instead
func (t *Dense) Data() interface{} {
	if t.IsScalar() {
		return t.Get(0)
	}
	return t.v
}

// DataSize returns the size of the array. Typically t.DataSize() == t.Shape().TotalSize()
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

// ScalarValue returns the scalar value of a *Tensor,
// IF and ONLY IF it's a Tensor representation of a scalar value.
// This is required because operations like a (vec · vec) would return a scalar value.
// I didn't want to return interface{} for all the API methods, so the next best solution is to
// wrap the scalar value in a *Tensor
func (t *Dense) ScalarValue() interface{} {
	if !t.IsScalar() {
		panic(fmt.Sprintf("ScalarValue only works when the Tensor is a representation of a scalar value. The value of the tensor is %v", t))
	}

	return t.Get(0)
}

//  IsView indicates if the Tensor is a view of another (typically from slicing)
func (t *Dense) IsView() bool {
	return t.viewOf != nil
}

// IsMaterializeable() indicates if the Tensor is materializable - if it has either gone through some transforms or slicing
func (t *Dense) IsMaterializable() bool {
	return t.viewOf != nil || t.old != nil
}

// IsManuallyManaged returns true if the memory associated with this *Dense is manually managed (by the user)
func (t *Dense) IsManuallyManaged() bool {
	return (t.flag>>manuallyManagedMem)&denseFlag(1) == 1
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
	retVal.lock()
	return retVal
}

// Uintptr returns the pointer of the first value of the slab
func (t *Dense) Uintptr() uintptr {
	return uintptr(t.data)
}

// MemSize returns how big the slice is in bytes
func (t *Dense) MemSize() uintptr {
	return uintptr(t.hdr.Len) * t.t.Size()
}

// Pointer returns the pointer of the first value of the slab, as an unsafe.Pointer
func (t *Dense) Pointer() unsafe.Pointer {
	return t.data
}

// Private methods

func (t *Dense) cap() int { return t.hdr.Cap }
func (t *Dense) len() int { return t.hdr.Len } // exactly the same as DataSize

func (t *Dense) setShape(s ...int) {
	t.unlock()
	t.SetShape(s...)
	t.lock()
	return
}

func (t *Dense) fix() {
	if t.AP == nil {
		return
	}

	switch {
	case t.IsScalar() && t.data == nil:
		t.makeArray(1)
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
	t.lock() // don't put this in a defer - if t.data == nil and t.Shape() == nil. then leave it unlocked
}

// sanity is a function that sanity checks that a tensor is correct.
func (t *Dense) sanity() error {
	if t.AP != nil && t.Shape() == nil && t.data == nil {
		return errors.New(emptyTensor)
	}

	size := t.hdr.Len
	expected := t.Size()
	if t.viewOf == nil && size != expected && !t.IsScalar() {
		return errors.Errorf(shapeMismatch, t.Shape(), size)
	}
	// TODO: sanity check for views
	return nil
}

// oshape returns the original shape
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
