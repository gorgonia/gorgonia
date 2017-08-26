package tensor

// DataOrder is a flag that indicates the order of data. The default DataOrder (0)
// is what this package uses by default.
type DataOrder byte

const (
	// ColMajor indicates that the data is stored in a col-major way.
	// A data can only be stored in either ColMajor(1) or RowMajor(0).
	// The way the DataOrder was designed causes the default to be RowMajor
	ColMajor DataOrder = 1 << iota
	// NonContiguous indicates that the data is not contiguous.
	// A data can either be Contiguous (0) or NonContiguous (2).
	// The way DataOrder was designed causes the default to be Contiguous.
	NonContiguous
)

// MakeDataOrder makes a data order. Typical examples:
//		MakeDataOrder(DataOrder(0))            // Row Major, contiguous
//		MakeDataOrder(NonContiguous            // Row Major, non-contiguous
// 		MakeDataOrder(ColMajor)                // Col Major, contiguous
//		MakeDataOrder(ColMajor, NonContiguous) // what it says on the tin
func MakeDataOrder(fs ...DataOrder) (retVal DataOrder) {
	if len(fs) == 1 {
		return fs[0]
	}
	for _, f := range fs {
		retVal |= f
	}
	return
}

func (f DataOrder) isColMajor() bool          { return (f & ColMajor) != 0 }
func (f DataOrder) isRowMajor() bool          { return !f.isColMajor() }
func (f DataOrder) isContiguous() bool        { return !f.isNotContiguous() }
func (f DataOrder) isNotContiguous() bool     { return (f & NonContiguous) != 0 }
func (f DataOrder) toggleColMajor() DataOrder { return f ^ (ColMajor) }
func (f DataOrder) hasSameOrder(other DataOrder) bool {
	return (f.isColMajor() && other.isColMajor()) || (f.isRowMajor() && other.isRowMajor())
}

// Triangle is a flag representing the "triangle"ness of a matrix
type Triangle byte

const (
	NotTriangle Triangle = iota
	Upper
	Lower
	Symmetric
)

// MemoryFlag is a flag representing the use possibilities of Memory
type MemoryFlag byte

const (
	// NativelyInaccessible indicates that the data in the memory cannot be accessed by Go code.
	NativelyInaccessible MemoryFlag = 1 << iota
	// ManuallyManaged indicates that the memory is managed by something else. Any Tensor with
	// manually managed memory will not be returned to the pool.
	ManuallyManaged
)

func MakeMemoryFlag(fs ...MemoryFlag) (retVal MemoryFlag) {
	if len(fs) == 1 {
		return fs[0]
	}

	for _, f := range fs {
		retVal |= f
	}
	return
}

func (f MemoryFlag) nativelyAccessible() bool { return !((f & NativelyInaccessible) != 0) }
func (f MemoryFlag) manuallyManaged() bool    { return (f & ManuallyManaged) != 0 }

// OpOpt are the options used to call ops
type OpOpt struct {
	reuse  Tensor
	incr   Tensor
	unsafe bool
	same   bool
	t      Dtype
}

// ParseFuncOpts parses a list of FuncOpt into a single unified method call structure.
func ParseFuncOpts(opts ...FuncOpt) *OpOpt {
	retVal := new(OpOpt)
	for _, opt := range opts {
		opt(retVal)
	}
	return retVal
}

// Incr returns the tensor to be incremented in the call. Can be nil.
func (fo *OpOpt) Incr() Tensor { return fo.incr }

// Reuse returns the tensor to be reused in the call. Can be nil.
func (fo *OpOpt) Reuse() Tensor { return fo.reuse }

// IncReuse returns whether a reuse tensor is to be used as the incr Tensor
func (fo *OpOpt) IncrReuse() (Tensor, bool) {
	if fo.incr != nil {
		return fo.incr, true
	}
	return fo.reuse, false
}

// Safe signals if the op is to be done safely
func (fo *OpOpt) Safe() bool { return !fo.unsafe }

// Same signals if the op is to return the same type as its inputs
func (fo *OpOpt) Same() bool { return fo.same }

// As returns the dtype of the return value of the method call.
// For example:
//		a.Lt(b, As(Bool))
// indicates that the result of the `Lt()` should be a Tensor of Bool.
//
// Another example:
//		a.Add(b, As(Int))
// indicates that the result of `Add()` should be converted to a Tensor of Int.
// Note that this function is not yet supported in most operations.
func (fo *OpOpt) As() Dtype { return fo.t }
