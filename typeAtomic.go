package gorgonia

// Dtype is data type
type Dtype byte

const (
	Float64 Dtype = iota
	Float32
	Int
	Int64
	Int32
	Byte
	Bool

	Ptr // equivalent to interface{}. Ugh Ugh Ugh
	MAXDTYPE
)

func (t Dtype) isScalar() bool { return true }
func (t Dtype) dims() int      { return 0 }
