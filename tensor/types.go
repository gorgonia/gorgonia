package tensor

import (
	"fmt"
	"math"
	"reflect"
	"unsafe"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
)

// header is like reflect.SliceHeader. It is used to do very dirty dirty things.
type header struct {
	ptr unsafe.Pointer
	l   int
	c   int
}

// Dtype represents a data type of a Tensor. Concretely it's implemented as an embedded reflect.Type
// which allows for easy reflection operations. It also implements hm.Type, for type inference in Gorgonia
type Dtype struct {
	reflect.Type
}

// note: the Name() and String() methods are already defined in reflect.Type. Might as well use the composed methods

func (dt Dtype) Apply(hm.Subs) hm.Substitutable                { return dt }
func (dt Dtype) FreeTypeVar() hm.TypeVarSet                    { return nil }
func (dt Dtype) Normalize(k, v hm.TypeVarSet) (hm.Type, error) { return dt, nil }
func (dt Dtype) Types() hm.Types                               { return nil }
func (dt Dtype) Format(s fmt.State, c rune)                    { fmt.Fprintf(s, "%s", dt.Name()) }
func (dt Dtype) Eq(other hm.Type) bool                         { return other == dt }

func (dt Dtype) id() int {
	for i, v := range allTypes {
		if v == dt {
			return i
		}
	}
	return -1
}

// NumpyDtype returns the Numpy's Dtype equivalent. This is predominantly used in converting a Tensor to a Numpy ndarray,
// however, not all Dtypes are supported
func (dt Dtype) numpyDtype() (string, error) {
	switch dt {
	case Bool:
		return "b1", nil
	case Int:
		return fmt.Sprintf("i%d", dt.Size()), nil
	case Int8:
		return "i1", nil
	case Int16:
		return "i2", nil
	case Int32:
		return "i4", nil
	case Int64:
		return "i8", nil
	case Uint:
		return fmt.Sprintf("u%d", dt.Size()), nil
	case Uint8:
		return "u1", nil
	case Uint16:
		return "u2", nil
	case Uint32:
		return "u4", nil
	case Uint64:
		return "u8", nil
	case Float32:
		return "f4", nil
	case Float64:
		return "f8", nil
	case Complex64:
		return "c8", nil
	case Complex128:
		return "c16", nil
	default:
		return "v", errors.Errorf("Unsupported Dtype conversion")
	}
}

func fromTypeID(i int) (Dtype, error) {
	if i > len(allTypes) || i < 0 {
		return Dtype{}, errors.Errorf("Unsupported Dtype for serialization")
	}
	return allTypes[i], nil
}

func fromNumpyDtype(t string) (Dtype, error) {
	switch t {
	case "b1":
		return Bool, nil
	case "i1":
		return Int8, nil
	case "i2":
		return Int16, nil
	case "i4":
		if Int.Size() == 4 {
			return Int, nil
		}
		return Int32, nil
	case "i8":
		if Int.Size() == 8 {
			return Int, nil
		}
		return Int64, nil
	case "u1":
		return Uint8, nil
	case "u2":
		return Uint16, nil
	case "u4":
		if Uint.Size() == 4 {
			return Uint, nil
		}
		return Uint32, nil
	case "u8":
		if Uint.Size() == 8 {
			return Uint, nil
		}
		return Uint64, nil
	case "f4":
		return Float32, nil
	case "f8":
		return Float64, nil
	case "c8":
		return Complex64, nil
	case "c16":
		return Complex128, nil
	}
	return Dtype{}, errors.Errorf("Unsupported Dtype conversion from %q to Dtype", t)
}

var parameterizedKinds = [...]reflect.Kind{
	reflect.Array,
	reflect.Chan,
	reflect.Func,
	reflect.Interface,
	reflect.Map,
	reflect.Ptr,
	reflect.Slice,
	reflect.Struct,
}

func isSimpleKind(k reflect.Kind) bool {
	for _, v := range parameterizedKinds {
		if v == k {
			return true
		}
	}
	return false
}

// oh how nice it'd be if I could make them immutable
var (
	Bool       = Dtype{reflect.TypeOf(true)}
	Int        = Dtype{reflect.TypeOf(int(1))}
	Int8       = Dtype{reflect.TypeOf(int8(1))}
	Int16      = Dtype{reflect.TypeOf(int16(1))}
	Int32      = Dtype{reflect.TypeOf(int32(1))}
	Int64      = Dtype{reflect.TypeOf(int64(1))}
	Uint       = Dtype{reflect.TypeOf(uint(1))}
	Uint8      = Dtype{reflect.TypeOf(uint8(1))}
	Uint16     = Dtype{reflect.TypeOf(uint16(1))}
	Uint32     = Dtype{reflect.TypeOf(uint32(1))}
	Uint64     = Dtype{reflect.TypeOf(uint64(1))}
	Float32    = Dtype{reflect.TypeOf(float32(1))}
	Float64    = Dtype{reflect.TypeOf(float64(1))}
	Complex64  = Dtype{reflect.TypeOf(complex64(1))}
	Complex128 = Dtype{reflect.TypeOf(complex128(1))}
	String     = Dtype{reflect.TypeOf("")}

	// aliases
	Byte = Uint8

	// extras
	Uintptr       = Dtype{reflect.TypeOf(uintptr(0))}
	UnsafePointer = Dtype{reflect.TypeOf(unsafe.Pointer(&Uintptr))}
)

// allTypes for indexing
var allTypes = []Dtype{
	Bool, Int, Int8, Int16, Int32, Int64, Uint, Uint8, Uint16, Uint32, Uint64, Float32, Float64, Complex64, Complex128, String, Uintptr, UnsafePointer,
}

// specialized types indicate that there are specialized code generated for these types
var specializedTypes = []Dtype{
	Bool, Int, Int8, Int16, Int32, Int64, Uint, Uint8, Uint16, Uint32, Uint64, Float32, Float64, Complex64, Complex128, String,
}

var numberTypes = []Dtype{
	Int, Int8, Int16, Int32, Int64, Uint, Uint8, Uint16, Uint32, Uint64, Float32, Float64, Complex64, Complex128,
}

var ordTypes = []Dtype{
	Int, Int8, Int16, Int32, Int64, Uint, Uint8, Uint16, Uint32, Uint64, Float32, Float64, Complex64, Complex128, String,
}

func isSpecialized(dt Dtype) bool {
	for _, s := range specializedTypes {
		if s.Kind() == dt.Kind() {
			return true
		}
	}
	return false
}

func isNumber(dt Dtype) bool {
	for _, s := range numberTypes {
		if s.Kind() == dt.Kind() {
			return true
		}
	}
	return false
}

func isFloat(dt Dtype) bool {
	return dt.Kind() == reflect.Float64 || dt.Kind() == reflect.Float32
}

// NormOrder represents the order of the norm. Ideally, we'd only represent norms with a uint/byte.
// But there are norm types that are outside numerical types, such as nuclear norm and fobenius norm.
// So it is internally represented by a float. If Go could use NaN and Inf as consts, it would have been best,
// Instead, we use constructors. Both Nuclear and Frobenius norm types are represented as NaNs
//
// The using of NaN and Inf as "special" Norm types lead to the need for IsInf() and IsFrobenius() and IsNuclear() method
type NormOrder float64

func Norm(ord int) NormOrder   { return NormOrder(float64(ord)) }
func InfNorm() NormOrder       { return NormOrder(math.Inf(1)) }
func NegInfNorm() NormOrder    { return NormOrder(math.Inf(-1)) }
func UnorderedNorm() NormOrder { return NormOrder(math.Float64frombits(0x7ff8000000000001)) }
func FrobeniusNorm() NormOrder { return NormOrder(math.Float64frombits(0x7ff8000000000002)) }
func NuclearNorm() NormOrder   { return NormOrder(math.Float64frombits(0x7ff8000000000003)) }

// Valid() is a helper method that deterines if the norm order is valid. A valid norm order is
// one where the fraction component is 0
func (n NormOrder) Valid() bool {
	switch {
	case math.IsNaN(float64(n)):
		nb := math.Float64bits(float64(n))
		if math.Float64bits(float64(UnorderedNorm())) == nb || math.Float64bits(float64(FrobeniusNorm())) == nb || math.Float64bits(float64(NuclearNorm())) == nb {
			return true
		}
	case math.IsInf(float64(n), 0):
		return true
	default:
		if _, frac := math.Modf(float64(n)); frac == 0.0 {
			return true
		}
	}
	return false
}

// IsUnordered returns true if the NormOrder is not an ordered norm
func (n NormOrder) IsUnordered() bool {
	return math.Float64bits(float64(n)) == math.Float64bits(float64(UnorderedNorm()))
}

// IsFrobenius returns true if the NormOrder is a Frobenius norm
func (n NormOrder) IsFrobenius() bool {
	return math.Float64bits(float64(n)) == math.Float64bits(float64(FrobeniusNorm()))
}

// IsNuclear returns true if the NormOrder is a nuclear norm
func (n NormOrder) IsNuclear() bool {
	return math.Float64bits(float64(n)) == math.Float64bits(float64(NuclearNorm()))
}

func (n NormOrder) IsInf(sign int) bool {
	return math.IsInf(float64(n), sign)
}

func (n NormOrder) String() string {
	switch {
	case n.IsUnordered():
		return "Unordered"
	case n.IsFrobenius():
		return "Frobenius"
	case n.IsNuclear():
		return "Nuclear"
	case n.IsInf(1):
		return "+Inf"
	case n.IsInf(-1):
		return "-Inf"
	default:
		return fmt.Sprintf("Norm %v", float64(n))
	}
	panic("unreachable")
}

// FuncOpt are optionals for calling Tensor function.
type FuncOpt func(*OpOpt)

// WithIncr passes in a Tensor to be incremented.
func WithIncr(incr Tensor) FuncOpt {
	f := func(opt *OpOpt) {
		opt.incr = incr
	}
	return f
}

// WithReuse passes in a Tensor to be reused.
func WithReuse(reuse Tensor) FuncOpt {
	f := func(opt *OpOpt) {
		opt.reuse = reuse
	}
	return f
}

// UseSafe ensures that the operation is a safe operation (copies data, does not clobber). This is the default option for most methods and functions
func UseSafe() FuncOpt {
	f := func(opt *OpOpt) {
		opt.unsafe = false
	}
	return f
}

// UseUnsafe ensures that the operation is an unsafe operation - data will be clobbered, and operations performed inplace
func UseUnsafe() FuncOpt {
	f := func(opt *OpOpt) {
		opt.unsafe = true
	}
	return f
}

// AsSameType makes sure that the return Tensor is the same type as input Tensors.
func AsSameType() FuncOpt {
	f := func(opt *OpOpt) {
		opt.same = true
	}
	return f
}

// As makes sure that the the return Tensor is of the type specified. Currently only works for FromMat64
func As(t Dtype) FuncOpt {
	f := func(opt *OpOpt) {
		opt.t = t
	}
	return f
}
