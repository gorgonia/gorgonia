package tensor

import (
	"fmt"
	"math"
	"reflect"

	"github.com/chewxy/hm"
)

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
)

// specialized types indicate that there are specialized code generated for these types
var specializedTypes = []Dtype{
	Bool, Int, Int8, Int16, Int32, Int64, Uint, Uint8, Uint16, Uint32, Uint64, Float32, Complex64, Complex128, String,
}

var numberTypes = []Dtype{
	Int, Int8, Int16, Int32, Int64, Uint, Uint8, Uint16, Uint32, Uint64, Float32, Complex64, Complex128,
}

var ordTypes = []Dtype{
	Int, Int8, Int16, Int32, Int64, Uint, Uint8, Uint16, Uint32, Uint64, Float32, Complex64, Complex128, String,
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

// FunctionFlag are flags for calling Tensor functions. Use only with FuncOpt
type FunctionFlag byte

const (
	SafeOp FunctionFlag = iota
	UnsafeOp
	Reuse
	Incr
	AsSame
	AsTensorF64
	AsTensorF32
	AsTensorInt
)

// FuncOpt are optionals for calling Tensor function.
type FuncOpt func(*funcOpt)

// WithIncr passes in a Tensor to be incremented.
func WithIncr(incr Tensor) FuncOpt {
	f := func(opt *funcOpt) {
		opt.incr = incr
	}
	return f
}

// WithReuse passes in a Tensor to be reused.
func WithReuse(reuse Tensor) FuncOpt {
	f := func(opt *funcOpt) {
		opt.reuse = reuse
	}
	return f
}

// UseSafe ensures that the operation is a safe operation (copies data, does not clobber). This is the default option for most methods and functions
func UseSafe() FuncOpt {
	f := func(opt *funcOpt) {
		opt.unsafe = false
	}
	return f
}

// UseUnsafe ensures that the operation is an unsafe operation - data will be clobbered, and operations performed inplace
func UseUnsafe() FuncOpt {
	f := func(opt *funcOpt) {
		opt.unsafe = true
	}
	return f
}

// AsSameType makes sure that the return Tensor is the same type as input Tensors.
func AsSameType() FuncOpt {
	f := func(opt *funcOpt) {
		opt.same = true
	}
	return f
}
