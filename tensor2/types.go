package tensor

import (
	"fmt"
	"math"

	"github.com/chewxy/hm"
)

// Dtype represents a data type of a Tensor.
type Dtype interface {
	hm.Type
	ZeroValue() interface{}
}

//go:generate stringer -type=dtype
type dtype byte

const (
	Float64 dtype = iota
	Float32
	Int
	Int64
	Int32
	Byte
	Bool

	MAXDTYPE
)

func (t dtype) Name() string                                  { return t.String() }
func (t dtype) Apply(hm.Subs) hm.Substitutable                { return t }
func (t dtype) FreeTypeVar() hm.TypeVarSet                    { return nil }
func (t dtype) Normalize(k, v hm.TypeVarSet) (hm.Type, error) { return t, nil }
func (t dtype) Types() hm.Types                               { return nil }
func (t dtype) Format(state fmt.State, c rune)                { state.Write([]byte(t.String())) }
func (t dtype) Eq(other hm.Type) bool                         { return t == other }

const _dtype_name = "Float64Float32IntInt64Int32ByteBoolMAXDTYPE"

var _dtype_index = [...]uint8{0, 7, 14, 17, 22, 27, 31, 35}

func (t dtype) String() string {
	if t >= dtype(len(_dtype_index)-1) {
		return fmt.Sprintf("dtype(%d)", t)
	}
	return _dtype_name[_dtype_index[t]:_dtype_index[t+1]]
}

func (t dtype) ZeroValue() interface{} {
	switch t {
	case Float64:
		return float64(0)
	case Float32:
		return float32(0)
	case Int:
		return int(0)
	case Int64:
		return int64(0)
	case Int32:
		return int32(0)
	case Byte:
		return byte(0)
	case Bool:
		return false
	}
	panic("Unreachable")
}

// MAXDTYPE is the amount of Dtypes supported internally by the tensor package

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

func (n NormOrder) IsUnordered() bool {
	return math.Float64bits(float64(n)) == math.Float64bits(float64(UnorderedNorm()))
}

func (n NormOrder) IsFrobenius() bool {
	return math.Float64bits(float64(n)) == math.Float64bits(float64(FrobeniusNorm()))
}

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
type FuncOpt func() (FunctionFlag, interface{})
