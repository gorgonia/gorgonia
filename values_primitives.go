package gorgonia

import (
	"fmt"

	"github.com/chewxy/gorgonia/tensor/types"
	"github.com/chewxy/hm"
)

type Scalar interface {
	Value
	Scalar() Scalar
}

type F64 float64
type F32 float32
type I int
type I32 int32
type I64 int64
type U8 byte
type B bool

func (v F64) Shape() types.Shape { return scalarShape }
func (v F32) Shape() types.Shape { return scalarShape }
func (v I) Shape() types.Shape   { return scalarShape }
func (v I64) Shape() types.Shape { return scalarShape }
func (v I32) Shape() types.Shape { return scalarShape }
func (v U8) Shape() types.Shape  { return scalarShape }
func (v B) Shape() types.Shape   { return scalarShape }

func (v F64) Size() int { return 0 }
func (v F32) Size() int { return 0 }
func (v I) Size() int   { return 0 }
func (v I64) Size() int { return 0 }
func (v I32) Size() int { return 0 }
func (v U8) Size() int  { return 0 }
func (v B) Size() int   { return 0 }

func (v F64) Data() interface{} { return v }
func (v F32) Data() interface{} { return v }
func (v I) Data() interface{}   { return v }
func (v I64) Data() interface{} { return v }
func (v I32) Data() interface{} { return v }
func (v U8) Data() interface{}  { return v }
func (v B) Data() interface{}   { return v }

func (v F64) Scalar() Scalar { return v }
func (v F32) Scalar() Scalar { return v }
func (v I) Scalar() Scalar   { return v }
func (v I64) Scalar() Scalar { return v }
func (v I32) Scalar() Scalar { return v }
func (v U8) Scalar() Scalar  { return v }
func (v B) Scalar() Scalar   { return v }

func anyToScalar(any interface{}) (Scalar, hm.Type) {
	switch at := any.(type) {
	case float64:
		return F64(at), Float64
	case float32:
		return F32(at), Float32
	case int:
		return I(at), Int
	case int32:
		return I32(at), Int32
	case int64:
		return I64(at), Int64
	case byte:
		return U8(at), Byte
	case uint8:
		return U8(at), Byte
	case bool:
		return B(at), Bool
	default:
		panic(fmt.Sprintf("%v(%T) not scalar/not handled"), any, any)
	}
}
