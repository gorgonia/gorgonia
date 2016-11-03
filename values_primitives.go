package gorgonia

import "github.com/chewxy/gorgonia/tensor/types"

type Scalar interface {
	Scalar() Scalar
}

type F64 float64
type F32 float32
type I int
type I32 int32
type I64 int64
type U8 byte

func (v F64) Shape() types.Shape { return scalarShape }
func (v F32) Shape() types.Shape { return scalarShape }
func (v I) Shape() types.Shape   { return scalarShape }
func (v I64) Shape() types.Shape { return scalarShape }
func (v I32) Shape() types.Shape { return scalarShape }
func (v U8) Shape() types.Shape  { return scalarShape }

func (v F64) Size() int { return 0 }
func (v F32) Size() int { return 0 }
func (v I) Size() int   { return 0 }
func (v I64) Size() int { return 0 }
func (v I32) Size() int { return 0 }
func (v U8) Size() int  { return 0 }

func (v F64) Data() interface{} { return v }
func (v F32) Data() interface{} { return v }
func (v I) Data() interface{}   { return v }
func (v I64) Data() interface{} { return v }
func (v I32) Data() interface{} { return v }
func (v U8) Data() interface{}  { return v }

func (v F64) Scalar() Scalar { return v }
func (v F32) Scalar() Scalar { return v }
func (v I) Scalar() Scalar   { return v }
func (v I64) Scalar() Scalar { return v }
func (v I32) Scalar() Scalar { return v }
func (v U8) Scalar() Scalar  { return v }
