// +build !cuda

package values

import "gorgonia.org/tensor"

// F64 represents a float64 value.
type F64 float64

// F32 represents a float32 value.
type F32 float32

// I represents a int value.
type I int

// I64 represents a int64 value.
type I64 int64

// I32 represents a int32 value.
type I32 int32

// U8 represents a byte value.
type U8 byte

// B represents a bool value.
type B bool

func NewF64(v float64) *F64 { r := F64(v); return &r }
func NewF32(v float32) *F32 { r := F32(v); return &r }
func NewI(v int) *I         { r := I(v); return &r }
func NewI64(v int64) *I64   { r := I64(v); return &r }
func NewI32(v int32) *I32   { r := I32(v); return &r }
func NewU8(v byte) *U8      { r := U8(v); return &r }
func NewB(v bool) *B        { r := B(v); return &r }

// Engine returns nil for all scalar Values
func (v *F64) Engine() tensor.Engine { return nil }

// Engine returns nil for all scalar Values
func (v *F32) Engine() tensor.Engine { return nil }

// Engine returns nil for all scalar Values
func (v *I) Engine() tensor.Engine { return nil }

// Engine returns nil for all scalar Values
func (v *I64) Engine() tensor.Engine { return nil }

// Engine returns nil for all scalar Values
func (v *I32) Engine() tensor.Engine { return nil }

// Engine returns nil for all scalar Values
func (v *U8) Engine() tensor.Engine { return nil }

// Engine returns nil for all scalar Values
func (v *B) Engine() tensor.Engine { return nil }
