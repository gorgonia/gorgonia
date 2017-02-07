package tensor

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/chewxy/math32"
)

/*
GENERATED FILE. DO NOT EDIT
*/

/* Test data */

var basicDenseI = New(WithShape(2, 3, 4, 5, 2), WithBacking([]int{3, 4, 2, 4, 3, 8, 3, 9, 7, 4, 3, 0, 3, 9, 9, 0, 6, 7, 3, 9, 4, 8, 5, 1, 1, 9, 4, 0, 4, 1, 6, 6, 4, 9, 3, 8, 1, 7, 0, 7, 4, 0, 6, 8, 2, 8, 0, 6, 1, 6, 2, 3, 7, 5, 7, 3, 0, 8, 6, 5, 6, 9, 7, 5, 6, 8, 7, 9, 5, 0, 8, 1, 4, 0, 6, 6, 3, 3, 8, 1, 1, 3, 2, 5, 9, 0, 4, 5, 3, 1, 9, 1, 9, 3, 9, 3, 3, 4, 5, 9, 4, 2, 2, 7, 9, 8, 1, 6, 9, 4, 4, 1, 8, 9, 8, 0, 9, 9, 4, 6, 7, 5, 9, 9, 4, 8, 5, 8, 2, 4, 8, 2, 7, 2, 8, 7, 2, 3, 7, 0, 9, 9, 8, 9, 2, 1, 7, 0, 7, 9, 0, 2, 4, 8, 7, 9, 6, 8, 3, 3, 7, 2, 9, 2, 8, 2, 3, 6, 0, 8, 7, 7, 0, 9, 0, 9, 3, 2, 6, 9, 5, 8, 6, 9, 5, 6, 1, 8, 7, 8, 1, 9, 9, 3, 7, 7, 6, 8, 2, 1, 1, 5, 1, 4, 0, 5, 1, 7, 9, 5, 6, 6, 8, 7, 5, 1, 3, 4, 0, 1, 8, 0, 2, 6, 9, 1, 4, 8, 0, 5, 6, 2, 9, 4, 4, 2, 4, 4, 4, 3}))
var basicDenseI8 = New(WithShape(2, 3, 4, 5, 2), WithBacking([]int8{3, 4, 2, 4, 3, 8, 3, 9, 7, 4, 3, 0, 3, 9, 9, 0, 6, 7, 3, 9, 4, 8, 5, 1, 1, 9, 4, 0, 4, 1, 6, 6, 4, 9, 3, 8, 1, 7, 0, 7, 4, 0, 6, 8, 2, 8, 0, 6, 1, 6, 2, 3, 7, 5, 7, 3, 0, 8, 6, 5, 6, 9, 7, 5, 6, 8, 7, 9, 5, 0, 8, 1, 4, 0, 6, 6, 3, 3, 8, 1, 1, 3, 2, 5, 9, 0, 4, 5, 3, 1, 9, 1, 9, 3, 9, 3, 3, 4, 5, 9, 4, 2, 2, 7, 9, 8, 1, 6, 9, 4, 4, 1, 8, 9, 8, 0, 9, 9, 4, 6, 7, 5, 9, 9, 4, 8, 5, 8, 2, 4, 8, 2, 7, 2, 8, 7, 2, 3, 7, 0, 9, 9, 8, 9, 2, 1, 7, 0, 7, 9, 0, 2, 4, 8, 7, 9, 6, 8, 3, 3, 7, 2, 9, 2, 8, 2, 3, 6, 0, 8, 7, 7, 0, 9, 0, 9, 3, 2, 6, 9, 5, 8, 6, 9, 5, 6, 1, 8, 7, 8, 1, 9, 9, 3, 7, 7, 6, 8, 2, 1, 1, 5, 1, 4, 0, 5, 1, 7, 9, 5, 6, 6, 8, 7, 5, 1, 3, 4, 0, 1, 8, 0, 2, 6, 9, 1, 4, 8, 0, 5, 6, 2, 9, 4, 4, 2, 4, 4, 4, 3}))
var basicDenseI16 = New(WithShape(2, 3, 4, 5, 2), WithBacking([]int16{3, 4, 2, 4, 3, 8, 3, 9, 7, 4, 3, 0, 3, 9, 9, 0, 6, 7, 3, 9, 4, 8, 5, 1, 1, 9, 4, 0, 4, 1, 6, 6, 4, 9, 3, 8, 1, 7, 0, 7, 4, 0, 6, 8, 2, 8, 0, 6, 1, 6, 2, 3, 7, 5, 7, 3, 0, 8, 6, 5, 6, 9, 7, 5, 6, 8, 7, 9, 5, 0, 8, 1, 4, 0, 6, 6, 3, 3, 8, 1, 1, 3, 2, 5, 9, 0, 4, 5, 3, 1, 9, 1, 9, 3, 9, 3, 3, 4, 5, 9, 4, 2, 2, 7, 9, 8, 1, 6, 9, 4, 4, 1, 8, 9, 8, 0, 9, 9, 4, 6, 7, 5, 9, 9, 4, 8, 5, 8, 2, 4, 8, 2, 7, 2, 8, 7, 2, 3, 7, 0, 9, 9, 8, 9, 2, 1, 7, 0, 7, 9, 0, 2, 4, 8, 7, 9, 6, 8, 3, 3, 7, 2, 9, 2, 8, 2, 3, 6, 0, 8, 7, 7, 0, 9, 0, 9, 3, 2, 6, 9, 5, 8, 6, 9, 5, 6, 1, 8, 7, 8, 1, 9, 9, 3, 7, 7, 6, 8, 2, 1, 1, 5, 1, 4, 0, 5, 1, 7, 9, 5, 6, 6, 8, 7, 5, 1, 3, 4, 0, 1, 8, 0, 2, 6, 9, 1, 4, 8, 0, 5, 6, 2, 9, 4, 4, 2, 4, 4, 4, 3}))
var basicDenseI32 = New(WithShape(2, 3, 4, 5, 2), WithBacking([]int32{3, 4, 2, 4, 3, 8, 3, 9, 7, 4, 3, 0, 3, 9, 9, 0, 6, 7, 3, 9, 4, 8, 5, 1, 1, 9, 4, 0, 4, 1, 6, 6, 4, 9, 3, 8, 1, 7, 0, 7, 4, 0, 6, 8, 2, 8, 0, 6, 1, 6, 2, 3, 7, 5, 7, 3, 0, 8, 6, 5, 6, 9, 7, 5, 6, 8, 7, 9, 5, 0, 8, 1, 4, 0, 6, 6, 3, 3, 8, 1, 1, 3, 2, 5, 9, 0, 4, 5, 3, 1, 9, 1, 9, 3, 9, 3, 3, 4, 5, 9, 4, 2, 2, 7, 9, 8, 1, 6, 9, 4, 4, 1, 8, 9, 8, 0, 9, 9, 4, 6, 7, 5, 9, 9, 4, 8, 5, 8, 2, 4, 8, 2, 7, 2, 8, 7, 2, 3, 7, 0, 9, 9, 8, 9, 2, 1, 7, 0, 7, 9, 0, 2, 4, 8, 7, 9, 6, 8, 3, 3, 7, 2, 9, 2, 8, 2, 3, 6, 0, 8, 7, 7, 0, 9, 0, 9, 3, 2, 6, 9, 5, 8, 6, 9, 5, 6, 1, 8, 7, 8, 1, 9, 9, 3, 7, 7, 6, 8, 2, 1, 1, 5, 1, 4, 0, 5, 1, 7, 9, 5, 6, 6, 8, 7, 5, 1, 3, 4, 0, 1, 8, 0, 2, 6, 9, 1, 4, 8, 0, 5, 6, 2, 9, 4, 4, 2, 4, 4, 4, 3}))
var basicDenseI64 = New(WithShape(2, 3, 4, 5, 2), WithBacking([]int64{3, 4, 2, 4, 3, 8, 3, 9, 7, 4, 3, 0, 3, 9, 9, 0, 6, 7, 3, 9, 4, 8, 5, 1, 1, 9, 4, 0, 4, 1, 6, 6, 4, 9, 3, 8, 1, 7, 0, 7, 4, 0, 6, 8, 2, 8, 0, 6, 1, 6, 2, 3, 7, 5, 7, 3, 0, 8, 6, 5, 6, 9, 7, 5, 6, 8, 7, 9, 5, 0, 8, 1, 4, 0, 6, 6, 3, 3, 8, 1, 1, 3, 2, 5, 9, 0, 4, 5, 3, 1, 9, 1, 9, 3, 9, 3, 3, 4, 5, 9, 4, 2, 2, 7, 9, 8, 1, 6, 9, 4, 4, 1, 8, 9, 8, 0, 9, 9, 4, 6, 7, 5, 9, 9, 4, 8, 5, 8, 2, 4, 8, 2, 7, 2, 8, 7, 2, 3, 7, 0, 9, 9, 8, 9, 2, 1, 7, 0, 7, 9, 0, 2, 4, 8, 7, 9, 6, 8, 3, 3, 7, 2, 9, 2, 8, 2, 3, 6, 0, 8, 7, 7, 0, 9, 0, 9, 3, 2, 6, 9, 5, 8, 6, 9, 5, 6, 1, 8, 7, 8, 1, 9, 9, 3, 7, 7, 6, 8, 2, 1, 1, 5, 1, 4, 0, 5, 1, 7, 9, 5, 6, 6, 8, 7, 5, 1, 3, 4, 0, 1, 8, 0, 2, 6, 9, 1, 4, 8, 0, 5, 6, 2, 9, 4, 4, 2, 4, 4, 4, 3}))
var basicDenseU = New(WithShape(2, 3, 4, 5, 2), WithBacking([]uint{3, 4, 2, 4, 3, 8, 3, 9, 7, 4, 3, 0, 3, 9, 9, 0, 6, 7, 3, 9, 4, 8, 5, 1, 1, 9, 4, 0, 4, 1, 6, 6, 4, 9, 3, 8, 1, 7, 0, 7, 4, 0, 6, 8, 2, 8, 0, 6, 1, 6, 2, 3, 7, 5, 7, 3, 0, 8, 6, 5, 6, 9, 7, 5, 6, 8, 7, 9, 5, 0, 8, 1, 4, 0, 6, 6, 3, 3, 8, 1, 1, 3, 2, 5, 9, 0, 4, 5, 3, 1, 9, 1, 9, 3, 9, 3, 3, 4, 5, 9, 4, 2, 2, 7, 9, 8, 1, 6, 9, 4, 4, 1, 8, 9, 8, 0, 9, 9, 4, 6, 7, 5, 9, 9, 4, 8, 5, 8, 2, 4, 8, 2, 7, 2, 8, 7, 2, 3, 7, 0, 9, 9, 8, 9, 2, 1, 7, 0, 7, 9, 0, 2, 4, 8, 7, 9, 6, 8, 3, 3, 7, 2, 9, 2, 8, 2, 3, 6, 0, 8, 7, 7, 0, 9, 0, 9, 3, 2, 6, 9, 5, 8, 6, 9, 5, 6, 1, 8, 7, 8, 1, 9, 9, 3, 7, 7, 6, 8, 2, 1, 1, 5, 1, 4, 0, 5, 1, 7, 9, 5, 6, 6, 8, 7, 5, 1, 3, 4, 0, 1, 8, 0, 2, 6, 9, 1, 4, 8, 0, 5, 6, 2, 9, 4, 4, 2, 4, 4, 4, 3}))
var basicDenseU8 = New(WithShape(2, 3, 4, 5, 2), WithBacking([]uint8{3, 4, 2, 4, 3, 8, 3, 9, 7, 4, 3, 0, 3, 9, 9, 0, 6, 7, 3, 9, 4, 8, 5, 1, 1, 9, 4, 0, 4, 1, 6, 6, 4, 9, 3, 8, 1, 7, 0, 7, 4, 0, 6, 8, 2, 8, 0, 6, 1, 6, 2, 3, 7, 5, 7, 3, 0, 8, 6, 5, 6, 9, 7, 5, 6, 8, 7, 9, 5, 0, 8, 1, 4, 0, 6, 6, 3, 3, 8, 1, 1, 3, 2, 5, 9, 0, 4, 5, 3, 1, 9, 1, 9, 3, 9, 3, 3, 4, 5, 9, 4, 2, 2, 7, 9, 8, 1, 6, 9, 4, 4, 1, 8, 9, 8, 0, 9, 9, 4, 6, 7, 5, 9, 9, 4, 8, 5, 8, 2, 4, 8, 2, 7, 2, 8, 7, 2, 3, 7, 0, 9, 9, 8, 9, 2, 1, 7, 0, 7, 9, 0, 2, 4, 8, 7, 9, 6, 8, 3, 3, 7, 2, 9, 2, 8, 2, 3, 6, 0, 8, 7, 7, 0, 9, 0, 9, 3, 2, 6, 9, 5, 8, 6, 9, 5, 6, 1, 8, 7, 8, 1, 9, 9, 3, 7, 7, 6, 8, 2, 1, 1, 5, 1, 4, 0, 5, 1, 7, 9, 5, 6, 6, 8, 7, 5, 1, 3, 4, 0, 1, 8, 0, 2, 6, 9, 1, 4, 8, 0, 5, 6, 2, 9, 4, 4, 2, 4, 4, 4, 3}))
var basicDenseU16 = New(WithShape(2, 3, 4, 5, 2), WithBacking([]uint16{3, 4, 2, 4, 3, 8, 3, 9, 7, 4, 3, 0, 3, 9, 9, 0, 6, 7, 3, 9, 4, 8, 5, 1, 1, 9, 4, 0, 4, 1, 6, 6, 4, 9, 3, 8, 1, 7, 0, 7, 4, 0, 6, 8, 2, 8, 0, 6, 1, 6, 2, 3, 7, 5, 7, 3, 0, 8, 6, 5, 6, 9, 7, 5, 6, 8, 7, 9, 5, 0, 8, 1, 4, 0, 6, 6, 3, 3, 8, 1, 1, 3, 2, 5, 9, 0, 4, 5, 3, 1, 9, 1, 9, 3, 9, 3, 3, 4, 5, 9, 4, 2, 2, 7, 9, 8, 1, 6, 9, 4, 4, 1, 8, 9, 8, 0, 9, 9, 4, 6, 7, 5, 9, 9, 4, 8, 5, 8, 2, 4, 8, 2, 7, 2, 8, 7, 2, 3, 7, 0, 9, 9, 8, 9, 2, 1, 7, 0, 7, 9, 0, 2, 4, 8, 7, 9, 6, 8, 3, 3, 7, 2, 9, 2, 8, 2, 3, 6, 0, 8, 7, 7, 0, 9, 0, 9, 3, 2, 6, 9, 5, 8, 6, 9, 5, 6, 1, 8, 7, 8, 1, 9, 9, 3, 7, 7, 6, 8, 2, 1, 1, 5, 1, 4, 0, 5, 1, 7, 9, 5, 6, 6, 8, 7, 5, 1, 3, 4, 0, 1, 8, 0, 2, 6, 9, 1, 4, 8, 0, 5, 6, 2, 9, 4, 4, 2, 4, 4, 4, 3}))
var basicDenseU32 = New(WithShape(2, 3, 4, 5, 2), WithBacking([]uint32{3, 4, 2, 4, 3, 8, 3, 9, 7, 4, 3, 0, 3, 9, 9, 0, 6, 7, 3, 9, 4, 8, 5, 1, 1, 9, 4, 0, 4, 1, 6, 6, 4, 9, 3, 8, 1, 7, 0, 7, 4, 0, 6, 8, 2, 8, 0, 6, 1, 6, 2, 3, 7, 5, 7, 3, 0, 8, 6, 5, 6, 9, 7, 5, 6, 8, 7, 9, 5, 0, 8, 1, 4, 0, 6, 6, 3, 3, 8, 1, 1, 3, 2, 5, 9, 0, 4, 5, 3, 1, 9, 1, 9, 3, 9, 3, 3, 4, 5, 9, 4, 2, 2, 7, 9, 8, 1, 6, 9, 4, 4, 1, 8, 9, 8, 0, 9, 9, 4, 6, 7, 5, 9, 9, 4, 8, 5, 8, 2, 4, 8, 2, 7, 2, 8, 7, 2, 3, 7, 0, 9, 9, 8, 9, 2, 1, 7, 0, 7, 9, 0, 2, 4, 8, 7, 9, 6, 8, 3, 3, 7, 2, 9, 2, 8, 2, 3, 6, 0, 8, 7, 7, 0, 9, 0, 9, 3, 2, 6, 9, 5, 8, 6, 9, 5, 6, 1, 8, 7, 8, 1, 9, 9, 3, 7, 7, 6, 8, 2, 1, 1, 5, 1, 4, 0, 5, 1, 7, 9, 5, 6, 6, 8, 7, 5, 1, 3, 4, 0, 1, 8, 0, 2, 6, 9, 1, 4, 8, 0, 5, 6, 2, 9, 4, 4, 2, 4, 4, 4, 3}))
var basicDenseU64 = New(WithShape(2, 3, 4, 5, 2), WithBacking([]uint64{3, 4, 2, 4, 3, 8, 3, 9, 7, 4, 3, 0, 3, 9, 9, 0, 6, 7, 3, 9, 4, 8, 5, 1, 1, 9, 4, 0, 4, 1, 6, 6, 4, 9, 3, 8, 1, 7, 0, 7, 4, 0, 6, 8, 2, 8, 0, 6, 1, 6, 2, 3, 7, 5, 7, 3, 0, 8, 6, 5, 6, 9, 7, 5, 6, 8, 7, 9, 5, 0, 8, 1, 4, 0, 6, 6, 3, 3, 8, 1, 1, 3, 2, 5, 9, 0, 4, 5, 3, 1, 9, 1, 9, 3, 9, 3, 3, 4, 5, 9, 4, 2, 2, 7, 9, 8, 1, 6, 9, 4, 4, 1, 8, 9, 8, 0, 9, 9, 4, 6, 7, 5, 9, 9, 4, 8, 5, 8, 2, 4, 8, 2, 7, 2, 8, 7, 2, 3, 7, 0, 9, 9, 8, 9, 2, 1, 7, 0, 7, 9, 0, 2, 4, 8, 7, 9, 6, 8, 3, 3, 7, 2, 9, 2, 8, 2, 3, 6, 0, 8, 7, 7, 0, 9, 0, 9, 3, 2, 6, 9, 5, 8, 6, 9, 5, 6, 1, 8, 7, 8, 1, 9, 9, 3, 7, 7, 6, 8, 2, 1, 1, 5, 1, 4, 0, 5, 1, 7, 9, 5, 6, 6, 8, 7, 5, 1, 3, 4, 0, 1, 8, 0, 2, 6, 9, 1, 4, 8, 0, 5, 6, 2, 9, 4, 4, 2, 4, 4, 4, 3}))
var basicDenseF32 = New(WithShape(2, 3, 4, 5, 2), WithBacking([]float32{3, 4, 2, 4, 3, 8, 3, 9, 7, 4, 3, 0, 3, 9, 9, 0, 6, 7, 3, 9, 4, 8, 5, 1, 1, 9, 4, 0, 4, 1, 6, 6, 4, 9, 3, 8, 1, 7, 0, 7, 4, 0, 6, 8, 2, 8, 0, 6, 1, 6, 2, 3, 7, 5, 7, 3, 0, 8, 6, 5, 6, 9, 7, 5, 6, 8, 7, 9, 5, 0, 8, 1, 4, 0, 6, 6, 3, 3, 8, 1, 1, 3, 2, 5, 9, 0, 4, 5, 3, 1, 9, 1, 9, 3, 9, 3, 3, 4, 5, 9, 4, 2, 2, 7, 9, 8, 1, 6, 9, 4, 4, 1, 8, 9, 8, 0, 9, 9, 4, 6, 7, 5, 9, 9, 4, 8, 5, 8, 2, 4, 8, 2, 7, 2, 8, 7, 2, 3, 7, 0, 9, 9, 8, 9, 2, 1, 7, 0, 7, 9, 0, 2, 4, 8, 7, 9, 6, 8, 3, 3, 7, 2, 9, 2, 8, 2, 3, 6, 0, 8, 7, 7, 0, 9, 0, 9, 3, 2, 6, 9, 5, 8, 6, 9, 5, 6, 1, 8, 7, 8, 1, 9, 9, 3, 7, 7, 6, 8, 2, 1, 1, 5, 1, 4, 0, 5, 1, 7, 9, 5, 6, 6, 8, 7, 5, 1, 3, 4, 0, 1, 8, 0, 2, 6, 9, 1, 4, 8, 0, 5, 6, 2, 9, 4, 4, 2, 4, 4, 4, 3}))
var basicDenseF64 = New(WithShape(2, 3, 4, 5, 2), WithBacking([]float64{3, 4, 2, 4, 3, 8, 3, 9, 7, 4, 3, 0, 3, 9, 9, 0, 6, 7, 3, 9, 4, 8, 5, 1, 1, 9, 4, 0, 4, 1, 6, 6, 4, 9, 3, 8, 1, 7, 0, 7, 4, 0, 6, 8, 2, 8, 0, 6, 1, 6, 2, 3, 7, 5, 7, 3, 0, 8, 6, 5, 6, 9, 7, 5, 6, 8, 7, 9, 5, 0, 8, 1, 4, 0, 6, 6, 3, 3, 8, 1, 1, 3, 2, 5, 9, 0, 4, 5, 3, 1, 9, 1, 9, 3, 9, 3, 3, 4, 5, 9, 4, 2, 2, 7, 9, 8, 1, 6, 9, 4, 4, 1, 8, 9, 8, 0, 9, 9, 4, 6, 7, 5, 9, 9, 4, 8, 5, 8, 2, 4, 8, 2, 7, 2, 8, 7, 2, 3, 7, 0, 9, 9, 8, 9, 2, 1, 7, 0, 7, 9, 0, 2, 4, 8, 7, 9, 6, 8, 3, 3, 7, 2, 9, 2, 8, 2, 3, 6, 0, 8, 7, 7, 0, 9, 0, 9, 3, 2, 6, 9, 5, 8, 6, 9, 5, 6, 1, 8, 7, 8, 1, 9, 9, 3, 7, 7, 6, 8, 2, 1, 1, 5, 1, 4, 0, 5, 1, 7, 9, 5, 6, 6, 8, 7, 5, 1, 3, 4, 0, 1, 8, 0, 2, 6, 9, 1, 4, 8, 0, 5, 6, 2, 9, 4, 4, 2, 4, 4, 4, 3}))

var argmaxCorrect = []struct {
	shape Shape
	data  []int
}{
	{Shape{3, 4, 5, 2}, []int{
		1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1,
		1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0,
		1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1,
		1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1,
		0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0,
		1, 0, 0, 0, 0,
	}},
	{Shape{2, 4, 5, 2}, []int{
		1, 0, 1, 1, 2, 0, 2, 0, 0, 1, 2, 1, 2, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1,
		2, 2, 0, 1, 1, 2, 2, 1, 0, 2, 0, 2, 0, 2, 2, 1, 0, 0, 0, 0, 0, 1, 0,
		0, 0, 2, 1, 0, 1, 2, 1, 0, 1, 1, 2, 0, 1, 0, 0, 0, 0, 2, 1, 0, 1, 0,
		0, 2, 1, 1, 0, 0, 0, 0, 0, 2, 0,
	}},
	{Shape{2, 3, 5, 2}, []int{
		3, 2, 2, 1, 1, 2, 1, 0, 0, 1, 3, 2, 1, 0, 1, 0, 2, 2, 3, 0, 1, 0, 1,
		3, 0, 2, 3, 3, 2, 1, 2, 2, 0, 0, 1, 3, 2, 0, 1, 2, 0, 3, 0, 1, 0, 1,
		3, 2, 2, 1, 2, 1, 3, 1, 2, 0, 2, 2, 0, 0,
	}},
	{Shape{2, 3, 4, 2}, []int{
		4, 3, 2, 1, 1, 2, 0, 1, 1, 1, 1, 3, 1, 0, 0, 2, 2, 1, 0, 4, 2, 2, 3,
		1, 1, 1, 0, 2, 0, 0, 2, 2, 1, 4, 0, 1, 4, 1, 1, 0, 4, 3, 1, 1, 2, 3,
		1, 1,
	}},
	{Shape{2, 3, 4, 5}, []int{
		1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1,
		1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0,
		0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1,
		0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1,
		1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1,
		0, 0, 0, 0, 0,
	}},
}

var argminCorrect = []struct {
	shape Shape
	data  []int
}{
	{Shape{3, 4, 5, 2}, []int{
		0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0,
		0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1,
		0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0,
		0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0,
		1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1,
		0, 1, 1, 0, 1,
	}},
	{Shape{2, 4, 5, 2}, []int{
		2, 1, 0, 0, 1, 2, 1, 2, 1, 2, 1, 0, 0, 2, 1, 0, 1, 2, 0, 1, 0, 2, 2,
		0, 0, 1, 2, 0, 0, 1, 2, 1, 0, 1, 0, 2, 0, 1, 0, 1, 2, 1, 2, 1, 2, 1,
		2, 1, 1, 0, 2, 0, 1, 0, 1, 2, 0, 1, 2, 0, 1, 2, 2, 2, 0, 0, 1, 0, 2,
		2, 0, 0, 0, 1, 2, 2, 2, 2, 1, 1,
	}},
	{Shape{2, 3, 5, 2}, []int{
		0, 1, 0, 2, 2, 1, 3, 2, 3, 2, 1, 0, 3, 3, 0, 1, 0, 3, 0, 2, 0, 1, 0,
		1, 3, 0, 2, 1, 0, 0, 3, 1, 3, 1, 2, 2, 1, 2, 0, 1, 3, 0, 1, 0, 1, 0,
		2, 1, 0, 3, 0, 2, 0, 0, 0, 1, 0, 1, 1, 1,
	}},
	{Shape{2, 3, 4, 2}, []int{
		1, 0, 0, 0, 2, 3, 4, 0, 3, 0, 3, 0, 4, 4, 3, 1, 0, 2, 3, 0, 3, 0, 0,
		2, 4, 4, 3, 4, 2, 3, 0, 0, 4, 0, 1, 3, 3, 2, 0, 4, 2, 1, 4, 2, 4, 0,
		2, 0,
	}},
	{Shape{2, 3, 4, 5}, []int{
		0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0,
		0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1,
		1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0,
		1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0,
		0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0,
		1, 1, 1, 0, 1,
	}},
}

func TestDense_Argmax_I(t *testing.T) {
	assert := assert.New(t)
	var T, argmax *Dense
	var err error
	T = basicDenseI.Clone().(*Dense)
	for i := 0; i < T.Dims(); i++ {
		if argmax, err = T.Argmax(i); err != nil {
			t.Error(err)
			continue
		}

		assert.True(argmaxCorrect[i].shape.Eq(argmax.Shape()), "Argmax(%d) error. Want shape %v. Got %v", i, argmaxCorrect[i].shape)
		assert.Equal(argmaxCorrect[i].data, argmax.Data(), "Argmax(%d) error. ", i)
	}
	// test all axes
	if argmax, err = T.Argmax(AllAxes); err != nil {
		t.Error(err)
		return
	}
	assert.True(argmax.IsScalar())
	assert.Equal(7, argmax.ScalarValue())

	// idiotsville
	_, err = T.Argmax(10000)
	assert.NotNil(err)

}
func TestDense_Argmin_I(t *testing.T) {
	assert := assert.New(t)
	var T, argmin *Dense
	var err error
	T = basicDenseI.Clone().(*Dense)
	for i := 0; i < T.Dims(); i++ {
		if argmin, err = T.Argmin(i); err != nil {
			t.Error(err)
			continue
		}

		assert.True(argminCorrect[i].shape.Eq(argmin.Shape()), "Argmin(%d) error. Want shape %v. Got %v", i, argminCorrect[i].shape)
		assert.Equal(argminCorrect[i].data, argmin.Data(), "Argmin(%d) error. ", i)
	}
	// test all axes
	if argmin, err = T.Argmin(AllAxes); err != nil {
		t.Error(err)
		return
	}
	assert.True(argmin.IsScalar())
	assert.Equal(11, argmin.ScalarValue())

	// idiotsville
	_, err = T.Argmin(10000)
	assert.NotNil(err)

}
func TestDense_Argmax_I8(t *testing.T) {
	assert := assert.New(t)
	var T, argmax *Dense
	var err error
	T = basicDenseI8.Clone().(*Dense)
	for i := 0; i < T.Dims(); i++ {
		if argmax, err = T.Argmax(i); err != nil {
			t.Error(err)
			continue
		}

		assert.True(argmaxCorrect[i].shape.Eq(argmax.Shape()), "Argmax(%d) error. Want shape %v. Got %v", i, argmaxCorrect[i].shape)
		assert.Equal(argmaxCorrect[i].data, argmax.Data(), "Argmax(%d) error. ", i)
	}
	// test all axes
	if argmax, err = T.Argmax(AllAxes); err != nil {
		t.Error(err)
		return
	}
	assert.True(argmax.IsScalar())
	assert.Equal(7, argmax.ScalarValue())

	// idiotsville
	_, err = T.Argmax(10000)
	assert.NotNil(err)

}
func TestDense_Argmin_I8(t *testing.T) {
	assert := assert.New(t)
	var T, argmin *Dense
	var err error
	T = basicDenseI8.Clone().(*Dense)
	for i := 0; i < T.Dims(); i++ {
		if argmin, err = T.Argmin(i); err != nil {
			t.Error(err)
			continue
		}

		assert.True(argminCorrect[i].shape.Eq(argmin.Shape()), "Argmin(%d) error. Want shape %v. Got %v", i, argminCorrect[i].shape)
		assert.Equal(argminCorrect[i].data, argmin.Data(), "Argmin(%d) error. ", i)
	}
	// test all axes
	if argmin, err = T.Argmin(AllAxes); err != nil {
		t.Error(err)
		return
	}
	assert.True(argmin.IsScalar())
	assert.Equal(11, argmin.ScalarValue())

	// idiotsville
	_, err = T.Argmin(10000)
	assert.NotNil(err)

}
func TestDense_Argmax_I16(t *testing.T) {
	assert := assert.New(t)
	var T, argmax *Dense
	var err error
	T = basicDenseI16.Clone().(*Dense)
	for i := 0; i < T.Dims(); i++ {
		if argmax, err = T.Argmax(i); err != nil {
			t.Error(err)
			continue
		}

		assert.True(argmaxCorrect[i].shape.Eq(argmax.Shape()), "Argmax(%d) error. Want shape %v. Got %v", i, argmaxCorrect[i].shape)
		assert.Equal(argmaxCorrect[i].data, argmax.Data(), "Argmax(%d) error. ", i)
	}
	// test all axes
	if argmax, err = T.Argmax(AllAxes); err != nil {
		t.Error(err)
		return
	}
	assert.True(argmax.IsScalar())
	assert.Equal(7, argmax.ScalarValue())

	// idiotsville
	_, err = T.Argmax(10000)
	assert.NotNil(err)

}
func TestDense_Argmin_I16(t *testing.T) {
	assert := assert.New(t)
	var T, argmin *Dense
	var err error
	T = basicDenseI16.Clone().(*Dense)
	for i := 0; i < T.Dims(); i++ {
		if argmin, err = T.Argmin(i); err != nil {
			t.Error(err)
			continue
		}

		assert.True(argminCorrect[i].shape.Eq(argmin.Shape()), "Argmin(%d) error. Want shape %v. Got %v", i, argminCorrect[i].shape)
		assert.Equal(argminCorrect[i].data, argmin.Data(), "Argmin(%d) error. ", i)
	}
	// test all axes
	if argmin, err = T.Argmin(AllAxes); err != nil {
		t.Error(err)
		return
	}
	assert.True(argmin.IsScalar())
	assert.Equal(11, argmin.ScalarValue())

	// idiotsville
	_, err = T.Argmin(10000)
	assert.NotNil(err)

}
func TestDense_Argmax_I32(t *testing.T) {
	assert := assert.New(t)
	var T, argmax *Dense
	var err error
	T = basicDenseI32.Clone().(*Dense)
	for i := 0; i < T.Dims(); i++ {
		if argmax, err = T.Argmax(i); err != nil {
			t.Error(err)
			continue
		}

		assert.True(argmaxCorrect[i].shape.Eq(argmax.Shape()), "Argmax(%d) error. Want shape %v. Got %v", i, argmaxCorrect[i].shape)
		assert.Equal(argmaxCorrect[i].data, argmax.Data(), "Argmax(%d) error. ", i)
	}
	// test all axes
	if argmax, err = T.Argmax(AllAxes); err != nil {
		t.Error(err)
		return
	}
	assert.True(argmax.IsScalar())
	assert.Equal(7, argmax.ScalarValue())

	// idiotsville
	_, err = T.Argmax(10000)
	assert.NotNil(err)

}
func TestDense_Argmin_I32(t *testing.T) {
	assert := assert.New(t)
	var T, argmin *Dense
	var err error
	T = basicDenseI32.Clone().(*Dense)
	for i := 0; i < T.Dims(); i++ {
		if argmin, err = T.Argmin(i); err != nil {
			t.Error(err)
			continue
		}

		assert.True(argminCorrect[i].shape.Eq(argmin.Shape()), "Argmin(%d) error. Want shape %v. Got %v", i, argminCorrect[i].shape)
		assert.Equal(argminCorrect[i].data, argmin.Data(), "Argmin(%d) error. ", i)
	}
	// test all axes
	if argmin, err = T.Argmin(AllAxes); err != nil {
		t.Error(err)
		return
	}
	assert.True(argmin.IsScalar())
	assert.Equal(11, argmin.ScalarValue())

	// idiotsville
	_, err = T.Argmin(10000)
	assert.NotNil(err)

}
func TestDense_Argmax_I64(t *testing.T) {
	assert := assert.New(t)
	var T, argmax *Dense
	var err error
	T = basicDenseI64.Clone().(*Dense)
	for i := 0; i < T.Dims(); i++ {
		if argmax, err = T.Argmax(i); err != nil {
			t.Error(err)
			continue
		}

		assert.True(argmaxCorrect[i].shape.Eq(argmax.Shape()), "Argmax(%d) error. Want shape %v. Got %v", i, argmaxCorrect[i].shape)
		assert.Equal(argmaxCorrect[i].data, argmax.Data(), "Argmax(%d) error. ", i)
	}
	// test all axes
	if argmax, err = T.Argmax(AllAxes); err != nil {
		t.Error(err)
		return
	}
	assert.True(argmax.IsScalar())
	assert.Equal(7, argmax.ScalarValue())

	// idiotsville
	_, err = T.Argmax(10000)
	assert.NotNil(err)

}
func TestDense_Argmin_I64(t *testing.T) {
	assert := assert.New(t)
	var T, argmin *Dense
	var err error
	T = basicDenseI64.Clone().(*Dense)
	for i := 0; i < T.Dims(); i++ {
		if argmin, err = T.Argmin(i); err != nil {
			t.Error(err)
			continue
		}

		assert.True(argminCorrect[i].shape.Eq(argmin.Shape()), "Argmin(%d) error. Want shape %v. Got %v", i, argminCorrect[i].shape)
		assert.Equal(argminCorrect[i].data, argmin.Data(), "Argmin(%d) error. ", i)
	}
	// test all axes
	if argmin, err = T.Argmin(AllAxes); err != nil {
		t.Error(err)
		return
	}
	assert.True(argmin.IsScalar())
	assert.Equal(11, argmin.ScalarValue())

	// idiotsville
	_, err = T.Argmin(10000)
	assert.NotNil(err)

}
func TestDense_Argmax_U(t *testing.T) {
	assert := assert.New(t)
	var T, argmax *Dense
	var err error
	T = basicDenseU.Clone().(*Dense)
	for i := 0; i < T.Dims(); i++ {
		if argmax, err = T.Argmax(i); err != nil {
			t.Error(err)
			continue
		}

		assert.True(argmaxCorrect[i].shape.Eq(argmax.Shape()), "Argmax(%d) error. Want shape %v. Got %v", i, argmaxCorrect[i].shape)
		assert.Equal(argmaxCorrect[i].data, argmax.Data(), "Argmax(%d) error. ", i)
	}
	// test all axes
	if argmax, err = T.Argmax(AllAxes); err != nil {
		t.Error(err)
		return
	}
	assert.True(argmax.IsScalar())
	assert.Equal(7, argmax.ScalarValue())

	// idiotsville
	_, err = T.Argmax(10000)
	assert.NotNil(err)

}
func TestDense_Argmin_U(t *testing.T) {
	assert := assert.New(t)
	var T, argmin *Dense
	var err error
	T = basicDenseU.Clone().(*Dense)
	for i := 0; i < T.Dims(); i++ {
		if argmin, err = T.Argmin(i); err != nil {
			t.Error(err)
			continue
		}

		assert.True(argminCorrect[i].shape.Eq(argmin.Shape()), "Argmin(%d) error. Want shape %v. Got %v", i, argminCorrect[i].shape)
		assert.Equal(argminCorrect[i].data, argmin.Data(), "Argmin(%d) error. ", i)
	}
	// test all axes
	if argmin, err = T.Argmin(AllAxes); err != nil {
		t.Error(err)
		return
	}
	assert.True(argmin.IsScalar())
	assert.Equal(11, argmin.ScalarValue())

	// idiotsville
	_, err = T.Argmin(10000)
	assert.NotNil(err)

}
func TestDense_Argmax_U8(t *testing.T) {
	assert := assert.New(t)
	var T, argmax *Dense
	var err error
	T = basicDenseU8.Clone().(*Dense)
	for i := 0; i < T.Dims(); i++ {
		if argmax, err = T.Argmax(i); err != nil {
			t.Error(err)
			continue
		}

		assert.True(argmaxCorrect[i].shape.Eq(argmax.Shape()), "Argmax(%d) error. Want shape %v. Got %v", i, argmaxCorrect[i].shape)
		assert.Equal(argmaxCorrect[i].data, argmax.Data(), "Argmax(%d) error. ", i)
	}
	// test all axes
	if argmax, err = T.Argmax(AllAxes); err != nil {
		t.Error(err)
		return
	}
	assert.True(argmax.IsScalar())
	assert.Equal(7, argmax.ScalarValue())

	// idiotsville
	_, err = T.Argmax(10000)
	assert.NotNil(err)

}
func TestDense_Argmin_U8(t *testing.T) {
	assert := assert.New(t)
	var T, argmin *Dense
	var err error
	T = basicDenseU8.Clone().(*Dense)
	for i := 0; i < T.Dims(); i++ {
		if argmin, err = T.Argmin(i); err != nil {
			t.Error(err)
			continue
		}

		assert.True(argminCorrect[i].shape.Eq(argmin.Shape()), "Argmin(%d) error. Want shape %v. Got %v", i, argminCorrect[i].shape)
		assert.Equal(argminCorrect[i].data, argmin.Data(), "Argmin(%d) error. ", i)
	}
	// test all axes
	if argmin, err = T.Argmin(AllAxes); err != nil {
		t.Error(err)
		return
	}
	assert.True(argmin.IsScalar())
	assert.Equal(11, argmin.ScalarValue())

	// idiotsville
	_, err = T.Argmin(10000)
	assert.NotNil(err)

}
func TestDense_Argmax_U16(t *testing.T) {
	assert := assert.New(t)
	var T, argmax *Dense
	var err error
	T = basicDenseU16.Clone().(*Dense)
	for i := 0; i < T.Dims(); i++ {
		if argmax, err = T.Argmax(i); err != nil {
			t.Error(err)
			continue
		}

		assert.True(argmaxCorrect[i].shape.Eq(argmax.Shape()), "Argmax(%d) error. Want shape %v. Got %v", i, argmaxCorrect[i].shape)
		assert.Equal(argmaxCorrect[i].data, argmax.Data(), "Argmax(%d) error. ", i)
	}
	// test all axes
	if argmax, err = T.Argmax(AllAxes); err != nil {
		t.Error(err)
		return
	}
	assert.True(argmax.IsScalar())
	assert.Equal(7, argmax.ScalarValue())

	// idiotsville
	_, err = T.Argmax(10000)
	assert.NotNil(err)

}
func TestDense_Argmin_U16(t *testing.T) {
	assert := assert.New(t)
	var T, argmin *Dense
	var err error
	T = basicDenseU16.Clone().(*Dense)
	for i := 0; i < T.Dims(); i++ {
		if argmin, err = T.Argmin(i); err != nil {
			t.Error(err)
			continue
		}

		assert.True(argminCorrect[i].shape.Eq(argmin.Shape()), "Argmin(%d) error. Want shape %v. Got %v", i, argminCorrect[i].shape)
		assert.Equal(argminCorrect[i].data, argmin.Data(), "Argmin(%d) error. ", i)
	}
	// test all axes
	if argmin, err = T.Argmin(AllAxes); err != nil {
		t.Error(err)
		return
	}
	assert.True(argmin.IsScalar())
	assert.Equal(11, argmin.ScalarValue())

	// idiotsville
	_, err = T.Argmin(10000)
	assert.NotNil(err)

}
func TestDense_Argmax_U32(t *testing.T) {
	assert := assert.New(t)
	var T, argmax *Dense
	var err error
	T = basicDenseU32.Clone().(*Dense)
	for i := 0; i < T.Dims(); i++ {
		if argmax, err = T.Argmax(i); err != nil {
			t.Error(err)
			continue
		}

		assert.True(argmaxCorrect[i].shape.Eq(argmax.Shape()), "Argmax(%d) error. Want shape %v. Got %v", i, argmaxCorrect[i].shape)
		assert.Equal(argmaxCorrect[i].data, argmax.Data(), "Argmax(%d) error. ", i)
	}
	// test all axes
	if argmax, err = T.Argmax(AllAxes); err != nil {
		t.Error(err)
		return
	}
	assert.True(argmax.IsScalar())
	assert.Equal(7, argmax.ScalarValue())

	// idiotsville
	_, err = T.Argmax(10000)
	assert.NotNil(err)

}
func TestDense_Argmin_U32(t *testing.T) {
	assert := assert.New(t)
	var T, argmin *Dense
	var err error
	T = basicDenseU32.Clone().(*Dense)
	for i := 0; i < T.Dims(); i++ {
		if argmin, err = T.Argmin(i); err != nil {
			t.Error(err)
			continue
		}

		assert.True(argminCorrect[i].shape.Eq(argmin.Shape()), "Argmin(%d) error. Want shape %v. Got %v", i, argminCorrect[i].shape)
		assert.Equal(argminCorrect[i].data, argmin.Data(), "Argmin(%d) error. ", i)
	}
	// test all axes
	if argmin, err = T.Argmin(AllAxes); err != nil {
		t.Error(err)
		return
	}
	assert.True(argmin.IsScalar())
	assert.Equal(11, argmin.ScalarValue())

	// idiotsville
	_, err = T.Argmin(10000)
	assert.NotNil(err)

}
func TestDense_Argmax_U64(t *testing.T) {
	assert := assert.New(t)
	var T, argmax *Dense
	var err error
	T = basicDenseU64.Clone().(*Dense)
	for i := 0; i < T.Dims(); i++ {
		if argmax, err = T.Argmax(i); err != nil {
			t.Error(err)
			continue
		}

		assert.True(argmaxCorrect[i].shape.Eq(argmax.Shape()), "Argmax(%d) error. Want shape %v. Got %v", i, argmaxCorrect[i].shape)
		assert.Equal(argmaxCorrect[i].data, argmax.Data(), "Argmax(%d) error. ", i)
	}
	// test all axes
	if argmax, err = T.Argmax(AllAxes); err != nil {
		t.Error(err)
		return
	}
	assert.True(argmax.IsScalar())
	assert.Equal(7, argmax.ScalarValue())

	// idiotsville
	_, err = T.Argmax(10000)
	assert.NotNil(err)

}
func TestDense_Argmin_U64(t *testing.T) {
	assert := assert.New(t)
	var T, argmin *Dense
	var err error
	T = basicDenseU64.Clone().(*Dense)
	for i := 0; i < T.Dims(); i++ {
		if argmin, err = T.Argmin(i); err != nil {
			t.Error(err)
			continue
		}

		assert.True(argminCorrect[i].shape.Eq(argmin.Shape()), "Argmin(%d) error. Want shape %v. Got %v", i, argminCorrect[i].shape)
		assert.Equal(argminCorrect[i].data, argmin.Data(), "Argmin(%d) error. ", i)
	}
	// test all axes
	if argmin, err = T.Argmin(AllAxes); err != nil {
		t.Error(err)
		return
	}
	assert.True(argmin.IsScalar())
	assert.Equal(11, argmin.ScalarValue())

	// idiotsville
	_, err = T.Argmin(10000)
	assert.NotNil(err)

}
func TestDense_Argmax_F32(t *testing.T) {
	assert := assert.New(t)
	var T, argmax *Dense
	var err error
	T = basicDenseF32.Clone().(*Dense)
	for i := 0; i < T.Dims(); i++ {
		if argmax, err = T.Argmax(i); err != nil {
			t.Error(err)
			continue
		}

		assert.True(argmaxCorrect[i].shape.Eq(argmax.Shape()), "Argmax(%d) error. Want shape %v. Got %v", i, argmaxCorrect[i].shape)
		assert.Equal(argmaxCorrect[i].data, argmax.Data(), "Argmax(%d) error. ", i)
	}
	// test all axes
	if argmax, err = T.Argmax(AllAxes); err != nil {
		t.Error(err)
		return
	}
	assert.True(argmax.IsScalar())
	assert.Equal(7, argmax.ScalarValue())

	// test with NaN
	T = New(WithShape(4), WithBacking([]float32{1, 2, math32.NaN(), 4}))
	if argmax, err = T.Argmax(AllAxes); err != nil {
		t.Errorf("Failed test with NaN", err)
	}
	assert.True(argmax.IsScalar())
	assert.Equal(2, argmax.ScalarValue(), "NaN test")

	// test with +Inf
	T = New(WithShape(4), WithBacking([]float32{1, 2, math32.Inf(1), 4}))
	if argmax, err = T.Argmax(AllAxes); err != nil {
		t.Errorf("Failed test with +Inf", err)
	}
	assert.True(argmax.IsScalar())
	assert.Equal(2, argmax.ScalarValue(), "+Inf test")

	// test with +Inf
	T = New(WithShape(4), WithBacking([]float32{1, 2, math32.Inf(-1), 4}))
	if argmax, err = T.Argmax(AllAxes); err != nil {
		t.Errorf("Failed test with -Inf", err)
	}
	assert.True(argmax.IsScalar())
	assert.Equal(3, argmax.ScalarValue(), "-Inf test")
	// idiotsville
	_, err = T.Argmax(10000)
	assert.NotNil(err)

}
func TestDense_Argmin_F32(t *testing.T) {
	assert := assert.New(t)
	var T, argmin *Dense
	var err error
	T = basicDenseF32.Clone().(*Dense)
	for i := 0; i < T.Dims(); i++ {
		if argmin, err = T.Argmin(i); err != nil {
			t.Error(err)
			continue
		}

		assert.True(argminCorrect[i].shape.Eq(argmin.Shape()), "Argmin(%d) error. Want shape %v. Got %v", i, argminCorrect[i].shape)
		assert.Equal(argminCorrect[i].data, argmin.Data(), "Argmin(%d) error. ", i)
	}
	// test all axes
	if argmin, err = T.Argmin(AllAxes); err != nil {
		t.Error(err)
		return
	}
	assert.True(argmin.IsScalar())
	assert.Equal(11, argmin.ScalarValue())

	// test with NaN
	T = New(WithShape(4), WithBacking([]float32{1, 2, math32.NaN(), 4}))
	if argmin, err = T.Argmin(AllAxes); err != nil {
		t.Errorf("Failed test with NaN", err)
	}
	assert.True(argmin.IsScalar())
	assert.Equal(2, argmin.ScalarValue(), "NaN test")

	// test with +Inf
	T = New(WithShape(4), WithBacking([]float32{1, 2, math32.Inf(1), 4}))
	if argmin, err = T.Argmin(AllAxes); err != nil {
		t.Errorf("Failed test with +Inf", err)
	}
	assert.True(argmin.IsScalar())
	assert.Equal(0, argmin.ScalarValue(), "+Inf test")

	// test with +Inf
	T = New(WithShape(4), WithBacking([]float32{1, 2, math32.Inf(-1), 4}))
	if argmin, err = T.Argmin(AllAxes); err != nil {
		t.Errorf("Failed test with -Inf", err)
	}
	assert.True(argmin.IsScalar())
	assert.Equal(2, argmin.ScalarValue(), "-Inf test")
	// idiotsville
	_, err = T.Argmin(10000)
	assert.NotNil(err)

}
func TestDense_Argmax_F64(t *testing.T) {
	assert := assert.New(t)
	var T, argmax *Dense
	var err error
	T = basicDenseF64.Clone().(*Dense)
	for i := 0; i < T.Dims(); i++ {
		if argmax, err = T.Argmax(i); err != nil {
			t.Error(err)
			continue
		}

		assert.True(argmaxCorrect[i].shape.Eq(argmax.Shape()), "Argmax(%d) error. Want shape %v. Got %v", i, argmaxCorrect[i].shape)
		assert.Equal(argmaxCorrect[i].data, argmax.Data(), "Argmax(%d) error. ", i)
	}
	// test all axes
	if argmax, err = T.Argmax(AllAxes); err != nil {
		t.Error(err)
		return
	}
	assert.True(argmax.IsScalar())
	assert.Equal(7, argmax.ScalarValue())

	// test with NaN
	T = New(WithShape(4), WithBacking([]float64{1, 2, math.NaN(), 4}))
	if argmax, err = T.Argmax(AllAxes); err != nil {
		t.Errorf("Failed test with NaN", err)
	}
	assert.True(argmax.IsScalar())
	assert.Equal(2, argmax.ScalarValue(), "NaN test")

	// test with +Inf
	T = New(WithShape(4), WithBacking([]float64{1, 2, math.Inf(1), 4}))
	if argmax, err = T.Argmax(AllAxes); err != nil {
		t.Errorf("Failed test with +Inf", err)
	}
	assert.True(argmax.IsScalar())
	assert.Equal(2, argmax.ScalarValue(), "+Inf test")

	// test with +Inf
	T = New(WithShape(4), WithBacking([]float64{1, 2, math.Inf(-1), 4}))
	if argmax, err = T.Argmax(AllAxes); err != nil {
		t.Errorf("Failed test with -Inf", err)
	}
	assert.True(argmax.IsScalar())
	assert.Equal(3, argmax.ScalarValue(), "-Inf test")
	// idiotsville
	_, err = T.Argmax(10000)
	assert.NotNil(err)

}
func TestDense_Argmin_F64(t *testing.T) {
	assert := assert.New(t)
	var T, argmin *Dense
	var err error
	T = basicDenseF64.Clone().(*Dense)
	for i := 0; i < T.Dims(); i++ {
		if argmin, err = T.Argmin(i); err != nil {
			t.Error(err)
			continue
		}

		assert.True(argminCorrect[i].shape.Eq(argmin.Shape()), "Argmin(%d) error. Want shape %v. Got %v", i, argminCorrect[i].shape)
		assert.Equal(argminCorrect[i].data, argmin.Data(), "Argmin(%d) error. ", i)
	}
	// test all axes
	if argmin, err = T.Argmin(AllAxes); err != nil {
		t.Error(err)
		return
	}
	assert.True(argmin.IsScalar())
	assert.Equal(11, argmin.ScalarValue())

	// test with NaN
	T = New(WithShape(4), WithBacking([]float64{1, 2, math.NaN(), 4}))
	if argmin, err = T.Argmin(AllAxes); err != nil {
		t.Errorf("Failed test with NaN", err)
	}
	assert.True(argmin.IsScalar())
	assert.Equal(2, argmin.ScalarValue(), "NaN test")

	// test with +Inf
	T = New(WithShape(4), WithBacking([]float64{1, 2, math.Inf(1), 4}))
	if argmin, err = T.Argmin(AllAxes); err != nil {
		t.Errorf("Failed test with +Inf", err)
	}
	assert.True(argmin.IsScalar())
	assert.Equal(0, argmin.ScalarValue(), "+Inf test")

	// test with +Inf
	T = New(WithShape(4), WithBacking([]float64{1, 2, math.Inf(-1), 4}))
	if argmin, err = T.Argmin(AllAxes); err != nil {
		t.Errorf("Failed test with -Inf", err)
	}
	assert.True(argmin.IsScalar())
	assert.Equal(2, argmin.ScalarValue(), "-Inf test")
	// idiotsville
	_, err = T.Argmin(10000)
	assert.NotNil(err)

}
