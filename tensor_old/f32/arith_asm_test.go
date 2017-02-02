// +build sse avx

/*

IMPORTANT NOTE:

Currently vecDiv does not handle division by zero correctly. It returns a NaN instead of +Inf

*/

package tensorf32

import (
	"testing"
	"unsafe"

	"github.com/chewxy/math32"
	"github.com/stretchr/testify/assert"
)

// 1049 is actually a prime, so it cannot be divisible by any other number
// This is a good way to test that the remainder part of the vecAdd/Sub/Mul/Div/Pow works
const (
	niceprime = 37
	// niceprime = 1049
	// niceprime = 597929
	// niceprime = 1299827 // because sometimes I feel like being an idiot
)

// this file is mainly added to facilitate testing of the ASM code, and that it matches up correctly with the expected results

func TestVecAdd(t *testing.T) {
	assert := assert.New(t)

	a := RangeFloat32(0, niceprime-1)

	correct := RangeFloat32(0, niceprime-1)
	for i, v := range correct {
		correct[i] = v + v
	}
	vecAdd(a, a)
	assert.Equal(correct, a)

	b := RangeFloat32(niceprime, 2*niceprime-1)
	for i := range correct {
		correct[i] = a[i] + b[i]
	}

	vecAdd(a, b)
	assert.Equal(correct, a)

	/* Weird Corner Cases*/
	for i := 1; i < 65; i++ {
		a = RangeFloat32(0, i)
		var testAlign bool
		addr := &a[0]
		u := uint(uintptr(unsafe.Pointer(addr)))
		if u&uint(32) != 0 {
			testAlign = true
		}

		if testAlign {
			b = RangeFloat32(i, 2*i)
			correct = make([]float32, i)
			for j := range correct {
				correct[j] = b[j] + a[j]
			}
			vecAdd(a, b)
			assert.Equal(correct, a)
		}
	}
}

func TestVecSub(t *testing.T) {
	assert := assert.New(t)

	a := RangeFloat32(0, niceprime-1)

	correct := RangeFloat32(0, niceprime-1)
	for i := range correct {
		correct[i] = correct[i] - correct[i]
	}

	vecSub(a, a)
	assert.Equal(correct, a)

	b := RangeFloat32(niceprime, 2*niceprime-1)
	for i := range correct {
		correct[i] = a[i] - b[i]
	}

	vecSub(a, b)
	assert.Equal(correct, a)

	/* Weird Corner Cases*/
	for i := 1; i < 65; i++ {
		a = RangeFloat32(0, i)
		var testAlign bool
		addr := &a[0]
		u := uint(uintptr(unsafe.Pointer(addr)))
		if u&uint(32) != 0 {
			testAlign = true
		}

		if testAlign {
			b = RangeFloat32(i, 2*i)
			correct = make([]float32, i)
			for j := range correct {
				correct[j] = a[j] - b[j]
			}
			vecSub(a, b)
			assert.Equal(correct, a)
		}
	}
}

func TestVecMul(t *testing.T) {
	assert := assert.New(t)

	a := RangeFloat32(0, niceprime-1)

	correct := RangeFloat32(0, niceprime-1)
	for i := range correct {
		correct[i] = correct[i] * correct[i]
	}
	vecMul(a, a)
	assert.Equal(correct, a)

	b := RangeFloat32(niceprime, 2*niceprime-1)
	for i := range correct {
		correct[i] = a[i] * b[i]
	}

	vecMul(a, b)
	assert.Equal(correct, a)

	/* Weird Corner Cases*/

	for i := 1; i < 65; i++ {
		a = RangeFloat32(0, i)
		var testAlign bool
		addr := &a[0]
		u := uint(uintptr(unsafe.Pointer(addr)))
		if u&uint(32) != 0 {
			testAlign = true
		}

		if testAlign {
			b = RangeFloat32(i, 2*i)
			correct = make([]float32, i)
			for j := range correct {
				correct[j] = a[j] * b[j]
			}
			vecMul(a, b)
			assert.Equal(correct, a)
		}
	}
}

func TestVecDiv(t *testing.T) {
	assert := assert.New(t)

	a := RangeFloat32(0, niceprime-1)

	correct := RangeFloat32(0, niceprime-1)
	for i := range correct {
		correct[i] = correct[i] / correct[i]
	}
	vecDiv(a, a)
	assert.Equal(correct[1:], a[1:])
	assert.Equal(true, math32.IsNaN(a[0]), "a[0] is: %v", a[0])

	b := RangeFloat32(niceprime, 2*niceprime-1)
	for i := range correct {
		correct[i] = a[i] / b[i]
	}

	vecDiv(a, b)
	assert.Equal(correct[1:], a[1:])
	assert.Equal(true, math32.IsNaN(a[0]), "a[0] is: %v", a[0])

	/* Weird Corner Cases*/

	for i := 1; i < 65; i++ {
		a = RangeFloat32(0, i)
		var testAlign bool
		addr := &a[0]
		u := uint(uintptr(unsafe.Pointer(addr)))
		if u&uint(32) != 0 {
			testAlign = true
		}

		if testAlign {
			b = RangeFloat32(i, 2*i)
			correct = make([]float32, i)
			for j := range correct {
				correct[j] = a[j] / b[j]
			}
			vecDiv(a, b)
			assert.Equal(correct[1:], a[1:])
		}
	}

}

func TestVecSqrt(t *testing.T) {
	assert := assert.New(t)

	a := RangeFloat32(0, niceprime-1)

	correct := RangeFloat32(0, niceprime-1)
	for i, v := range correct {
		correct[i] = math32.Sqrt(v)
	}
	vecSqrt(a)
	assert.Equal(correct, a)

	// negatives
	a = []float32{-1, -2, -3, -4}
	vecSqrt(a)

	for _, v := range a {
		if !math32.IsNaN(v) {
			t.Error("Expected NaN")
		}
	}

	/* Weird Corner Cases*/
	for i := 1; i < 65; i++ {
		a = RangeFloat32(0, i)
		var testAlign bool
		addr := &a[0]
		u := uint(uintptr(unsafe.Pointer(addr)))
		if u&uint(32) != 0 {
			testAlign = true
		}

		if testAlign {
			correct = make([]float32, i)
			for j := range correct {
				correct[j] = math32.Sqrt(a[j])
			}
			vecSqrt(a)
			assert.Equal(correct, a)
		}
	}
}

func TestVecInvSqrt(t *testing.T) {

	assert := assert.New(t)
	a := RangeFloat32(0, niceprime-1)

	correct := RangeFloat32(0, niceprime-1)
	for i, v := range correct {
		correct[i] = float32(1.0) / math32.Sqrt(v)
	}

	vecInvSqrt(a)
	assert.Equal(correct[1:], a[1:])
	if !math32.IsInf(a[0], 0) {
		t.Error("1/0 should be +Inf or -Inf")
	}

	// Weird Corner Cases

	for i := 1; i < 65; i++ {
		a = RangeFloat32(0, i)
		var testAlign bool
		addr := &a[0]
		u := uint(uintptr(unsafe.Pointer(addr)))
		if u&uint(32) != 0 {
			testAlign = true
		}

		if testAlign {
			correct = make([]float32, i)
			for j := range correct {
				correct[j] = 1.0 / math32.Sqrt(a[j])
			}
			vecInvSqrt(a)
			assert.Equal(correct[1:], a[1:], "i = %d, %v", i, RangeFloat32(0, i))
			if !math32.IsInf(a[0], 0) {
				t.Error("1/0 should be +Inf or -Inf")
			}
		}
	}
}

/* BENCHMARKS */

func _vanillaVecAdd(a, b []float32) {
	for i := range a {
		a[i] += b[i]
	}
}

func BenchmarkVecAdd(b *testing.B) {
	x := RangeFloat32(0, niceprime)
	y := RangeFloat32(niceprime, 2*niceprime)

	for n := 0; n < b.N; n++ {
		vecAdd(x, y)
	}
}

func BenchmarkVanillaVecAdd(b *testing.B) {
	x := RangeFloat32(0, niceprime)
	y := RangeFloat32(niceprime, 2*niceprime)

	for n := 0; n < b.N; n++ {
		_vanillaVecAdd(x, y)
	}
}

func _vanillaVecSub(a, b []float32) {
	for i := range a {
		a[i] -= b[i]
	}
}

func BenchmarkVecSub(b *testing.B) {
	x := RangeFloat32(0, niceprime)
	y := RangeFloat32(niceprime, 2*niceprime)

	for n := 0; n < b.N; n++ {
		vecSub(x, y)
	}
}

func BenchmarkVanillaVecSub(b *testing.B) {
	x := RangeFloat32(0, niceprime)
	y := RangeFloat32(niceprime, 2*niceprime)

	for n := 0; n < b.N; n++ {
		_vanillaVecSub(x, y)
	}
}

func _vanillaVecMul(a, b []float32) {
	for i := range a {
		a[i] *= b[i]
	}
}

func BenchmarkVecMul(b *testing.B) {
	x := RangeFloat32(0, niceprime)
	y := RangeFloat32(niceprime, 2*niceprime)

	for n := 0; n < b.N; n++ {
		vecMul(x, y)
	}
}

func BenchmarkVanillaVecMul(b *testing.B) {
	x := RangeFloat32(0, niceprime)
	y := RangeFloat32(niceprime, 2*niceprime)

	for n := 0; n < b.N; n++ {
		_vanillaVecMul(x, y)
	}
}

func _vanillaVecDiv(a, b []float32) {
	for i := range a {
		a[i] /= b[i]
	}
}

func BenchmarkVecDiv(b *testing.B) {
	x := RangeFloat32(0, niceprime)
	y := RangeFloat32(niceprime, 2*niceprime)

	for n := 0; n < b.N; n++ {
		vecDiv(x, y)
	}
}

func BenchmarkVanillaVecDiv(b *testing.B) {
	x := RangeFloat32(0, niceprime)
	y := RangeFloat32(niceprime, 2*niceprime)

	for n := 0; n < b.N; n++ {
		_vanillaVecDiv(x, y)
	}
}

func _vanillaVecSqrt(a []float32) {
	for i, v := range a {
		a[i] = math32.Sqrt(v)
	}
}

func BenchmarkVecSqrt(b *testing.B) {
	x := RangeFloat32(0, niceprime)

	for n := 0; n < b.N; n++ {
		vecSqrt(x)
	}
}

func BenchmarkVanillaVecSqrt(b *testing.B) {
	x := RangeFloat32(0, niceprime)

	for n := 0; n < b.N; n++ {
		_vanillaVecSqrt(x)
	}
}

func _vanillaVecInverseSqrt(a []float32) {
	for i, v := range a {
		a[i] = 1.0 / math32.Sqrt(v)
	}
}

func BenchmarkVecInvSqrt(b *testing.B) {
	x := RangeFloat32(0, niceprime)

	for n := 0; n < b.N; n++ {
		vecInvSqrt(x)
	}
}

func BenchmarkVanillaVecInvSqrt(b *testing.B) {
	x := RangeFloat32(0, niceprime)

	for n := 0; n < b.N; n++ {
		_vanillaVecInverseSqrt(x)
	}
}
