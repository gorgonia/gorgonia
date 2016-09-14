// +build DONOTUSE
// +build sse avx

/*

IMPORTANT NOTE:

Currently vecDiv does not handle division by zero correctly. It returns a NaN instead of +Inf

*/

package tensorf32

import (
	"testing"

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
	t.Logf("correct: %v", correct)
	for i, v := range correct {
		correct[i] = v + v
	}
	t.Logf("correct: %v", correct)
	t.Logf("a : %v", a)

	vecAdd(a, a)
	t.Logf("correct: %v", correct)
	t.Logf("a : %v", a)
	assert.Equal(correct, a)

	// b := RangeFloat32(niceprime, 2*niceprime-1)
	// for i := range correct {
	// 	correct[i] = a[i] + b[i]
	// }

	// vecAdd(a, b)
	// assert.Equal(correct, a)

	// /* Weird Corner Cases*/
	// for i := 1; i < 65; i++ {
	// 	a = RangeFloat32(0, i)
	// 	var testAlign bool
	// 	addr := &a[0]
	// 	u := uint(uintptr(unsafe.Pointer(addr)))
	// 	if u&uint(32) != 0 {
	// 		testAlign = true
	// 	}

	// 	if testAlign {
	// 		b = RangeFloat32(i, 2*i)
	// 		correct = make([]float32, i)
	// 		for j := range correct {
	// 			correct[j] = b[j] + a[j]
	// 		}
	// 		vecAdd(a, b)
	// 		assert.Equal(correct, a)
	// 	}
	// }
}
