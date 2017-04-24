package tensor

import (
	"fmt"
	"reflect"
)

/*
GENERATED FILE. DO NOT EDIT
*/

/* bool */

func (t *Dense) doViewStackB(retVal *Dense, axisStride, batches int, ch chan int, others []*Dense, chs []chan int) {
	data := retVal.bools()[:0]
	mask := retVal.mask[:0]
	if t.IsMasked() {
		fmt.Println("do this")
	}
	retIsMasked := t.IsMasked()
	for _, ot := range others {
		retIsMasked = retIsMasked || ot.IsMasked()
	}
	for i := 0; i < batches; i++ {
		isMasked := t.IsMasked()
		var j int
		for j = 0; j < axisStride; j++ {
			id, ok := <-ch
			if !ok {
				break
			}
			data = append(data, t.bools()[id])
			if isMasked {
				mask = append(mask, t.mask[id])
			}
		}
		if retIsMasked && (!isMasked) {
			mask = append(mask, make([]bool, j)...)
		}

		var ot *Dense
		for j, ot = range others {
			isMasked = ot.IsMasked()
			var k int
			for k = 0; k < axisStride; k++ {
				id, ok := <-chs[j]
				if !ok {
					break
				}
				data = append(data, ot.bools()[id])
				if isMasked {
					mask = append(mask, ot.mask[id])
				}
			}
			if retIsMasked && (!isMasked) {
				mask = append(mask, make([]bool, k)...)
			}
		}
	}
	retVal.mask = mask
}

/* int */

func (t *Dense) doViewStackI(retVal *Dense, axisStride, batches int, ch chan int, others []*Dense, chs []chan int) {
	data := retVal.ints()[:0]
	mask := retVal.mask[:0]
	if t.IsMasked() {
		fmt.Println("do this")
	}
	retIsMasked := t.IsMasked()
	for _, ot := range others {
		retIsMasked = retIsMasked || ot.IsMasked()
	}
	for i := 0; i < batches; i++ {
		isMasked := t.IsMasked()
		var j int
		for j = 0; j < axisStride; j++ {
			id, ok := <-ch
			if !ok {
				break
			}
			data = append(data, t.ints()[id])
			if isMasked {
				mask = append(mask, t.mask[id])
			}
		}
		if retIsMasked && (!isMasked) {
			mask = append(mask, make([]bool, j)...)
		}

		var ot *Dense
		for j, ot = range others {
			isMasked = ot.IsMasked()
			var k int
			for k = 0; k < axisStride; k++ {
				id, ok := <-chs[j]
				if !ok {
					break
				}
				data = append(data, ot.ints()[id])
				if isMasked {
					mask = append(mask, ot.mask[id])
				}
			}
			if retIsMasked && (!isMasked) {
				mask = append(mask, make([]bool, k)...)
			}
		}
	}
	retVal.mask = mask
}

/* int8 */

func (t *Dense) doViewStackI8(retVal *Dense, axisStride, batches int, ch chan int, others []*Dense, chs []chan int) {
	data := retVal.int8s()[:0]
	mask := retVal.mask[:0]
	if t.IsMasked() {
		fmt.Println("do this")
	}
	retIsMasked := t.IsMasked()
	for _, ot := range others {
		retIsMasked = retIsMasked || ot.IsMasked()
	}
	for i := 0; i < batches; i++ {
		isMasked := t.IsMasked()
		var j int
		for j = 0; j < axisStride; j++ {
			id, ok := <-ch
			if !ok {
				break
			}
			data = append(data, t.int8s()[id])
			if isMasked {
				mask = append(mask, t.mask[id])
			}
		}
		if retIsMasked && (!isMasked) {
			mask = append(mask, make([]bool, j)...)
		}

		var ot *Dense
		for j, ot = range others {
			isMasked = ot.IsMasked()
			var k int
			for k = 0; k < axisStride; k++ {
				id, ok := <-chs[j]
				if !ok {
					break
				}
				data = append(data, ot.int8s()[id])
				if isMasked {
					mask = append(mask, ot.mask[id])
				}
			}
			if retIsMasked && (!isMasked) {
				mask = append(mask, make([]bool, k)...)
			}
		}
	}
	retVal.mask = mask
}

/* int16 */

func (t *Dense) doViewStackI16(retVal *Dense, axisStride, batches int, ch chan int, others []*Dense, chs []chan int) {
	data := retVal.int16s()[:0]
	mask := retVal.mask[:0]
	if t.IsMasked() {
		fmt.Println("do this")
	}
	retIsMasked := t.IsMasked()
	for _, ot := range others {
		retIsMasked = retIsMasked || ot.IsMasked()
	}
	for i := 0; i < batches; i++ {
		isMasked := t.IsMasked()
		var j int
		for j = 0; j < axisStride; j++ {
			id, ok := <-ch
			if !ok {
				break
			}
			data = append(data, t.int16s()[id])
			if isMasked {
				mask = append(mask, t.mask[id])
			}
		}
		if retIsMasked && (!isMasked) {
			mask = append(mask, make([]bool, j)...)
		}

		var ot *Dense
		for j, ot = range others {
			isMasked = ot.IsMasked()
			var k int
			for k = 0; k < axisStride; k++ {
				id, ok := <-chs[j]
				if !ok {
					break
				}
				data = append(data, ot.int16s()[id])
				if isMasked {
					mask = append(mask, ot.mask[id])
				}
			}
			if retIsMasked && (!isMasked) {
				mask = append(mask, make([]bool, k)...)
			}
		}
	}
	retVal.mask = mask
}

/* int32 */

func (t *Dense) doViewStackI32(retVal *Dense, axisStride, batches int, ch chan int, others []*Dense, chs []chan int) {
	data := retVal.int32s()[:0]
	mask := retVal.mask[:0]
	if t.IsMasked() {
		fmt.Println("do this")
	}
	retIsMasked := t.IsMasked()
	for _, ot := range others {
		retIsMasked = retIsMasked || ot.IsMasked()
	}
	for i := 0; i < batches; i++ {
		isMasked := t.IsMasked()
		var j int
		for j = 0; j < axisStride; j++ {
			id, ok := <-ch
			if !ok {
				break
			}
			data = append(data, t.int32s()[id])
			if isMasked {
				mask = append(mask, t.mask[id])
			}
		}
		if retIsMasked && (!isMasked) {
			mask = append(mask, make([]bool, j)...)
		}

		var ot *Dense
		for j, ot = range others {
			isMasked = ot.IsMasked()
			var k int
			for k = 0; k < axisStride; k++ {
				id, ok := <-chs[j]
				if !ok {
					break
				}
				data = append(data, ot.int32s()[id])
				if isMasked {
					mask = append(mask, ot.mask[id])
				}
			}
			if retIsMasked && (!isMasked) {
				mask = append(mask, make([]bool, k)...)
			}
		}
	}
	retVal.mask = mask
}

/* int64 */

func (t *Dense) doViewStackI64(retVal *Dense, axisStride, batches int, ch chan int, others []*Dense, chs []chan int) {
	data := retVal.int64s()[:0]
	mask := retVal.mask[:0]
	if t.IsMasked() {
		fmt.Println("do this")
	}
	retIsMasked := t.IsMasked()
	for _, ot := range others {
		retIsMasked = retIsMasked || ot.IsMasked()
	}
	for i := 0; i < batches; i++ {
		isMasked := t.IsMasked()
		var j int
		for j = 0; j < axisStride; j++ {
			id, ok := <-ch
			if !ok {
				break
			}
			data = append(data, t.int64s()[id])
			if isMasked {
				mask = append(mask, t.mask[id])
			}
		}
		if retIsMasked && (!isMasked) {
			mask = append(mask, make([]bool, j)...)
		}

		var ot *Dense
		for j, ot = range others {
			isMasked = ot.IsMasked()
			var k int
			for k = 0; k < axisStride; k++ {
				id, ok := <-chs[j]
				if !ok {
					break
				}
				data = append(data, ot.int64s()[id])
				if isMasked {
					mask = append(mask, ot.mask[id])
				}
			}
			if retIsMasked && (!isMasked) {
				mask = append(mask, make([]bool, k)...)
			}
		}
	}
	retVal.mask = mask
}

/* uint */

func (t *Dense) doViewStackU(retVal *Dense, axisStride, batches int, ch chan int, others []*Dense, chs []chan int) {
	data := retVal.uints()[:0]
	mask := retVal.mask[:0]
	if t.IsMasked() {
		fmt.Println("do this")
	}
	retIsMasked := t.IsMasked()
	for _, ot := range others {
		retIsMasked = retIsMasked || ot.IsMasked()
	}
	for i := 0; i < batches; i++ {
		isMasked := t.IsMasked()
		var j int
		for j = 0; j < axisStride; j++ {
			id, ok := <-ch
			if !ok {
				break
			}
			data = append(data, t.uints()[id])
			if isMasked {
				mask = append(mask, t.mask[id])
			}
		}
		if retIsMasked && (!isMasked) {
			mask = append(mask, make([]bool, j)...)
		}

		var ot *Dense
		for j, ot = range others {
			isMasked = ot.IsMasked()
			var k int
			for k = 0; k < axisStride; k++ {
				id, ok := <-chs[j]
				if !ok {
					break
				}
				data = append(data, ot.uints()[id])
				if isMasked {
					mask = append(mask, ot.mask[id])
				}
			}
			if retIsMasked && (!isMasked) {
				mask = append(mask, make([]bool, k)...)
			}
		}
	}
	retVal.mask = mask
}

/* uint8 */

func (t *Dense) doViewStackU8(retVal *Dense, axisStride, batches int, ch chan int, others []*Dense, chs []chan int) {
	data := retVal.uint8s()[:0]
	mask := retVal.mask[:0]
	if t.IsMasked() {
		fmt.Println("do this")
	}
	retIsMasked := t.IsMasked()
	for _, ot := range others {
		retIsMasked = retIsMasked || ot.IsMasked()
	}
	for i := 0; i < batches; i++ {
		isMasked := t.IsMasked()
		var j int
		for j = 0; j < axisStride; j++ {
			id, ok := <-ch
			if !ok {
				break
			}
			data = append(data, t.uint8s()[id])
			if isMasked {
				mask = append(mask, t.mask[id])
			}
		}
		if retIsMasked && (!isMasked) {
			mask = append(mask, make([]bool, j)...)
		}

		var ot *Dense
		for j, ot = range others {
			isMasked = ot.IsMasked()
			var k int
			for k = 0; k < axisStride; k++ {
				id, ok := <-chs[j]
				if !ok {
					break
				}
				data = append(data, ot.uint8s()[id])
				if isMasked {
					mask = append(mask, ot.mask[id])
				}
			}
			if retIsMasked && (!isMasked) {
				mask = append(mask, make([]bool, k)...)
			}
		}
	}
	retVal.mask = mask
}

/* uint16 */

func (t *Dense) doViewStackU16(retVal *Dense, axisStride, batches int, ch chan int, others []*Dense, chs []chan int) {
	data := retVal.uint16s()[:0]
	mask := retVal.mask[:0]
	if t.IsMasked() {
		fmt.Println("do this")
	}
	retIsMasked := t.IsMasked()
	for _, ot := range others {
		retIsMasked = retIsMasked || ot.IsMasked()
	}
	for i := 0; i < batches; i++ {
		isMasked := t.IsMasked()
		var j int
		for j = 0; j < axisStride; j++ {
			id, ok := <-ch
			if !ok {
				break
			}
			data = append(data, t.uint16s()[id])
			if isMasked {
				mask = append(mask, t.mask[id])
			}
		}
		if retIsMasked && (!isMasked) {
			mask = append(mask, make([]bool, j)...)
		}

		var ot *Dense
		for j, ot = range others {
			isMasked = ot.IsMasked()
			var k int
			for k = 0; k < axisStride; k++ {
				id, ok := <-chs[j]
				if !ok {
					break
				}
				data = append(data, ot.uint16s()[id])
				if isMasked {
					mask = append(mask, ot.mask[id])
				}
			}
			if retIsMasked && (!isMasked) {
				mask = append(mask, make([]bool, k)...)
			}
		}
	}
	retVal.mask = mask
}

/* uint32 */

func (t *Dense) doViewStackU32(retVal *Dense, axisStride, batches int, ch chan int, others []*Dense, chs []chan int) {
	data := retVal.uint32s()[:0]
	mask := retVal.mask[:0]
	if t.IsMasked() {
		fmt.Println("do this")
	}
	retIsMasked := t.IsMasked()
	for _, ot := range others {
		retIsMasked = retIsMasked || ot.IsMasked()
	}
	for i := 0; i < batches; i++ {
		isMasked := t.IsMasked()
		var j int
		for j = 0; j < axisStride; j++ {
			id, ok := <-ch
			if !ok {
				break
			}
			data = append(data, t.uint32s()[id])
			if isMasked {
				mask = append(mask, t.mask[id])
			}
		}
		if retIsMasked && (!isMasked) {
			mask = append(mask, make([]bool, j)...)
		}

		var ot *Dense
		for j, ot = range others {
			isMasked = ot.IsMasked()
			var k int
			for k = 0; k < axisStride; k++ {
				id, ok := <-chs[j]
				if !ok {
					break
				}
				data = append(data, ot.uint32s()[id])
				if isMasked {
					mask = append(mask, ot.mask[id])
				}
			}
			if retIsMasked && (!isMasked) {
				mask = append(mask, make([]bool, k)...)
			}
		}
	}
	retVal.mask = mask
}

/* uint64 */

func (t *Dense) doViewStackU64(retVal *Dense, axisStride, batches int, ch chan int, others []*Dense, chs []chan int) {
	data := retVal.uint64s()[:0]
	mask := retVal.mask[:0]
	if t.IsMasked() {
		fmt.Println("do this")
	}
	retIsMasked := t.IsMasked()
	for _, ot := range others {
		retIsMasked = retIsMasked || ot.IsMasked()
	}
	for i := 0; i < batches; i++ {
		isMasked := t.IsMasked()
		var j int
		for j = 0; j < axisStride; j++ {
			id, ok := <-ch
			if !ok {
				break
			}
			data = append(data, t.uint64s()[id])
			if isMasked {
				mask = append(mask, t.mask[id])
			}
		}
		if retIsMasked && (!isMasked) {
			mask = append(mask, make([]bool, j)...)
		}

		var ot *Dense
		for j, ot = range others {
			isMasked = ot.IsMasked()
			var k int
			for k = 0; k < axisStride; k++ {
				id, ok := <-chs[j]
				if !ok {
					break
				}
				data = append(data, ot.uint64s()[id])
				if isMasked {
					mask = append(mask, ot.mask[id])
				}
			}
			if retIsMasked && (!isMasked) {
				mask = append(mask, make([]bool, k)...)
			}
		}
	}
	retVal.mask = mask
}

/* float32 */

func (t *Dense) doViewStackF32(retVal *Dense, axisStride, batches int, ch chan int, others []*Dense, chs []chan int) {
	data := retVal.float32s()[:0]
	mask := retVal.mask[:0]
	if t.IsMasked() {
		fmt.Println("do this")
	}
	retIsMasked := t.IsMasked()
	for _, ot := range others {
		retIsMasked = retIsMasked || ot.IsMasked()
	}
	for i := 0; i < batches; i++ {
		isMasked := t.IsMasked()
		var j int
		for j = 0; j < axisStride; j++ {
			id, ok := <-ch
			if !ok {
				break
			}
			data = append(data, t.float32s()[id])
			if isMasked {
				mask = append(mask, t.mask[id])
			}
		}
		if retIsMasked && (!isMasked) {
			mask = append(mask, make([]bool, j)...)
		}

		var ot *Dense
		for j, ot = range others {
			isMasked = ot.IsMasked()
			var k int
			for k = 0; k < axisStride; k++ {
				id, ok := <-chs[j]
				if !ok {
					break
				}
				data = append(data, ot.float32s()[id])
				if isMasked {
					mask = append(mask, ot.mask[id])
				}
			}
			if retIsMasked && (!isMasked) {
				mask = append(mask, make([]bool, k)...)
			}
		}
	}
	retVal.mask = mask
}

/* float64 */

func (t *Dense) doViewStackF64(retVal *Dense, axisStride, batches int, ch chan int, others []*Dense, chs []chan int) {
	data := retVal.float64s()[:0]
	mask := retVal.mask[:0]
	if t.IsMasked() {
		fmt.Println("do this")
	}
	retIsMasked := t.IsMasked()
	for _, ot := range others {
		retIsMasked = retIsMasked || ot.IsMasked()
	}
	for i := 0; i < batches; i++ {
		isMasked := t.IsMasked()
		var j int
		for j = 0; j < axisStride; j++ {
			id, ok := <-ch
			if !ok {
				break
			}
			data = append(data, t.float64s()[id])
			if isMasked {
				mask = append(mask, t.mask[id])
			}
		}
		if retIsMasked && (!isMasked) {
			mask = append(mask, make([]bool, j)...)
		}

		var ot *Dense
		for j, ot = range others {
			isMasked = ot.IsMasked()
			var k int
			for k = 0; k < axisStride; k++ {
				id, ok := <-chs[j]
				if !ok {
					break
				}
				data = append(data, ot.float64s()[id])
				if isMasked {
					mask = append(mask, ot.mask[id])
				}
			}
			if retIsMasked && (!isMasked) {
				mask = append(mask, make([]bool, k)...)
			}
		}
	}
	retVal.mask = mask
}

/* complex64 */

func (t *Dense) doViewStackC64(retVal *Dense, axisStride, batches int, ch chan int, others []*Dense, chs []chan int) {
	data := retVal.complex64s()[:0]
	mask := retVal.mask[:0]
	if t.IsMasked() {
		fmt.Println("do this")
	}
	retIsMasked := t.IsMasked()
	for _, ot := range others {
		retIsMasked = retIsMasked || ot.IsMasked()
	}
	for i := 0; i < batches; i++ {
		isMasked := t.IsMasked()
		var j int
		for j = 0; j < axisStride; j++ {
			id, ok := <-ch
			if !ok {
				break
			}
			data = append(data, t.complex64s()[id])
			if isMasked {
				mask = append(mask, t.mask[id])
			}
		}
		if retIsMasked && (!isMasked) {
			mask = append(mask, make([]bool, j)...)
		}

		var ot *Dense
		for j, ot = range others {
			isMasked = ot.IsMasked()
			var k int
			for k = 0; k < axisStride; k++ {
				id, ok := <-chs[j]
				if !ok {
					break
				}
				data = append(data, ot.complex64s()[id])
				if isMasked {
					mask = append(mask, ot.mask[id])
				}
			}
			if retIsMasked && (!isMasked) {
				mask = append(mask, make([]bool, k)...)
			}
		}
	}
	retVal.mask = mask
}

/* complex128 */

func (t *Dense) doViewStackC128(retVal *Dense, axisStride, batches int, ch chan int, others []*Dense, chs []chan int) {
	data := retVal.complex128s()[:0]
	mask := retVal.mask[:0]
	if t.IsMasked() {
		fmt.Println("do this")
	}
	retIsMasked := t.IsMasked()
	for _, ot := range others {
		retIsMasked = retIsMasked || ot.IsMasked()
	}
	for i := 0; i < batches; i++ {
		isMasked := t.IsMasked()
		var j int
		for j = 0; j < axisStride; j++ {
			id, ok := <-ch
			if !ok {
				break
			}
			data = append(data, t.complex128s()[id])
			if isMasked {
				mask = append(mask, t.mask[id])
			}
		}
		if retIsMasked && (!isMasked) {
			mask = append(mask, make([]bool, j)...)
		}

		var ot *Dense
		for j, ot = range others {
			isMasked = ot.IsMasked()
			var k int
			for k = 0; k < axisStride; k++ {
				id, ok := <-chs[j]
				if !ok {
					break
				}
				data = append(data, ot.complex128s()[id])
				if isMasked {
					mask = append(mask, ot.mask[id])
				}
			}
			if retIsMasked && (!isMasked) {
				mask = append(mask, make([]bool, k)...)
			}
		}
	}
	retVal.mask = mask
}

/* string */

func (t *Dense) doViewStackStr(retVal *Dense, axisStride, batches int, ch chan int, others []*Dense, chs []chan int) {
	data := retVal.strings()[:0]
	mask := retVal.mask[:0]
	if t.IsMasked() {
		fmt.Println("do this")
	}
	retIsMasked := t.IsMasked()
	for _, ot := range others {
		retIsMasked = retIsMasked || ot.IsMasked()
	}
	for i := 0; i < batches; i++ {
		isMasked := t.IsMasked()
		var j int
		for j = 0; j < axisStride; j++ {
			id, ok := <-ch
			if !ok {
				break
			}
			data = append(data, t.strings()[id])
			if isMasked {
				mask = append(mask, t.mask[id])
			}
		}
		if retIsMasked && (!isMasked) {
			mask = append(mask, make([]bool, j)...)
		}

		var ot *Dense
		for j, ot = range others {
			isMasked = ot.IsMasked()
			var k int
			for k = 0; k < axisStride; k++ {
				id, ok := <-chs[j]
				if !ok {
					break
				}
				data = append(data, ot.strings()[id])
				if isMasked {
					mask = append(mask, ot.mask[id])
				}
			}
			if retIsMasked && (!isMasked) {
				mask = append(mask, make([]bool, k)...)
			}
		}
	}
	retVal.mask = mask
}

func (t *Dense) doViewStack(retVal *Dense, axisStride, batches int, ch chan int, others []*Dense, chs []chan int) {
	switch t.t.Kind() {
	case reflect.Bool:
		t.doViewStackB(retVal, axisStride, batches, ch, others, chs)
	case reflect.Int:
		t.doViewStackI(retVal, axisStride, batches, ch, others, chs)
	case reflect.Int8:
		t.doViewStackI8(retVal, axisStride, batches, ch, others, chs)
	case reflect.Int16:
		t.doViewStackI16(retVal, axisStride, batches, ch, others, chs)
	case reflect.Int32:
		t.doViewStackI32(retVal, axisStride, batches, ch, others, chs)
	case reflect.Int64:
		t.doViewStackI64(retVal, axisStride, batches, ch, others, chs)
	case reflect.Uint:
		t.doViewStackU(retVal, axisStride, batches, ch, others, chs)
	case reflect.Uint8:
		t.doViewStackU8(retVal, axisStride, batches, ch, others, chs)
	case reflect.Uint16:
		t.doViewStackU16(retVal, axisStride, batches, ch, others, chs)
	case reflect.Uint32:
		t.doViewStackU32(retVal, axisStride, batches, ch, others, chs)
	case reflect.Uint64:
		t.doViewStackU64(retVal, axisStride, batches, ch, others, chs)
	case reflect.Float32:
		t.doViewStackF32(retVal, axisStride, batches, ch, others, chs)
	case reflect.Float64:
		t.doViewStackF64(retVal, axisStride, batches, ch, others, chs)
	case reflect.Complex64:
		t.doViewStackC64(retVal, axisStride, batches, ch, others, chs)
	case reflect.Complex128:
		t.doViewStackC128(retVal, axisStride, batches, ch, others, chs)
	case reflect.String:
		t.doViewStackStr(retVal, axisStride, batches, ch, others, chs)
	default:
		var index int
		retIsMasked := t.IsMasked()
		mask := retVal.mask[:0]
		for _, ot := range others {
			retIsMasked = retIsMasked || ot.IsMasked()
		}
		for i := 0; i < batches; i++ {
			isMasked := t.IsMasked()
			var j int
			for j = 0; j < axisStride; j++ {
				id, ok := <-ch
				if !ok {
					break
				}
				retVal.Set(index, t.Get(id))
				index++
				if isMasked {
					mask = append(mask, t.mask[id])
				}
			}
			if retIsMasked && (!isMasked) {
				mask = append(mask, make([]bool, j)...)
			}
			var ot *Dense
			for j, ot = range others {
				isMasked = ot.IsMasked()
				var k int
				for k = 0; k < axisStride; k++ {
					id, ok := <-chs[j]
					if !ok {
						break
					}
					retVal.Set(index, ot.Get(id))
					index++
					if isMasked {
						mask = append(mask, ot.mask[id])
					}
				}
				if retIsMasked && (!isMasked) {
					mask = append(mask, make([]bool, k)...)
				}
			}
		}
		retVal.mask = mask
	}
}
