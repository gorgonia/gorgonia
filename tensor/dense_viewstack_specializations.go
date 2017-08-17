package tensor

import (
	"fmt"
	"log"
	"reflect"
)

/*
GENERATED FILE. DO NOT EDIT
*/

/* bool */

func (t *Dense) doViewStackB(retVal *Dense, axisStride, batches int, ch chan int, others []*Dense, chs []chan int) {
	data := retVal.Bools()[:0]
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
			data = append(data, t.Bools()[id])
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
				data = append(data, ot.Bools()[id])
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
	data := retVal.Ints()[:0]
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
			data = append(data, t.Ints()[id])
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
				data = append(data, ot.Ints()[id])
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
	data := retVal.Int8s()[:0]
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
			data = append(data, t.Int8s()[id])
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
				data = append(data, ot.Int8s()[id])
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
	data := retVal.Int16s()[:0]
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
			data = append(data, t.Int16s()[id])
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
				data = append(data, ot.Int16s()[id])
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
	data := retVal.Int32s()[:0]
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
			data = append(data, t.Int32s()[id])
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
				data = append(data, ot.Int32s()[id])
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
	data := retVal.Int64s()[:0]
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
			data = append(data, t.Int64s()[id])
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
				data = append(data, ot.Int64s()[id])
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
	data := retVal.Uints()[:0]
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
			data = append(data, t.Uints()[id])
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
				data = append(data, ot.Uints()[id])
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
	data := retVal.Uint8s()[:0]
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
			data = append(data, t.Uint8s()[id])
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
				data = append(data, ot.Uint8s()[id])
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
	data := retVal.Uint16s()[:0]
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
			data = append(data, t.Uint16s()[id])
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
				data = append(data, ot.Uint16s()[id])
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
	data := retVal.Uint32s()[:0]
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
			data = append(data, t.Uint32s()[id])
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
				data = append(data, ot.Uint32s()[id])
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
	data := retVal.Uint64s()[:0]
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
			data = append(data, t.Uint64s()[id])
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
				data = append(data, ot.Uint64s()[id])
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
	data := retVal.Float32s()[:0]
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
			data = append(data, t.Float32s()[id])
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
				data = append(data, ot.Float32s()[id])
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
	log.Printf("retVal %v", retVal.Float64s())
	log.Printf("Ch: %d, %d", len(ch), cap(ch))
	data := retVal.Float64s()[:0]
	log.Printf("data %v", data)
	mask := retVal.mask[:0]
	if t.IsMasked() {
		fmt.Println("do this")
	}
	//defer func() {
	//	if r := recover(); r != nil {
	//		log.Printf("dat %v | %v", data, t.Float64s())
	//		log.Printf("%#v", r)
	//	}
	//}()
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
			log.Printf("id %d %x  (%d, %d) | %d ", id, id, len(ch), cap(ch), len(t.Float64s()))
			data = append(data, t.Float64s()[id])
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
				data = append(data, ot.Float64s()[id])
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
	data := retVal.Complex64s()[:0]
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
			data = append(data, t.Complex64s()[id])
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
				data = append(data, ot.Complex64s()[id])
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
	data := retVal.Complex128s()[:0]
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
			data = append(data, t.Complex128s()[id])
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
				data = append(data, ot.Complex128s()[id])
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
	data := retVal.Strings()[:0]
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
			data = append(data, t.Strings()[id])
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
				data = append(data, ot.Strings()[id])
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
