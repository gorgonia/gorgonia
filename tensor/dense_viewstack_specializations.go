package tensor

import "github.com/pkg/errors"

// func (e StdEng) Stack(t Tensor, axis int, others ...Tensor) (retVal Tensor, err error) {
// 	// checks

// }

func (e StdEng) StackDense(t DenseTensor, axis int, others ...DenseTensor) (retVal DenseTensor, err error) {
	opdims := t.Dims()
	if axis >= opdims+1 {
		err = errors.Errorf(dimMismatch, opdims+1, axis)
		return
	}

	newShape := Shape(BorrowInts(opdims + 1))
	newShape[axis] = len(others) + 1
	shape := t.Shape()
	var cur int
	for i, s := range shape {
		if i == axis {
			cur++
		}
		newShape[cur] = s
		cur++
	}

	info := t.Info()
	var newStrides []int
	if info.o.isColMajor() {
		newStrides = newShape.calcStridesColMajor()
	} else {
		newStrides = newShape.calcStrides()

	}
	ap := NewAP(newShape, newStrides)
	ap.o = info.o
	ap.Δ = info.Δ

	allNoMat := !requiresIterator(t)
	for _, ot := range others {
		if allNoMat && requiresIterator(ot) {
			allNoMat = false
		}
	}

	retVal = recycledDense(t.Dtype(), ap.Shape(), WithEngine(e))
	ReturnAP(retVal.Info())
	retVal.setAP(ap)

	// the "viewStack" method is the more generalized method
	// and will work for all Tensors, regardless of whether it's a view
	// But the simpleStack is faster, and is an optimization

	if allNoMat {
		retVal = e.denseSimpleStack(t, retVal, axis, others)
	} else {
		retVal, err = e.denseViewStack(t, retVal, axis, others)
	}
	return
}

func (e StdEng) denseSimpleStack(t, retVal DenseTensor, axis int, others []DenseTensor) DenseTensor {
	switch axis {
	case 0:
		copyDense(retVal, t)
		next := t.len()
		for _, ot := range others {
			copyDenseSliced(retVal, next, retVal.len(), ot, 0, ot.len())
			next += ot.len()
		}
	default:
		axisStride := retVal.Info().Strides()[axis]
		batches := retVal.len() / axisStride

		destStart := 0
		start := 0
		end := start + axisStride

		for i := 0; i < batches; i++ {
			copyDenseSliced(retVal, destStart, retVal.len(), t, start, end)
			for _, ot := range others {
				destStart += axisStride
				copyDenseSliced(retVal, destStart, retVal.len(), ot, start, end)
				i++
			}
			destStart += axisStride
			start += axisStride
			end += axisStride
		}
	}
	return retVal
}

func (e StdEng) denseViewStack(t, retVal DenseTensor, axis int, others []DenseTensor) (DenseTensor, error) {
	axisStride := retVal.Info().Strides()[axis]
	batches := retVal.len() / axisStride

	it := IteratorFromDense(t)
	its := make([]Iterator, 0, len(others))
	for _, ot := range others {
		oter := IteratorFromDense(ot)
		its = append(its, oter)
	}

	err := e.doViewStack(t, retVal, axisStride, batches, it, others, its)
	return retVal, err
}

func (e StdEng) doViewStack(t, retVal DenseTensor, axisStride, batches int, it Iterator, others []DenseTensor, its []Iterator) error {
	switch int(t.Dtype().Size()) {
	case 1:
		return e.doViewStack1(t, retVal, axisStride, batches, it, others, its)
	case 2:
		return e.doViewStack2(t, retVal, axisStride, batches, it, others, its)
	case 4:
		return e.doViewStack4(t, retVal, axisStride, batches, it, others, its)
	case 8:
		return e.doViewStack8(t, retVal, axisStride, batches, it, others, its)
	default:
		return errors.Errorf(methodNYI, "doviewStack")
	}
}

func (e StdEng) doViewStack1(t, retVal DenseTensor, axisStride, batches int, it Iterator, others []DenseTensor, its []Iterator) (err error) {
	data := retVal.hdr().Uint8s()[:0]
	var mask []bool
	var retIsMasked bool
	if mt, ok := t.(MaskedTensor); ok {
		retIsMasked = mt.IsMasked()
	}
	for _, ot := range others {
		if mt, ok := ot.(MaskedTensor); ok {
			retIsMasked = retIsMasked || mt.IsMasked()
		}
	}

	f := func(t DenseTensor, it Iterator) (last int, isMasked bool, err error) {
		var tmask []bool
		if mt, ok := t.(MaskedTensor); ok {
			tmask = mt.Mask()
			isMasked = mt.IsMasked()
		}

		for last = 0; last < axisStride; last++ {
			id, err := it.Next()
			if handleNoOp(err) != nil {
				return -1, isMasked, errors.Wrap(err, "doviewStackfailed")
			}
			if err != nil {
				break
			}
			data = append(data, t.hdr().Uint8s()[id])
			if isMasked {
				mask = append(mask, tmask[id])
			}
		}
		return
	}

	for i := 0; i < batches; i++ {
		var last int
		var isMasked bool
		if last, isMasked, err = f(t, it); err != nil {
			return
		}
		if retIsMasked && (!isMasked) {
			mask = append(mask, make([]bool, last)...)
		}
		for j, ot := range others {
			if last, isMasked, err = f(ot, its[j]); err != nil {
				return
			}
			if retIsMasked && (!isMasked) {
				mask = append(mask, make([]bool, last)...)
			}
		}
	}

	if mt, ok := retVal.(MaskedTensor); ok {
		mt.SetMask(mask)
	}
	return nil
}

func (e StdEng) doViewStack2(t, retVal DenseTensor, axisStride, batches int, it Iterator, others []DenseTensor, its []Iterator) (err error) {
	data := retVal.hdr().Uint16s()[:0]
	var mask []bool
	var retIsMasked bool
	if mt, ok := t.(MaskedTensor); ok {
		retIsMasked = mt.IsMasked()
	}
	for _, ot := range others {
		if mt, ok := ot.(MaskedTensor); ok {
			retIsMasked = retIsMasked || mt.IsMasked()
		}
	}

	f := func(t DenseTensor, it Iterator) (last int, isMasked bool, err error) {
		var tmask []bool
		if mt, ok := t.(MaskedTensor); ok {
			tmask = mt.Mask()
			isMasked = mt.IsMasked()
		}

		for last = 0; last < axisStride; last++ {
			id, err := it.Next()
			if handleNoOp(err) != nil {
				return -1, isMasked, errors.Wrap(err, "doviewStackfailed")
			}
			if err != nil {
				break
			}
			data = append(data, t.hdr().Uint16s()[id])
			if isMasked {
				mask = append(mask, tmask[id])
			}
		}
		return
	}

	for i := 0; i < batches; i++ {
		var last int
		var isMasked bool
		if last, isMasked, err = f(t, it); err != nil {
			return
		}
		if retIsMasked && (!isMasked) {
			mask = append(mask, make([]bool, last)...)
		}
		for j, ot := range others {
			if last, isMasked, err = f(ot, its[j]); err != nil {
				return
			}
			if retIsMasked && (!isMasked) {
				mask = append(mask, make([]bool, last)...)
			}
		}
	}

	if mt, ok := retVal.(MaskedTensor); ok {
		mt.SetMask(mask)
	}
	return nil
}

func (e StdEng) doViewStack4(t, retVal DenseTensor, axisStride, batches int, it Iterator, others []DenseTensor, its []Iterator) (err error) {
	data := retVal.hdr().Uint32s()[:0]
	var mask []bool
	var retIsMasked bool
	if mt, ok := t.(MaskedTensor); ok {
		retIsMasked = mt.IsMasked()
	}
	for _, ot := range others {
		if mt, ok := ot.(MaskedTensor); ok {
			retIsMasked = retIsMasked || mt.IsMasked()
		}
	}

	f := func(t DenseTensor, it Iterator) (last int, isMasked bool, err error) {
		var tmask []bool
		if mt, ok := t.(MaskedTensor); ok {
			tmask = mt.Mask()
			isMasked = mt.IsMasked()
		}

		for last = 0; last < axisStride; last++ {
			id, err := it.Next()
			if handleNoOp(err) != nil {
				return -1, isMasked, errors.Wrap(err, "doviewStackfailed")
			}
			if err != nil {
				break
			}
			data = append(data, t.hdr().Uint32s()[id])
			if isMasked {
				mask = append(mask, tmask[id])
			}
		}
		return
	}

	for i := 0; i < batches; i++ {
		var last int
		var isMasked bool
		if last, isMasked, err = f(t, it); err != nil {
			return
		}
		if retIsMasked && (!isMasked) {
			mask = append(mask, make([]bool, last)...)
		}
		for j, ot := range others {
			if last, isMasked, err = f(ot, its[j]); err != nil {
				return
			}
			if retIsMasked && (!isMasked) {
				mask = append(mask, make([]bool, last)...)
			}
		}
	}

	if mt, ok := retVal.(MaskedTensor); ok {
		mt.SetMask(mask)
	}
	return nil
}

func (e StdEng) doViewStack8(t, retVal DenseTensor, axisStride, batches int, it Iterator, others []DenseTensor, its []Iterator) (err error) {
	data := retVal.hdr().Uint64s()[:0]
	var mask []bool
	var retIsMasked bool
	if mt, ok := t.(MaskedTensor); ok {
		retIsMasked = mt.IsMasked()
	}
	for _, ot := range others {
		if mt, ok := ot.(MaskedTensor); ok {
			retIsMasked = retIsMasked || mt.IsMasked()
		}
	}

	f := func(t DenseTensor, it Iterator) (last int, isMasked bool, err error) {
		var tmask []bool
		if mt, ok := t.(MaskedTensor); ok {
			tmask = mt.Mask()
			isMasked = mt.IsMasked()
		}

		for last = 0; last < axisStride; last++ {
			id, err := it.Next()
			if handleNoOp(err) != nil {
				return -1, isMasked, errors.Wrap(err, "doviewStackfailed")
			}
			if err != nil {
				break
			}
			data = append(data, t.hdr().Uint64s()[id])
			if isMasked {
				mask = append(mask, tmask[id])
			}
		}
		return
	}

	for i := 0; i < batches; i++ {
		var last int
		var isMasked bool
		if last, isMasked, err = f(t, it); err != nil {
			return
		}
		if retIsMasked && (!isMasked) {
			mask = append(mask, make([]bool, last)...)
		}
		for j, ot := range others {
			if last, isMasked, err = f(ot, its[j]); err != nil {
				return
			}
			if retIsMasked && (!isMasked) {
				mask = append(mask, make([]bool, last)...)
			}
		}
	}

	if mt, ok := retVal.(MaskedTensor); ok {
		mt.SetMask(mask)
	}
	return nil
}

// func (t *Dense) doViewStack(retVal *Dense, axisStride, batches int, ch chan int, others []*Dense, chs []chan int) {

// 	switch t.t.Kind() {
// 	case reflect.Bool:
// 		t.doViewStackB(retVal, axisStride, batches, ch, others, chs)
// 	case reflect.Int:
// 		t.doViewStackI(retVal, axisStride, batches, ch, others, chs)
// 	case reflect.Int8:
// 		t.doViewStackI8(retVal, axisStride, batches, ch, others, chs)
// 	case reflect.Int16:
// 		t.doViewStackI16(retVal, axisStride, batches, ch, others, chs)
// 	case reflect.Int32:
// 		t.doViewStackI32(retVal, axisStride, batches, ch, others, chs)
// 	case reflect.Int64:
// 		t.doViewStackI64(retVal, axisStride, batches, ch, others, chs)
// 	case reflect.Uint:
// 		t.doViewStackU(retVal, axisStride, batches, ch, others, chs)
// 	case reflect.Uint8:
// 		t.doViewStackU8(retVal, axisStride, batches, ch, others, chs)
// 	case reflect.Uint16:
// 		t.doViewStackU16(retVal, axisStride, batches, ch, others, chs)
// 	case reflect.Uint32:
// 		t.doViewStackU32(retVal, axisStride, batches, ch, others, chs)
// 	case reflect.Uint64:
// 		t.doViewStackU64(retVal, axisStride, batches, ch, others, chs)
// 	case reflect.Float32:
// 		t.doViewStackF32(retVal, axisStride, batches, ch, others, chs)
// 	case reflect.Float64:
// 		t.doViewStackF64(retVal, axisStride, batches, ch, others, chs)
// 	case reflect.Complex64:
// 		t.doViewStackC64(retVal, axisStride, batches, ch, others, chs)
// 	case reflect.Complex128:
// 		t.doViewStackC128(retVal, axisStride, batches, ch, others, chs)
// 	case reflect.String:
// 		t.doViewStackStr(retVal, axisStride, batches, ch, others, chs)
// 	default:
// 		var index int
// 		retIsMasked := t.IsMasked()
// 		mask := retVal.mask[:0]
// 		for _, ot := range others {
// 			retIsMasked = retIsMasked || ot.IsMasked()
// 		}
// 		for i := 0; i < batches; i++ {
// 			isMasked := t.IsMasked()
// 			var j int
// 			for j = 0; j < axisStride; j++ {
// 				id, ok := <-ch
// 				if !ok {
// 					break
// 				}
// 				retVal.Set(index, t.Get(id))
// 				index++
// 				if isMasked {
// 					mask = append(mask, t.mask[id])
// 				}
// 			}
// 			if retIsMasked && (!isMasked) {
// 				mask = append(mask, make([]bool, j)...)
// 			}
// 			var ot *Dense
// 			for j, ot = range others {
// 				isMasked = ot.IsMasked()
// 				var k int
// 				for k = 0; k < axisStride; k++ {
// 					id, ok := <-chs[j]
// 					if !ok {
// 						break
// 					}
// 					retVal.Set(index, ot.Get(id))
// 					index++
// 					if isMasked {
// 						mask = append(mask, ot.mask[id])
// 					}
// 				}
// 				if retIsMasked && (!isMasked) {
// 					mask = append(mask, make([]bool, k)...)
// 				}
// 			}
// 		}
// 		retVal.mask = mask
// 	}
// }
