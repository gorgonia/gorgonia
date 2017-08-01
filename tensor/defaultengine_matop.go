package tensor

import "github.com/pkg/errors"

func (e StdEng) Transpose(a Tensor, expStrides []int) error {
	if !a.IsNativelyAccessible() {
		return errors.Errorf("Cannot Transpose() on non-natively accessible tensor")
	}
	if dt, ok := a.(DenseTensor); ok {
		e.denseTranspose(dt, expStrides)
		return nil
	}
	return errors.Errorf("Tranpose for tensor of %T not supported", a)
}

func (e StdEng) denseTranspose(a DenseTensor, expStrides []int) {
	switch a.rtype().Size() {
	case 1:
		e.denseTranspose1(a, expStrides)
	case 2:
		e.denseTranspose2(a, expStrides)
	case 4:
		e.denseTranspose4(a, expStrides)
	case 8:
		e.denseTranspose8(a, expStrides)
	default:
		e.denseTransposeArbitrary(a, expStrides)
	}
}

func (e StdEng) denseTranspose1(a DenseTensor, expStrides []int) {
	axes := a.transposeAxes()
	size := a.len()

	// first we'll create a bit-map to track which elements have been moved to their correct places
	track := NewBitMap(size)
	track.Set(0)
	track.Set(size - 1) // first and last element of a transposedon't change

	var saved, tmp byte
	var i int

	data := a.hdr().Uint8s()
	for i = 1; ; {
		dest := a.transposeIndex(i, axes, expStrides)

		if track.IsSet(i) && track.IsSet(dest) {
			data[i] = saved
			saved = 0
			for i < size && track.IsSet(i) {
				i++
			}
			if i >= size {
				break
			}
			continue
		}
		track.Set(i)
		tmp = data[i]
		data[i] = saved
		saved = tmp

		i = dest
	}
}

func (e StdEng) denseTranspose2(a DenseTensor, expStrides []int) {
	axes := a.transposeAxes()
	size := a.len()

	// first we'll create a bit-map to track which elements have been moved to their correct places
	track := NewBitMap(size)
	track.Set(0)
	track.Set(size - 1) // first and last element of a transposedon't change

	var saved, tmp uint16
	var i int

	data := a.hdr().Uint16s()
	for i = 1; ; {
		dest := a.transposeIndex(i, axes, expStrides)

		if track.IsSet(i) && track.IsSet(dest) {
			data[i] = saved
			saved = 0
			for i < size && track.IsSet(i) {
				i++
			}
			if i >= size {
				break
			}
			continue
		}
		track.Set(i)
		tmp = data[i]
		data[i] = saved
		saved = tmp

		i = dest
	}
}

func (e StdEng) denseTranspose4(a DenseTensor, expStrides []int) {
	axes := a.transposeAxes()
	size := a.len()

	// first we'll create a bit-map to track which elements have been moved to their correct places
	track := NewBitMap(size)
	track.Set(0)
	track.Set(size - 1) // first and last element of a transposedon't change

	var saved, tmp uint32
	var i int

	data := a.hdr().Uint32s()
	for i = 1; ; {
		dest := a.transposeIndex(i, axes, expStrides)

		if track.IsSet(i) && track.IsSet(dest) {
			data[i] = saved
			saved = 0
			for i < size && track.IsSet(i) {
				i++
			}
			if i >= size {
				break
			}
			continue
		}
		track.Set(i)
		tmp = data[i]
		data[i] = saved
		saved = tmp

		i = dest
	}
}

func (e StdEng) denseTranspose8(a DenseTensor, expStrides []int) {
	axes := a.transposeAxes()
	size := a.len()

	// first we'll create a bit-map to track which elements have been moved to their correct places
	track := NewBitMap(size)
	track.Set(0)
	track.Set(size - 1) // first and last element of a transposedon't change

	var saved, tmp uint64
	var i int

	data := a.hdr().Uint64s()
	for i = 1; ; {
		dest := a.transposeIndex(i, axes, expStrides)

		if track.IsSet(i) && track.IsSet(dest) {
			data[i] = saved
			saved = 0
			for i < size && track.IsSet(i) {
				i++
			}
			if i >= size {
				break
			}
			continue
		}
		track.Set(i)
		tmp = data[i]
		data[i] = saved
		saved = tmp

		i = dest
	}
}

func (e StdEng) denseTransposeArbitrary(a DenseTensor, expStrides []int) {
	axes := a.transposeAxes()
	size := a.len()
	typeSize := int(a.rtype().Size())

	// first we'll create a bit-map to track which elements have been moved to their correct places
	track := NewBitMap(size)
	track.Set(0)
	track.Set(size - 1) // first and last element of a transposedon't change

	saved := make([]byte, typeSize, typeSize)
	tmp := make([]byte, typeSize, typeSize)
	var i int

	data := a.hdr().Uint8s()
	for i = 1; ; {
		dest := a.transposeIndex(i, axes, expStrides)
		start := typeSize * i

		if track.IsSet(i) && track.IsSet(dest) {
			copy(data[start:start+typeSize], saved)
			for i := range saved {
				saved[i] = 0
			}
			for i < size && track.IsSet(i) {
				i++
			}
			if i >= size {
				break
			}
			continue
		}
		track.Set(i)
		copy(tmp, data[start:start+typeSize])
		copy(data[start:start+typeSize], saved)
		saved = tmp

		i = dest
	}
}
