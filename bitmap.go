package gorgonia

// bitmap is a very simple bitmap. It only supports Set, IsSet and Clear methods. It's mostly used for tracking which element has been set
type bitmap struct {
	n   []uint64
	max int
}

// newBitmap creates a new bitmap.
func newBitmap(size int) *bitmap {
	q, r := divmod(size, 64)

	if r > 0 {
		q++
	}

	return &bitmap{
		n:   make([]uint64, q),
		max: size,
	}
}

// Set sets the ith bit of the bit map to 1. It panics if i is greater or equal to the defined max
func (bm *bitmap) Set(i int) {
	if i >= bm.max || i < 0 {
		panic("Index out of range")
	}

	block, pos := divmod(i, 64)
	bm.n[block] |= uint64(1) << uint64(pos)
}

// IsSet returns true if the ith bit is set. It panics if the i is greater or equal to the defined max
func (bm *bitmap) IsSet(i int) bool {
	if i >= bm.max || i < 0 {
		panic("Index out of range")
	}

	block, pos := divmod(i, 64)
	return bm.n[block]>>uint64(pos)&uint64(1) == uint64(1)
}

// Clear clears the ith bit. It panics if i is greater or equal to the defined max
func (bm *bitmap) Clear(i int) {
	if i >= bm.max || i < 0 {
		panic("Index out of range")
	}

	block, pos := divmod(i, 64)
	bm.n[block] &= ^(uint64(1) << uint64(pos))
}
