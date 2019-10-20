package gorgonia

import "math/bits"

const (
	bitmapBits = 64
)

// bitmap is a very simple bitmap. It only supports Set, IsSet and Clear methods. It's mostly used for tracking which element has been set
type bitmap struct {
	n   []uint64
	max int
}

// newBitmap creates a new bitmap.
func newBitmap(size int) *bitmap {
	q, r := divmod(size, bitmapBits)

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

	block, pos := divmod(i, bitmapBits)
	bm.n[block] |= uint64(1) << uint64(pos)
}

// IsSet returns true if the ith bit is set. It panics if the i is greater or equal to the defined max
func (bm *bitmap) IsSet(i int) bool {
	if i >= bm.max || i < 0 {
		panic("Index out of range")
	}

	block, pos := divmod(i, bitmapBits)
	return bm.n[block]>>uint64(pos)&uint64(1) == uint64(1)
}

// Clear clears the ith bit. It panics if i is greater or equal to the defined max
func (bm *bitmap) Clear(i int) {
	if i >= bm.max || i < 0 {
		panic("Index out of range")
	}

	block, pos := divmod(i, bitmapBits)
	bm.n[block] &= ^(uint64(1) << uint64(pos))
}

// BlocksWithZero finds the first block with zeroes in the bit. atleast specifies how many consecutive zeroes need be found
func (bm *bitmap) BlocksWithZero(atleast int) int {
	retVal := -1
	for i, b := range bm.n {
		if bits.OnesCount64(b) != bitmapBits {
			// shortcut:
			if bits.LeadingZeros64(b) > atleast {
				return i
			}

			var consecutive int
			for j := 0; j < bitmapBits; j++ {
				if b>>uint64(j)&uint64(1) == 0 {
					consecutive++
				} else {
					consecutive = 0
				}
				if consecutive > atleast {
					return i
				}
			}
		}
	}
	return retVal
}
