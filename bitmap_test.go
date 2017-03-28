package gorgonia

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestBitMap(t *testing.T) {
	assert := assert.New(t)
	bm := newBitmap(64)
	assert.Equal(1, len(bm.n))

	track := uint64(0)
	for i := 0; i < 64; i++ {
		bm.Set(i)
		track |= uint64(1) << uint64(i)
		assert.Equal(track, bm.n[0])
		assert.Equal(true, bm.IsSet(i))
		if i < 63 {
			assert.Equal(false, bm.IsSet(i+1))
		} else {
			fails := func() {
				bm.IsSet(i + 1)
			}
			assert.Panics(fails)
		}
	}

	for i := 0; i < 64; i++ {
		bm.Clear(i)
		track &= ^(uint64(1) << uint64(i))
		assert.Equal(track, bm.n[0])
		assert.Equal(false, bm.IsSet(i))
	}

	bm = newBitmap(124)
	assert.Equal(2, len(bm.n))

	track0 := uint64(0)
	track1 := uint64(0)
	for i := 0; i < 128; i++ {
		if i < 124 {
			bm.Set(i)
		} else {
			fails := func() {
				bm.Set(i)
			}
			assert.Panics(fails)
		}
		if i < 64 {
			track0 |= uint64(1) << uint64(i)
			assert.Equal(track0, bm.n[0])
			assert.Equal(true, bm.IsSet(i))
		} else if i > 123 {
			fails := func() {
				bm.IsSet(i)
			}
			assert.Panics(fails)
		} else {
			track1 |= uint64(1) << uint64(i-64)
			assert.Equal(track1, bm.n[1])
			assert.Equal(true, bm.IsSet(i))
		}

		if i < 123 {
			assert.Equal(false, bm.IsSet(i+1))
		} else {
			fails := func() {
				bm.IsSet(i + 1)
			}
			assert.Panics(fails)
		}
	}

	for i := 48; i < 70; i++ {
		bm.Clear(i)
	}

	for i := 48; i < 70; i++ {
		assert.Equal(false, bm.IsSet(i))
	}

	fails := func() {
		bm.Clear(125)
	}
	assert.Panics(fails)

	// idiots section!
	bm = newBitmap(3)
	fails = func() {
		bm.Set(-1)
	}
	assert.Panics(fails)

	fails = func() {
		bm.Set(3)
	}
	assert.Panics(fails)

}
