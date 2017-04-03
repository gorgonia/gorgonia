package gorgonia

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestIntervalMethods(t *testing.T) {
	assert := assert.New(t)
	iv := newInterval()

	iv.setFrom(2)

	if !iv.noUsePositions() {
		t.Error("interval has no useposition. How did a usePos get in?")
	}

	//simulate an register being defined at instruction 2, and its last use was 7
	iv.addRange(2, 8)
	iv.addUsePositions(6)
	assert.Equal([]int{6}, iv.usePositions)
	assert.Equal([]intervalRange{{2, 8}}, iv.ranges)

	// now comes a new player... it essentially uses the same data in the same register as iv
	// but was defined a few instructions down the road.
	iv2 := newInterval()
	iv2.addRange(20, 25)
	iv2.addUsePositions(22)

	iv.merge(iv2)
	assert.Equal([]int{6, 22}, iv.usePositions)
	assert.Equal(iv2.end, iv.end)
	assert.Equal(2, iv.start)
	assert.Equal([]intervalRange{iv.ranges[0], iv2.ranges[0]}, iv.ranges)
}

func TestRegAlloc(t *testing.T) {
	var sorted Nodes
	var err error

	g, x, y, z := simpleVecEqn()
	z2 := Must(Square(z))
	if sorted, err = Sort(g); err != nil {
		t.Fatal(err)
	}
	reverseNodes(sorted)

	df := analyze(g, sorted)
	df.buildIntervals(sorted)
	is := df.intervals

	ra := newRegalloc(df)
	ra.alloc(sorted)

	if is[x].result.id >= len(is) {
		t.Error("x is an input, and would have a lifetime of the entire program")
	}

	if is[y].result.id >= len(is) {
		t.Error("y is an input, and would have a lifetime of the entire program")
	}

	var onDev bool
	switch z2.op.(type) {
	case CUDADoer:
		onDev = true
	case CLDoer:
		onDev = true
	}

	switch {
	case z2.op.CallsExtern() && !onDev:
		if is[z].result.id == is[z2].result.id {
			t.Error("z2 should NOT reuse the register of z")
		}
	default:
		if is[z].result.id != is[z2].result.id {
			t.Error("z2 should reuse the register of z")
		}
	}

}
