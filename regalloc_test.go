package gorgonia

import (
	"bytes"
	"fmt"
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

func TestBuildIntervals(t *testing.T) {
	assert := assert.New(t)
	var err error
	g, x, y, z := simpleVecEqn()

	var readVal Value
	r := Read(z, &readVal)

	z2 := Must(Square(z))
	z2y := Must(HadamardProd(z2, y))

	// because sorting is unstable, we need to test many times
	var sorted Nodes
	var intervals map[*Node]*interval
	for i := 0; i < 100; i++ {
		if sorted, err = Sort(g); err != nil {
			t.Fatal(err)
		}
		reverseNodes(sorted)
		intervals = buildIntervals(sorted)

		df := newdataflow()
		df.intervals = intervals
		df.debugIntervals(sorted)

		// inputs are live until the last instruction
		assert.Equal(len(intervals), intervals[x].end, "%v", len(sorted))
		if intervals[x].start != 1 && intervals[x].start != 0 {
			t.Errorf("x starts at 1 or 0 (depending on how the sort allocates it)")
		}

		assert.Equal(len(g.AllNodes()), intervals[y].end)
		if intervals[y].start != 1 && intervals[y].start != 0 {
			t.Errorf("y starts at 1 or 0 (depending on how the sort allocates it)")
		}

		assert.Equal(2, intervals[z].start)
		if intervals[z2].start > intervals[z].end {
			t.Error("z2 should start before z ends")
		}

		assert.Equal(intervals[r].start, intervals[r].end)
		if intervals[r].start < intervals[z].start {
			t.Error("z should have an earlier start than r")
		}
		if intervals[r].start > intervals[z].end {
			t.Error("z should end before r starts (or at the same as r start")
		}

		if intervals[z2].end <= intervals[z2].start {
			t.Error("Given that z2y uses z2, the intervals should not end at the same as its start")
		}
		if intervals[z2].start < intervals[z].start {
			t.Error("z should have an earlier start than z2")
		}
		if intervals[z2].start > intervals[z].end {
			t.Error("z should end before r starts (or at the same as z2 start")
		}

		assert.Equal(intervals[z2y].start, intervals[z2y].end)
		if intervals[z2y].start < intervals[z2].start {
			t.Error("z2 should have an earlier start than z2y")
		}
		if intervals[z2y].start > intervals[z2].end {
			t.Error("z2 should end before r starts (or at the same as z2y start")
		}

		if t.Failed() {
			break
		}

	}

	// visual reminder
	var buf bytes.Buffer
	buf.WriteString("VISUAL REMINDER OF INTERVALS\n")
	sorted.reverse()
	for i, n := range sorted {
		in := intervals[n]
		fmt.Fprintf(&buf, "%d\t%v\tfrom %v to %v \n", i, n, in.start, in.end)

	}
	t.Log(buf.String())
}

func TestRegAlloc(t *testing.T) {
	var sorted Nodes
	var err error

	g, x, y, z := simpleVecEqn()
	z2 := Must(Square(z))
	if sorted, err = Sort(g); err != nil {
		t.Fatal(err)
	}

	is := buildIntervals(sorted)
	df := analyze(g, sorted)

	df.intervals = is
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
