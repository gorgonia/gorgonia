package gorgonia

import (
	"bytes"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestBuildIntervals(t *testing.T) {
	assert := assert.New(t)
	var err error
	g, x, y, z := simpleVecEqn()
	var readVal Value
	r := Read(z, &readVal)

	z2 := Must(Square(z))
	z2y := Must(HadamardProd(z2, y, 0))
	c := NewConstant(1.0, WithName("FOOO")) // const
	g.addToAll(c)                           // this is a hack because there is no good way to get a constant into a graph since In() won't work on constatns

	// because sorting is unstable, we need to test many times
	var sorted Nodes
	var intervals map[*Node]*interval

	for i := 0; i < 100; i++ {
		if sorted, err = Sort(g); err != nil {
			t.Fatal(err)
		}
		reverseNodes(sorted)

		df := analyze(g, sorted)
		df.buildIntervals(sorted)
		df.debugIntervals(sorted) // prints intervals on debug mode
		intervals = df.intervals

		// inputs are live until the last instruction
		assert.Equal(len(intervals), intervals[x].end, "%v", len(sorted))
		if intervals[x].start != 1 && intervals[x].start != 0 {
			t.Errorf("x starts at 1 or 0 (depending on how the sort allocates it)")
		}

		assert.Equal(len(g.AllNodes()), intervals[y].end)
		if intervals[y].start != 1 && intervals[y].start != 0 {
			t.Errorf("y starts at 1 or 0 (depending on how the sort allocates it)")
		}

		// constants should be live until the last instruction
		assert.Equal(len(intervals), intervals[c].end, "%v", len(sorted))

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
