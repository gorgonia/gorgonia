package gorgonia

import "testing"

var overlapsTests = []struct {
	a, b     memblock
	overlaps bool
}{
	{memblock{0, 1024}, memblock{1024, 2048}, false},
	{memblock{1024, 2048}, memblock{0, 1024}, false},
	{memblock{0, 1024}, memblock{0, 512}, true},
	{memblock{0, 1024}, memblock{512, 1024}, true},
	{memblock{512, 2048}, memblock{0, 1024}, true},
}

func TestMemblock_overlaps(t *testing.T) {
	for i, ot := range overlapsTests {
		if ot.a.overlaps(ot.b) != ot.overlaps {
			t.Errorf("Test %d Expected Overlap: %v. Got opposite result", i, ot.overlaps)
		}
	}
}

var memblockLTTests = []struct {
	a, b memblock
	lt   bool
}{
	{memblock{0, 1024}, memblock{2048, 1024}, true},
	{memblock{2048, 1024}, memblock{0, 1024}, false},

	// corner cases - I'm unsure of what the correct result should be for now
	{memblock{0, 1024}, memblock{1024, 1024}, false},
	{memblock{1024, 1024}, memblock{0, 1024}, false},

	// overlaps
	{memblock{0, 1024}, memblock{0, 512}, false},
}

func TestMemblock_lt(t *testing.T) {
	for i, ltt := range memblockLTTests {
		if ltt.a.lt(ltt.b) != ltt.lt {
			t.Errorf("Test %d expected to be lt: %v. Got opposite result", i, ltt.lt)
		}
	}
}

func TestBFC(t *testing.T) {
	align := int64(32)
	bfc := newBFC(align)
	bfc.size = 1024

	_, err := bfc.alloc(0)
	if err == nil {
		t.Error(err)
	}

	// smaller than alignment
	var addr0 uintptr
	if addr0, err = bfc.alloc(21); err != nil {
		t.Error(err)
	}
	if addr0%uintptr(align) != 0 {
		t.Error("Expected all memories to be well aligned")
	}
	if bfc.freelist.Len() != 2 {
		t.Errorf("Expected the free list to have 2 elements")
	}
	t.Logf("%v", bfc.freelist)

	// exactly the size of the alignment
	var addr1 uintptr
	if addr1, err = bfc.alloc(align); err != nil {
		t.Error(err)
	}
	if addr1%uintptr(align) != 0 {
		t.Error("Expected all memories to be well aligned")
	}
	if bfc.freelist.Len() != 2 {
		t.Errorf("Expected the free list to have 2 elements")
	}

	// larger than alignment
	var addr2 uintptr
	if addr2, err = bfc.alloc(69); err != nil {
		t.Error(err)
	}
	if addr2%uintptr(align) != 0 {
		t.Error("Expected all memories to be well aligned")
	}
	if bfc.freelist.Len() != 3 {
		t.Error("Expected free list to be size of 3")
	}

	// free memory
	bfc.free(addr1)
	if len(bfc.bestFits[align]) != 1 {
		t.Error("Expected BFC bestfits to have only one entry")
	}
	if bfc.freelist.Len() != 4 {
		t.Error("Expected free list to be size of 4")
	}

	// allocate again, same size as addr1
	var addr3 uintptr
	if addr3, err = bfc.alloc(32); err != nil {
		t.Error(err)
	}

	if addr1 != addr3 {
		t.Errorf("Expected addr1 to be reused")
	}

	if len(bfc.bestFits[32]) != 0 {
		t.Error("Expected BFC Bestfits for size 32 to have 0 elemenets")
	}
	if bfc.freelist.Len() != 3 {
		t.Error("Expected free list to be size of 3 after reusing a block")
	}

	// memory's getting fragmented now... let's coalesce
	bfc.free(addr3)
	bfc.free(addr2)
	t.Logf("%v", bfc.freelist)
	bfc.coalesce()
	t.Logf("post coalesce: %v", bfc.freelist)
}
