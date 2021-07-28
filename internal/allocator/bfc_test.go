package allocator

import "testing"

var overlapsTests = []struct {
	a, b     *memblock
	overlaps bool
}{
	{newMemblock(0, 1024), newMemblock(1024, 2048), false},
	{newMemblock(1024, 2048), newMemblock(0, 1024), false},
	{newMemblock(0, 1024), newMemblock(0, 512), true},
	{newMemblock(0, 1024), newMemblock(512, 1024), true},
	{newMemblock(512, 2048), newMemblock(0, 1024), true},
}

func TestMemblock_overlaps(t *testing.T) {
	for i, ot := range overlapsTests {
		if ot.a.overlaps(ot.b) != ot.overlaps {
			t.Errorf("Test %d Expected Overlap: %v. Got opposite result", i, ot.overlaps)
		}
	}
}

var memblockLTTests = []struct {
	a, b *memblock
	lt   bool
}{
	{newMemblock(0, 1024), newMemblock(2048, 1024), true},
	{newMemblock(2048, 1024), newMemblock(0, 1024), false},

	// corner cases - I'm unsure of what the correct result should be for now
	{newMemblock(0, 1024), newMemblock(1024, 1024), false},
	{newMemblock(1024, 1024), newMemblock(0, 1024), false},

	// overlaps
	{newMemblock(0, 1024), newMemblock(0, 512), false},
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
	bfc := New(align)
	bfc.Reserve(0, 1024)

	_, err := bfc.Alloc(0)
	if err == nil {
		t.Error(err)
	}

	// smaller than alignment
	var addr0 uintptr
	if addr0, err = bfc.Alloc(21); err != nil {
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
	if addr1, err = bfc.Alloc(align); err != nil {
		t.Error(err)
	}
	if addr1%uintptr(align) != 0 {
		t.Error("Expected all memories to be well aligned")
	}
	if bfc.freelist.Len() != 2 {
		t.Errorf("Expected the free list to have 2 elements. Got %v", bfc.freelist)
	}

	// larger than alignment
	var addr2 uintptr
	if addr2, err = bfc.Alloc(69); err != nil {
		t.Error(err)
	}
	if addr2%uintptr(align) != 0 {
		t.Error("Expected all memories to be well aligned")
	}
	if bfc.freelist.Len() != 3 {
		t.Error("Expected free list to be size of 3")
	}

	// free memory
	bfc.Free(addr1)
	if bfc.freelist.Len() != 4 {
		t.Error("Expected free list to be size of 4")
	}

	// allocate again, same size as addr1
	var addr3 uintptr
	if addr3, err = bfc.Alloc(32); err != nil {
		t.Error(err)
	}
	t.Logf("addr3 %v", addr3)

	if addr1 != addr3 {
		t.Errorf("Expected addr1 to be reused")
	}
	if bfc.freelist.Len() != 3 {
		t.Errorf("Expected free list to be size of 3 after reusing a block. Got %v", bfc.freelist)
	}

	// memory's getting fragmented now... let's coalesce
	bfc.Free(addr3)
	bfc.Free(addr2)
	t.Logf("pre coalesce: %v", bfc.freelist)
	bfc.coalesce()
	t.Logf("post coalesce: %v", bfc.freelist)
}

func TestBFC_coalesce(t *testing.T) {
	t.SkipNow()
	b := New(32)
	b.Reserve(0, 114080)

	// yanked from a failing real example
	list := []*memblock{
		newMemblock(3280, 16),
		newMemblock(3376, 16),
		newMemblock(3392, 8),
		newMemblock(3400, 24),
		newMemblock(3424, 64),
		newMemblock(3472, 16),
		newMemblock(3488, 110808),
	}
	for i, b := range list {
		if i == len(list)-1 {
			break
		}
		b.next = list[i+1]
		list[i+1].prev = b
	}
	b.freelist.first = list[0]
	b.freelist.last = list[len(list)-1]

	b.coalesce()
	t.Logf("%v", b.freelist)
}
