package cuda

import (
	"bytes"
	"fmt"

	"github.com/pkg/errors"
)

const (
	minAllocBits = 8
	minAllocSize = 1 << minAllocBits

	freeAllocTresh = 0.75
)

var nilBlock = memblock{}

// memblock is a tuple of address and the size of the block - think of it as a slicehdr, where the cap is the size
type memblock struct {
	address uintptr
	size    int64

	next, prev *memblock
}

func newMemblock(addr uintptr, size int64) *memblock {
	return &memblock{
		address: addr,
		size:    size,
	}
}

func (a memblock) cap() uintptr { return a.address + uintptr(a.size) }

// overlaps checks if two memblocks are overlapping.
func (a *memblock) overlaps(b *memblock) bool {
	if a == b {
		return true
	}
	if a.address == b.address {
		return true // it doesn't matter how many elements there are in the memory. As long as they start at the same address, they overlap
	}

	capA := a.cap()
	capB := b.cap()

	switch {
	case a.address < b.address:
		if b.address < capA {
			return true
		}
	case a.address > b.address:
		if a.address < capB {
			return true
		}
	}
	return false
}

func (a *memblock) split(size int64) (b *memblock) {
	if size >= a.size {
		allocatorLogf("block %v, size %v", a, size)
		panic("IMPOSSIBLE")
	}
	newAddress := a.address + uintptr(size)
	newSize := a.size - size
	a.size = size
	b = newMemblock(newAddress, newSize)
	return b
}

// we say a memblock is less than another memblock when:
//		a.address < b.address and they don't both overlap
func (a *memblock) lt(b *memblock) bool {
	if a.address == b.address {
		return false
	}

	capA := a.cap()

	if a.address < b.address && capA < b.address {
		return true
	}

	// any other thing is not strictly less than
	return false
}

func (a *memblock) String() string {
	return fmt.Sprintf("{0x%x %d}", a.address, a.size)
}

// freelist is simply implemented as a linkedlist of memblocks
type freelist struct {
	first, last *memblock
	l           int
}

func (l *freelist) Len() int { return l.l }

func (l *freelist) String() string {
	var buf bytes.Buffer
	fmt.Fprintf(&buf, "FIRST: %v, LAST %v | [", l.first, l.last)
	for block := l.first; block != nil; block = block.next {
		fmt.Fprintf(&buf, "%v, ", block)
	}
	fmt.Fprintf(&buf, "]")
	return buf.String()
}

// insert inserts a block in an ordered fashion. This helps with coaalescing.
func (l *freelist) insert(block *memblock) {
	allocatorLogf("Inserting %v", block)
	if l.first == nil {
		l.first = block
		l.last = block

		l.l++
		return
	}
	if block.address >= l.last.address {
		allocatorLogf("greater than last")
		overlaps := block.overlaps(l.last)
		switch {
		case overlaps:
			blockCap := block.cap()
			lastCap := l.last.cap()
			if blockCap < lastCap {
				return
			}
			l.last.size += int64(blockCap - lastCap)
			return
		default:
			l.last.next = block
			block.prev = l.last
			block.next = nil
			l.last = block

			l.l++
			return
		}
	}

	if block.address < l.first.address {
		allocatorLogf("lt first")
		overlaps := block.overlaps(l.first)
		if overlaps {
			blockCap := block.cap()
			firstCap := l.first.cap()
			if firstCap < blockCap {
				return
			}
			l.first.size += int64(blockCap - firstCap)
			return
		}

		l.first.prev = block
		block.next = l.first
		l.first = block
		l.l++
		return
	}

	allocatorLogf("insert block")
insert:
	for b := l.first; b != nil; b = b.next {
		overlaps := b.overlaps(block)
		switch {
		case b.address < block.address && overlaps:
			// coalesce block into b
			blockCap := block.cap()
			bcap := b.cap()
			if blockCap <= bcap {
				return // do nothing, since block is already in b
			}

			newSize := int64(bcap - blockCap)
			b.size += newSize
			return

		case b.address < block.address && !overlaps:
			if b.next == nil {
				allocatorLogf("Uh oh")
				allocatorLogf("b: %v", b)
				allocatorLogf("l %v", l)
			}
			if b.next.address >= block.cap() {
				bnext := b.next
				b.next = block
				block.next = bnext
				block.prev = b
				bnext.prev = block
				l.l++
				return

			}
		case b.address == block.address:
			if b.size > block.size {
				return
			}
			b.size = block.size
			return
		case b.address > block.address && overlaps:
			blockCap := block.cap()
			bcap := b.cap()
			if bcap <= blockCap {
				b.address = block.address
				b.size = block.size
				return
			}
			b.address = block.address
			b.size = block.size + int64(bcap-blockCap)
			return
		case b.address > block.address && !overlaps:
			// gone too far.
			break insert
		default:
			panic("WTF")
		}
	}
	panic("Unreachable")
}

func (l *freelist) remove(block *memblock) {
	allocatorLogf("remove %v from free list", block)
	if l.first == block {
		l.first = block.next
	} else {
		block.prev.next = block.next
	}

	if l.last == block {
		l.last = block.prev
	} else {
		block.next.prev = block.prev
	}

	// cleanup
	block.next = nil
	block.prev = nil
	l.l--
}

// splitOrRemove returns the block that is removed from the list
func (l *freelist) splitOrRemove(block *memblock, aligned, size int64) {
	if block.size > aligned {
		split := block.split(aligned)
		l.insert(split)
	}
	if block.size > size {
		remnant := block.split(size)
		l.insert(remnant)
	}
	l.remove(block)
}

// bfc an accounting structure for memory allocation,
// directly inspired by TensorFlows' Best Fit With Coalescing memory allocator, which is a type of buddy memory allocator.
//
// Why is this needed?
// This allocator is needed because it's been shown that:
//	1. allocating and copying data from Host to Device has in fact taken most amount of time.
//	2. allocating memory on CUDA is a blocking call even on the BatchedContext. This has the effect of making extra cgo calls and is inefficient.
//	3. It's more efficient to just allocate a large block of memory upfront and then manage it internally.
//
// Why does this allocator allocate aligned memory?
// For no reason other than performance. CUDA memory are aligned to 32-byte, 64-byte and 128 byte boundaries.
// While it would be significantly easier to manage memory without alignment, some additional book keeping is worth it for the performance gains.
//
// Why is the freelist just a slice of blocks?
// Because I'm generally a not-great programmer, and couldn't get a splay tree or a skip list to work properly. Rotating trees hurt my brain.
// In fact I spent more than 2 weeks getting a splay tree or skip list to test properly. In the end I thought the saner choice
// would be to leave this for any future developers to pick it up.
//
// How does it work?
// It's a bookkeeping system. Everytime memory is requested, it will go to the free list, and grab the blocks required. Any spares is then
// re-inserted into the free list. Spares are rarely used - mainly because they aren't aligned to the blocksizes.
// There is a map which tracks which address is used (and how big the block is);
// There is a map which tracks which addresses are free for use (and how big the block is);
// There is a "shortcut" map which doesn't require an iteration thru the free list for getting free stuff.
// There are two trackers for tracking the amount of frees and alloc calls. If the ratio is past a certain amount, the memories will be coalesced.
//
// How is the bfc used?
// Every VM will have a bfc (or multiple if there are multiple devices). At startup, an analysis of the inserted Program will be run
// which determines how much memory the VM will need to request from the device. The VM then requests TWICE as much memory (for just-in-case).
// Creation of new Tensors will then use call the alloc methods of the VM, for memories.
//
// Is this the Best Memory Book Keeping System?
// Hell No. There are better ones, but I'm not too good at implementing them. Please feel free to upgrade this.
//
// More information:
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/bfc_allocator.cc
//
// More information about memory allocation and implementing one:
// https://github.com/angrave/SystemProgramming/wiki/Memory,-Part-2%3A-Implementing-a-Memory-Allocator
// https://www.codeproject.com/Articles/14525/Heap-Manager-for-Allocating-Memory-from-a-Shared-M
type bfc struct {
	start        uintptr
	size         int64
	blockSize    int64
	reservedSize int64

	freelist *freelist
	used     map[uintptr]int64 // keeps track of the sizes of each block

	// statistics
	allocated int64
	allocs    int
	frees     int
}

func newBFC(alignment int64) *bfc {
	b := makeBFC(alignment)
	return &b
}

func makeBFC(alignment int64) bfc {
	return bfc{
		blockSize: alignment,
		freelist:  new(freelist),
		used:      make(map[uintptr]int64),
	}
}

func (b *bfc) reset() {
	b.allocated = 0
	b.allocs = 0
	b.frees = 0

}

func (b *bfc) reserve(start uintptr, size int64) {
	allocatorLogf("RESERVE starts: 0x%x | size: %v", start, size)
	b.start = start
	b.size = size - (size % b.blockSize)
	b.reservedSize = size
	b.freelist.insert(newMemblock(0, size))
	allocatorLogf("Start: 0x%x | Size %v", b.start, b.size)
}

func (b *bfc) release() uintptr {
	retVal := b.start
	b.start = 0
	b.size = 0
	b.freelist = new(freelist)
	b.used = make(map[uintptr]int64)
	return retVal
}

func (b *bfc) alloc(size int64) (mem uintptr, err error) {
	allocatorLogf("BFC Allocating %v", size)
	allocatorLogf("before alloc: %v", b.freelist)
	defer allocatorLogf("after alloc: %v", b.freelist)
	enterLogScope()
	defer leaveLogScope()
	if size <= 0 {
		return 0, errors.Errorf("Cannot allocate memory with size 0 or less")
	}
	aligned := b.align(size)
	block := b.bestFit(aligned)
	allocatorLogf("Got a block %v", block)
	if block == nil {
		// first try to coalesce
		b.coalesce()
		if block = b.bestFit(aligned); block == nil {
			// then we're really OOM
			return 0, oomError{
				res:       b.size,
				allocated: b.allocated,
			}
		}

	}
	b.freelist.splitOrRemove(block, aligned, size)
	b.used[block.address] = size

	b.allocated += size
	b.allocs++

	return block.address + b.start, nil
}

func (b *bfc) free(address uintptr) {
	allocatorLogf("BFC Free 0x%x", address)
	enterLogScope()
	defer leaveLogScope()

	allocatorLogf("Before: %v", b.freelist)
	defer allocatorLogf("After: %v", b.freelist)

	a := address - b.start // get internal address
	allocatorLogf("Internal address 0x%x", a)
	size, ok := b.used[a]
	if !ok {
		allocatorLogf("a: 0x%x | 0x%x", a, address)
		allocatorLogf("a: %v | %v %v", a, address, b.start)
		return
		// panic("Cannot free")

	}
	block := newMemblock(a, size)
	b.freelist.insert(block)
	delete(b.used, a)

	b.allocated -= size
	b.frees++
	if float64(b.frees)/float64(b.allocs) >= freeAllocTresh {
		b.coalesce()
	}
}

func (b *bfc) bestFit(size int64) (best *memblock) {
	for block := b.freelist.first; block != nil; block = block.next {
		if block.size >= size {
			return block
		}
	}
	return nil
}

// coalesce coalesces the freelist using these two rules:
//		- address must be aligned to the alignment
//		- if two blocks next to each other share a fencepost, then they will be merged
func (b *bfc) coalesce() {
	allocatorLogf("PreCOALESCE: %v", b.freelist)
	defer allocatorLogf("POSTCOALESCE: %v", b.freelist)
	for block := b.freelist.first; block != nil; block = block.next {
		if block.address%uintptr(b.blockSize) != 0 {
			continue
		}
	inner:
		for next := block.next; next != nil; next = block.next {
			switch {
			case block.cap() == next.address:
				block.size += next.size
				block.next = next.next
				next.next = nil
				next.prev = nil // kill i

				if next == b.freelist.last {
					b.freelist.last = block
				}

				b.freelist.l--
			case block.overlaps(next):
				// unhandled yet
				panic("Unhandled: overlapping coalesceing")
			default:
				break inner
			}
		}
	}
}

func (b *bfc) align(size int64) int64 {
	blocks := size % b.blockSize
	if blocks == 0 {
		return size
	}
	size -= blocks
	return size + b.blockSize
}
