package gorgonia

import (
	"sort"

	"github.com/pkg/errors"
)

const (
	minAllocBits = 8
	minAllocSize = 1 << minAllocBits

	freeAllocTresh = 0.75
)

// memblock is a tuple of address and the size of the block - it's almost like a slice hdr, but smaller
type memblock struct {
	address uintptr
	size    int64
}

func (a memblock) cap() uintptr { return a.address + uintptr(a.size) }

// overlaps checks if two memblocks are overlapping.
func (a memblock) overlaps(b memblock) bool {
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

// we say a memblock is less than another memblock when:
//		a.address < b.address and they don't both overlap
func (a memblock) lt(b memblock) bool {
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

// freelist is a data structure that handles free lists. Currently it's underlying data structure is a flat slice
type freelist struct {
	l []memblock
}

func (l *freelist) Len() int           { return len(l.l) }
func (l *freelist) Less(i, j int) bool { return l.l[i].address < l.l[j].address }
func (l *freelist) Swap(i, j int) {
	l.l[i].address, l.l[j].address = l.l[j].address, l.l[i].address
	l.l[i].size, l.l[j].size = l.l[j].size, l.l[i].size
}

func newFreelist(size int) *freelist {
	return &freelist{
		l: make([]memblock, 0, size),
	}
}

func (l *freelist) insert(block memblock) {
	var i int
	var b memblock
	for i, b = range l.l {
		if b.address >= block.cap() {
			break
		}
	}
	l.insertAt(i, block)
}

func (l *freelist) insertAt(i int, block memblock) {
	cudaLogf("Insert %v at %v", block, i)
	switch {
	case i < len(l.l)-1:
		switch {
		case block.cap() == l.l[i+1].address:
			// coalesce when possible
			l.l[i+1].address = block.address
			l.l[i+1].size += block.size
			return
		case l.l[i].address == block.address:
			l.l[i].size = block.size
		default:
		}
	case i < len(l.l):
		if l.l[i].address == block.address {
			l.l[i].size = block.size
			return
		}
	default:
	}
	l.l = append(l.l, memblock{})
	copy(l.l[i+1:], l.l[i:])
	l.l[i] = block
}

func (l *freelist) split(i int, block memblock, aligned, size int64) {
	cudaLogf("Split %v at %d for %v (want %d)", block, i, aligned, size)
	newAddress := block.address + uintptr(size)
	newSize := aligned - size
	newBlock := memblock{newAddress, newSize}

	cudaLogf("newBlock: %v | cap %v | old size %v", newBlock, newBlock.cap(), l.l[i])
	// l.l[i] = newBlock
	l.l[i].address = newBlock.cap()
	l.l[i].size -= size
	if newSize != 0 {
		l.insertAt(i, newBlock)
	}
	cudaLogf("split freelist %v", l)
}

func (l *freelist) remove(i int) {
	copy(l.l[i:], l.l[i+1:])
	l.l = l.l[:len(l.l)-1]
}

func (l *freelist) bestFit(size int64) (int, memblock, error) {
	for i, v := range l.l {
		if v.size >= size {
			return i, v, nil
		}

		// otherwise, keep going until a best fit is found
	}

	return -1, memblock{}, noopError{} // well it should be OOM
}

// bfc an accounting structure for memory allocation,
// directly inspired by TensorFlows' Best Fit With Coalescing memory allocator, which is a type of buddy memory allocator.
// Where TensorFlow's is a simplified dlmalloc, this is an even more simplified version.
// Because the bfc exists solely as an bookkeeping structure, we can make it a lot simpler than using a linked list for tracking a free list.
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
	start     uintptr
	size      int64
	allocated int64
	blockSize int64

	freelist *freelist           // replace this with a proper/better implementation of a free lis with a splay tree underlying it
	bestFits map[int64][]uintptr // for quicker access - for realz, this should be replaced by a splay tree
	usedpool map[uintptr]int64
	freepool map[uintptr]int64

	allocs, frees int
}

func newBFC(alignment int64) *bfc {
	// aligned := size - (size % blockSize)
	b := &bfc{
		// size:      aligned,
		blockSize: alignment,
		freelist:  newFreelist(256), // typical simple progs don't have more than 128 allocations. We'll double that.

		usedpool: make(map[uintptr]int64),
		freepool: make(map[uintptr]int64),
		bestFits: make(map[int64][]uintptr),
	}
	// b.freelist = append(b.freelist, memblock{0, size})
	return b
}

func (b *bfc) reserve(start uintptr, size int64) {
	logf("RESERVE starts: 0x%x | size: %v", start, size)
	b.start = start
	b.size = size - (size % b.blockSize)
	b.freelist.insert(memblock{0, size})
	logf("Actually reserved %v", b.size)
}

func (b *bfc) alloc(size int64) (mem uintptr, err error) {
	cudaLogf("Allocating %v", size)
	cudaLogf("Freelist: %v", b.freelist)
	defer cudaLogf("Freelist After: %v", b.freelist)
	enterLoggingContext()
	defer leaveLoggingContext()
	if size <= 0 {
		return 0, errors.Errorf("Cannot allocate memory with size 0 or less")
	}

	// try to get from quick access
	/*
		cudaLogf("Trying from quick access")
		fits := b.bestFits[size]
		if len(fits) > 0 {
			mem = fits[len(fits)-1]
			fits = fits[:len(fits)-1]
			b.bestFits[size] = fits

			var i int
			var block memblock
			var split bool
			for i, block = range b.freelist.l {
				if block.address == mem {
					if block.size > size {
						split = true
					}
					break
				}
			}

			b.usedpool[mem] = size

			// split or remove
			cudaLogf("remove or split free list blocks %v", block)
			if split {
				aligned := b.align(size)
				b.freelist.split(i, block, aligned, size)
			} else {
				b.freelist.remove(i)

			}

			b.allocated += size
			mem += b.start
			cudaLogf("%v", b.freelist)
			return
		}
	*/

	cudaLogf("Get from free list instead")
	aligned := b.align(size)
	i, block, err := b.freelist.bestFit(aligned)
	if err != nil {
		err = oomError{
			res:       b.size,
			allocated: b.allocated,
		}
		return
	}

	// remove block from free list or split
	cudaLogf("remove or split free list blocks %v", block)
	if block.size > size {
		// split
		b.freelist.split(i, block, aligned, size)
	} else {
		// remove
		b.freelist.remove(i)
	}

	b.usedpool[block.address] = size
	b.allocs++
	b.allocated += size
	return block.address + b.start, nil
}

func (b *bfc) free(address uintptr) {
	cudaLogf("Free 0x%x", address)
	cudaLogf("%v", b.freelist)
	defer cudaLogf("AFTER: %v", b.freelist)
	enterLoggingContext()
	defer leaveLoggingContext()
	a := address - b.start // get internal address

	cudaLogf("Free internal %v", a)
	size := b.usedpool[a]
	delete(b.usedpool, a)

	block := memblock{a, size}
	cudaLogf("inserting block %v", block)
	// b.freepool[a] = size
	b.freelist.insert(block)
	b.freepool[a] = size

	// cudaLogf("Adding %v to best fits", size)
	// b.bestFits[size] = append(b.bestFits[size], a)

	b.allocated -= size
	b.frees++
	if float64(b.frees)/float64(b.allocs) >= freeAllocTresh {
		b.coalesce()
	}
}

func (b *bfc) coalesce() {
	cudaLogf("Coalesce: %v", b.freelist)
	cudaLogf("Best Fits %v", b.bestFits)
	defer cudaLogf("after Coalesce: %v", b.freelist)
	enterLoggingContext()
	defer leaveLoggingContext()
	// toRemove, sizes := b.freelist.coalesce(b.blockSize)

	cudaLogf("Before Sort %v", b.freelist)
	cudaLogf("IS SORTED: %v", sort.IsSorted(b.freelist))
	sort.Sort(b.freelist) // shouldn't be necessary given items in the free list were inserted sorted
	cudaLogf("After Sort %v", b.freelist)

	l := b.freelist
	var toRemove []memblock
	// var coalesceds, olds []memblock
	for i := 0; i < len(l.l); i++ {
		if i == len(l.l)-1 {
			break
		}
		block := l.l[i]
		next := l.l[i+1]
		nextAligned := next.address%uintptr(b.blockSize) == 0
		blockAligned := block.address%uintptr(b.blockSize) == 0
		canCoalesce := !nextAligned || blockAligned

		cudaLogf("COALESCEING onto %v", block)
		enterLoggingContext()
		cudaLogf("next: %v | canCoalesce: %v | cap: %v, %v", next, canCoalesce, block.cap(), next.address)
		// oldBlock := memblock{block.address, block.size}
		// var coalesced bool
		for canCoalesce && next.address == block.cap() {
			cudaLogf("Next: %v", next)
			// coalesce happens here
			l.l[i].size += next.size
			block.size += next.size
			toRemove = append(toRemove, next)

			// shrink the slice
			if i+2 < len(l.l) {
				cudaLogf("shrink")
				copy(l.l[i+1:], l.l[i+2:])
				l.l = l.l[:len(l.l)-1]
			} else if i+2 == len(l.l) {
				cudaLogf("i+2: %v - Last: %v", i+2, l.l[i+1])
				l.l = l.l[:i+1]
				break
			}

			next = l.l[i+1]
			nextAligned = next.address%uintptr(b.blockSize) == 0
			canCoalesce = !nextAligned || blockAligned
			// coalesced = true
		}
		leaveLoggingContext()
		cudaLogf("DONE COALESCING")

		// if coalesced {
		// 	coalesceds = append(coalesceds, block)
		// 	olds = append(olds, oldBlock)
		// }
	}
	/*
		// remove the coalesced blocks from any caches
		for _, block := range toRemove {
			ptr := block.address
			size := block.size
			delete(b.freepool, ptr)

			addrs := b.bestFits[size]
			cudaLogf("Best fits for %v: %v", size, addrs)
			var del int
			for _, a := range addrs {
				if a == ptr {
					continue
				}
				addrs[del] = a
				del++
			}
			addrs = addrs[:del]
			b.bestFits[size] = addrs
			cudaLogf("After removal %v", addrs)
		}

	*/
	/*
		// update any caches
		logf("Updating caches")
		for i, old := range olds {
			logf("dealing with old %v", old)
			enterLoggingContext()
			// we'll delete the address from this block
			addrs := b.bestFits[old.size]
			logf("addrs %v", addrs)
			var del int
			for _, a := range addrs {
				if a == old.address {
					continue
				}
				addrs[del] = a
				del++
			}
			addrs = addrs[:del]
			b.bestFits[old.size] = addrs

			// add the new addresses
			block := coalesceds[i]
			b.bestFits[block.size] = append(b.bestFits[block.size], block.address)
			leaveLoggingContext()
		}
	*/
}

func (b *bfc) align(size int64) int64 {
	blocks := size % b.blockSize
	if blocks == 0 {
		return size
	}
	size -= blocks
	return size + b.blockSize
}
