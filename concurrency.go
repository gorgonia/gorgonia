package gorgonia

import "runtime"

const (
	// defaultBlockSize indicates the default size to chunk an array.
	defaultBlockSize = 64

	// minParallelBlocks indicates how many blocks an array must be split up into
	// before we decide to parallelize
	minParallelBlocks = 4
)

// workersChan creates a channel that is limited by the number of processor cores.
// To signal that a processor core is being used:
// 	ch <- struct{}{}
// When done:
// 	<- ch
func workersChan() chan struct{} { return make(chan struct{}, runtime.GOMAXPROCS(0)) }

// calcBlocks calculuates the best number of blocks given a blocksize
func calcBlocks(dim, blocksize int) int { return (dim + blocksize - 1) / blocksize }
