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

// it's just a generic ceiling function. Added here to avoid mixing with any potential ceilInt operation
func calcBlocks(n, maxThreads int) int {
	return (n + maxThreads - 1) / maxThreads
}
