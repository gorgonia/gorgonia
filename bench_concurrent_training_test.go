// +build concurrentTraining

package gorgonia_test

import (
	"runtime"
	"testing"
)

func BenchmarkTrainingConcurrent(b *testing.B) {
	xV, yV, bs := prep()

	for i := 0; i < b.N; i++ {
		concurrentTraining(xV, yV, bs, 10)
	}

	runtime.GC()
}

func BenchmarkTrainingNonConcurrent(b *testing.B) {
	xV, yV, _ := prep()

	for i := 0; i < b.N; i++ {
		nonConcurrentTraining(xV, yV, 10)
	}

	runtime.GC()
}
