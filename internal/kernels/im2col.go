package kernels

import (
	"sync"

	"gorgonia.org/shapes"
	"gorgonia.org/tensor"
)

type Im2ColOp struct {
	H, W                 int // kernel height and  width
	PadH, PadW           int
	StrideH, StrideW     int
	DilationH, DilationW int

	Chans  int
	Height int
	Width  int

	ChanStride  int
	InRowStride int

	RetH, RetW int
}

// Im2Col implements an optimized version of the kernel for the Im2Col Op for CPU execution.
func Im2Col[DT tensor.Num](op Im2ColOp, im, col []DT, wg *sync.WaitGroup, workers chan struct{}) {
	chans, height, width := op.Chans, op.Height, op.Width
	chanStride := op.ChanStride
	inRowStride := op.InRowStride
	retHeight := op.RetH
	retWidth := op.RetW
	workers <- struct{}{}
	var colIdx, inputRow, inputCol int
	for outputRow := 0; outputRow < retHeight; outputRow++ {
		for outputCol := 0; outputCol < retWidth; outputCol++ {
			for ch := 0; ch < chans; ch++ {
				for kernelRow := 0; kernelRow < op.h; kernelRow++ {
					inputRow = -op.padH + kernelRow*op.dilationH + outputRow*op.strideH
					for kernelCol := 0; kernelCol < op.w; kernelCol++ {
						if inputRow < 0 || inputRow >= height {
							col[colIdx] = 0
							colIdx++
							continue
						}
						inputCol = -op.padW + kernelCol*op.dilationW + outputCol*op.strideW
						if inputCol < 0 || inputCol >= width {
							col[colIdx] = 0
						} else {
							imIdx := chanStride*ch + inputRow*width + inputCol
							col[colIdx] = im[imIdx]
						}
						colIdx++
					}
				}
			}
		}
	}
	<-workers
	wg.Done()
}
