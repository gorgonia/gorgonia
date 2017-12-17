// package mnist handles the mnist data set
package mnist

import (
	"os"
	"path/filepath"

	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

// Image holds the pixel intensities of an image.
// 255 is foreground (black), 0 is background (white).
type RawImage []byte

// Label is a digit label in 0 to 9
type Label uint8

// Load loads the mnist data into two tensors
//
// typ can be "train", "test"
//
// loc represents where the mnist files are held
func Load(typ, loc string, as tensor.Dtype) (inputs, targets tensor.Tensor, err error) {
	const (
		trainLabel = "train-labels.idx1-ubyte"
		trainData  = "train-images.idx3-ubyte"
		testLabel  = "t10k-labels.idx1-ubyte"
		testData   = "t10k-images.idx3-ubyte"
	)

	var labelFile, dataFile string
	switch typ {
	case "train", "dev":
		labelFile = filepath.Join(loc, trainLabel)
		dataFile = filepath.Join(loc, trainData)
	case "test":
		labelFile = filepath.Join(loc, testLabel)
		dataFile = filepath.Join(loc, testData)
	}

	var labelData []Label
	var imageData []RawImage

	if labelData, err = readLabelFile(os.Open(labelFile)); err != nil {
		return nil, nil, errors.Wrap(err, "Unable to read Labels")
	}

	if imageData, err = readImageFile(os.Open(dataFile)); err != nil {
		return nil, nil, errors.Wrap(err, "Unable to read image data")
	}

	inputs = prepareX(imageData, as)
	targets = prepareY(labelData, as)
	return
}

func pixelWeight(px byte) float64 {
	retVal := float64(px)/pixelRange*0.9 + 0.1
	if retVal == 1.0 {
		return 0.999
	}
	return retVal
}

func reversePixelWeight(px float64) byte {
	return byte((pixelRange*px - pixelRange) / 0.9)
}

func prepareX(M []RawImage, dt tensor.Dtype) (retVal tensor.Tensor) {
	rows := len(M)
	cols := len(M[0])

	var backing interface{}
	switch dt {
	case tensor.Float64:
		b := make([]float64, rows*cols, rows*cols)
		b = b[:0]
		for i := 0; i < rows; i++ {
			for j := 0; j < len(M[i]); j++ {
				b = append(b, pixelWeight(M[i][j]))
			}
		}
		backing = b
	case tensor.Float32:
		b := make([]float32, rows*cols, rows*cols)
		b = b[:0]
		for i := 0; i < rows; i++ {
			for j := 0; j < len(M[i]); j++ {
				b = append(b, float32(pixelWeight(M[i][j])))
			}
		}
		backing = b
	}
	retVal = tensor.New(tensor.WithShape(rows, cols), tensor.WithBacking(backing))
	return
}

func prepareY(N []Label, dt tensor.Dtype) (retVal tensor.Tensor) {
	rows := len(N)
	cols := 10

	var backing interface{}
	switch dt {
	case tensor.Float64:
		b := make([]float64, rows*cols, rows*cols)
		b = b[:0]
		for i := 0; i < rows; i++ {
			for j := 0; j < 10; j++ {
				if j == int(N[i]) {
					b = append(b, 0.9)
				} else {
					b = append(b, 0.1)
				}
			}
		}
		backing = b
	case tensor.Float32:
		b := make([]float32, rows*cols, rows*cols)
		b = b[:0]
		for i := 0; i < rows; i++ {
			for j := 0; j < 10; j++ {
				if j == int(N[i]) {
					b = append(b, 0.9)
				} else {
					b = append(b, 0.1)
				}
			}
		}
		backing = b

	}
	retVal = tensor.New(tensor.WithShape(rows, cols), tensor.WithBacking(backing))
	return
}
