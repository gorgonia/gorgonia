package main

import (
	"encoding/binary"
	"io"
	"math"
	"os"

	"gorgonia.org/tensor"
)

// ParseTinyYOLOv2 Parse darknet weights (v2)
func ParseTinyYOLOv2(fname string) []float32 {
	fp, err := os.Open(fname)
	if err != nil {
		panic(err)
	}
	defer fp.Close()
	summary := []byte{}
	data := make([]byte, 4096)
	for {
		data = data[:cap(data)]
		n, err := fp.Read(data)
		if err != nil {
			if err == io.EOF {
				break
			}
			return nil
		}
		data = data[:n]
		summary = append(summary, data...)
	}
	dataF32 := []float32{}
	for i := 0; i < len(summary); i += 4 {
		tempSlice := summary[i : i+4]
		tempFloat32 := Float32frombytes(tempSlice)
		dataF32 = append(dataF32, tempFloat32)
	}
	return dataF32
}

// PrepareData Prepare kernels, biaseses, means, gammas and vars
func PrepareData(biases, gammas, means, vars, kernels map[string][]float32, data []float32, layerName string, convShape tensor.Shape, lastIdx *int, batchNorm, biased bool) {
	if batchNorm {
		nb := convShape[0]
		nk := convShape.TotalSize()
		biases[layerName] = make([]float32, 0)
		biases[layerName] = append(biases[layerName], data[*lastIdx:*lastIdx+nb]...)
		*lastIdx += nb
		gammas[layerName] = make([]float32, 0)
		gammas[layerName] = append(gammas[layerName], data[*lastIdx:*lastIdx+nb]...)
		*lastIdx += nb
		means[layerName] = make([]float32, 0)
		means[layerName] = append(means[layerName], data[*lastIdx:*lastIdx+nb]...)
		*lastIdx += nb
		vars[layerName] = make([]float32, 0)
		vars[layerName] = append(vars[layerName], data[*lastIdx:*lastIdx+nb]...)
		*lastIdx += nb
		kernels[layerName] = make([]float32, 0)
		kernels[layerName] = append(kernels[layerName], data[*lastIdx:*lastIdx+nk]...)
		*lastIdx += nk
	} else {
		if biased {
			nb := convShape[0]
			nk := convShape.TotalSize()
			biases[layerName] = make([]float32, 0)
			biases[layerName] = append(biases[layerName], data[*lastIdx:*lastIdx+nb]...)
			*lastIdx += nb
			kernels[layerName] = make([]float32, 0)
			kernels[layerName] = append(kernels[layerName], data[*lastIdx:*lastIdx+nk]...)
			*lastIdx += nk
		}
	}
}

// DenormalizeWeights Denormilize biases and kernels
func DenormalizeWeights(biases, gammas, means, vars, kernels map[string][]float32, layerName string, convShape tensor.Shape, epsilon float32) {
	biasesExtract := biases[layerName]
	gammasExtract := gammas[layerName]
	meansExtract := means[layerName]
	varsExtract := vars[layerName]
	kernelsExtract := kernels[layerName]

	for i := 0; i < convShape[0]; i++ {
		scale := gammasExtract[i] / float32(math.Sqrt(float64(varsExtract[i]+epsilon)))

		biasesExtract[i] = biasesExtract[i] - meansExtract[i]*scale
		isize := convShape[1] * convShape[2] * convShape[3]
		for j := 0; j < isize; j++ {
			kernelsExtract[isize*i+j] = kernelsExtract[isize*i+j] * scale
		}
	}

	biases[layerName] = biasesExtract
	kernels[layerName] = kernelsExtract

}

// Float32frombytes []byte to float32
func Float32frombytes(bytes []byte) float32 {
	bits := binary.LittleEndian.Uint32(bytes)
	float := math.Float32frombits(bits)
	return float
}

// Float32bytes float32 to []byte
func Float32bytes(float float32) []byte {
	bits := math.Float32bits(float)
	bytes := make([]byte, 8)
	binary.LittleEndian.PutUint32(bytes, bits)
	return bytes
}
