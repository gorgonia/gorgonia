package main

import (
	"image"
	"math"
)

type sli struct {
	start, end, step int
}

func (s sli) Start() int { return s.start }
func (s sli) End() int   { return s.end }
func (s sli) Step() int  { return s.step }

func s(start int) sli {
	return sli{
		start: start,
		end:   start + 1,
		step:  0,
	}
}

func hasOne(a []float64) bool {
	for _, v := range a {
		if v == 1.0 {
			return true
		}
	}
	return false
}

func avgF64s(a []float64) (retVal float64) {
	for _, v := range a {
		retVal += v
	}
	retVal /= float64(len(a))
	return
}

const numLabels = 10
const pixelRange = 255

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

func visualizeRow(x []float64) *image.Gray {
	// since this is a square, we can take advantage of that
	l := len(x)
	side := int(math.Sqrt(float64(l)))
	r := image.Rect(0, 0, side, side)
	img := image.NewGray(r)

	pix := make([]byte, l)
	for i, px := range x {
		pix[i] = reversePixelWeight(px)
	}
	img.Pix = pix

	return img
}
