package main

import (
	"image"

	"github.com/chewxy/math32"
)

// Rectify Creates rectangle
func Rectify(x, y, h, w, maxwidth, maxheight int) image.Rectangle {
	return image.Rect(MaxInt(x-w/2, 0), MaxInt(y-h/2, 0), MinInt(x+w/2+1, maxwidth), MinInt(y+h/2+1, maxheight))
}

// IOUFloat32 Intersection Over Union
func IOUFloat32(r1, r2 image.Rectangle) float32 {
	intersection := r1.Intersect(r2)
	interArea := intersection.Dx() * intersection.Dy()
	r1Area := r1.Dx() * r1.Dy()
	r2Area := r2.Dx() * r2.Dy()
	return float32(interArea) / float32(r1Area+r2Area-interArea)
}

// MaxInt Maximum between two integers
func MaxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// MinInt Minimum between two integers
func MinInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Softmax Implementation of softmax
func Softmax(a []float32) []float32 {
	sum := float32(0.0)
	output := make([]float32, len(a))
	for i := 0; i < len(a); i++ {
		output[i] = math32.Exp(a[i])
		sum += output[i]
	}
	for i := 0; i < len(output); i++ {
		output[i] = output[i] / sum
	}
	return output
}

// MaxFloat32 Finds maximum in slice of float32's
func MaxFloat32(cl []float32) (float32, int) {
	max, maxi := float32(-1.0), -1
	for i := range cl {
		if max < cl[i] {
			max = cl[i]
			maxi = i
		}
	}
	return max, maxi
}

// Sigmoid Implementation of sigmoid function
func Sigmoid(sum float32) float32 {
	return 1.0 / (1.0 + math32.Exp(-sum))
}
