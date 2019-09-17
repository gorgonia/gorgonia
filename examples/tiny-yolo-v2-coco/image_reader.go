package main

import (
	"image"
	"image/jpeg"
	"os"
)

// GetFloat32Image Returns []float32 representation of image file
func GetFloat32Image(fname string) ([]float32, error) {
	file, err := os.Open(fname)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	img, err := jpeg.Decode(file)
	if err != nil {
		return nil, err
	}

	return Image2Float32(img)
}

// Image2Float32 Returns []float32 representation of image.Image
func Image2Float32(img image.Image) ([]float32, error) {
	width := img.Bounds().Dx()
	height := img.Bounds().Dy()
	imgwh := width * height
	imgSize := imgwh * 3

	ans := make([]float32, imgSize)
	for x := 0; x < width; x++ {
		for y := 0; y < height; y++ {
			r, g, b, _ := img.At(y, x).RGBA()
			rpix, gpix, bpix := float32(r>>8)/float32(255.0), float32(g>>8)/float32(255.0), float32(b>>8)/float32(255.0)
			ans[y+x*height] = rpix
			ans[y+x*height+imgwh] = gpix
			ans[y+x*height+imgwh+imgwh] = bpix
		}
	}
	return ans, nil
}
