package main

import (
	"encoding/binary"
	"image"
	"io"
	"log"
	"math"
	"os"

	tf64 "github.com/chewxy/gorgonia/tensor/f64"
)

const numLabels = 10
const pixelRange = 255

const (
	imageMagic = 0x00000803
	labelMagic = 0x00000801
	Width      = 28
	Height     = 28
)

// Image holds the pixel intensities of an image.
// 255 is foreground (black), 0 is background (white).
type RawImage []byte

// Label is a digit label in 0 to 9
type Label uint8

func readLabelFile(r io.Reader) (labels []Label, err error) {
	var (
		magic int32
		n     int32
	)
	if err = binary.Read(r, binary.BigEndian, &magic); err != nil {
		return nil, err
	}
	if magic != labelMagic {
		return nil, os.ErrInvalid
	}
	if err = binary.Read(r, binary.BigEndian, &n); err != nil {
		return nil, err
	}
	labels = make([]Label, n)
	for i := 0; i < int(n); i++ {
		var l Label
		if err := binary.Read(r, binary.BigEndian, &l); err != nil {
			return nil, err
		}
		labels[i] = l
	}
	return labels, nil
}

func readImageFile(r io.Reader) (rows, cols int, imgs []RawImage, err error) {
	var (
		magic int32
		n     int32
		nrow  int32
		ncol  int32
	)
	if err = binary.Read(r, binary.BigEndian, &magic); err != nil {
		return 0, 0, nil, err
	}
	if magic != imageMagic {
		return 0, 0, nil, err /*os.ErrInvalid*/
	}
	if err = binary.Read(r, binary.BigEndian, &n); err != nil {
		return 0, 0, nil, err
	}
	if err = binary.Read(r, binary.BigEndian, &nrow); err != nil {
		return 0, 0, nil, err
	}
	if err = binary.Read(r, binary.BigEndian, &ncol); err != nil {
		return 0, 0, nil, err
	}
	imgs = make([]RawImage, n)
	m := int(nrow * ncol)
	for i := 0; i < int(n); i++ {
		imgs[i] = make(RawImage, m)
		m_, err := io.ReadFull(r, imgs[i])
		if err != nil {
			return 0, 0, nil, err
		}
		if m_ != int(m) {
			return 0, 0, nil, os.ErrInvalid
		}
	}
	return int(nrow), int(ncol), imgs, nil
}

func open(path string) *os.File {
	file, err := os.Open(path)
	if err != nil {
		log.Printf("path: %q err: %v", path, err)
		os.Exit(-1)
	}
	return file
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

func prepareX(M []RawImage) (retVal *tf64.Tensor) {
	rows := len(M)
	cols := len(M[0])

	backing := make([]float64, rows*cols, rows*cols)
	backing = backing[:0]
	for i := 0; i < rows; i++ {
		for j := 0; j < len(M[i]); j++ {
			backing = append(backing, pixelWeight(M[i][j]))
		}
	}
	retVal = tf64.NewTensor(tf64.WithShape(rows, cols), tf64.WithBacking(backing))
	return
}

func prepareY(N []Label) (retVal *tf64.Tensor) {
	rows := len(N)
	cols := 10
	backing := make([]float64, rows*cols, rows*cols)
	backing = backing[:0]

	for i := 0; i < rows; i++ {
		for j := 0; j < 10; j++ {
			if j == int(N[i]) {
				backing = append(backing, 0.9)
			} else {
				backing = append(backing, 0.1)
			}
		}
	}
	retVal = tf64.NewTensor(tf64.WithShape(rows, cols), tf64.WithBacking(backing))
	return
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
