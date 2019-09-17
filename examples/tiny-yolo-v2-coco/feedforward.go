package main

import (
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// FeedForward Forward pass
func (tiny *TinyYOLOv2Net) FeedForward(g *gorgonia.ExprGraph, x *gorgonia.Node) (err error) {

	// 	0 conv     16       3 x 3/ 1    416 x 416 x   3 ->  416 x 416 x  16
	var conv0, bias0, leaky0 *gorgonia.Node
	//    1 max               2 x 2/ 2    416 x 416 x  16 ->  208 x 208 x  16
	var max1 *gorgonia.Node
	//    2 conv     32       3 x 3/ 1    208 x 208 x  16 ->  208 x 208 x  32
	var conv2, bias2, leaky2 *gorgonia.Node
	//    3 max               2 x 2/ 2    208 x 208 x  32 ->  104 x 104 x  32
	var max3 *gorgonia.Node
	//    4 conv     64       3 x 3/ 1    104 x 104 x  32 ->  104 x 104 x  64
	var conv4, bias4, leaky4 *gorgonia.Node
	//    5 max               2 x 2/ 2    104 x 104 x  64 ->   52 x  52 x  64
	var max5 *gorgonia.Node
	//    6 conv    128       3 x 3/ 1     52 x  52 x  64 ->   52 x  52 x 128
	var conv6, bias6, leaky6 *gorgonia.Node
	//    7 max               2 x 2/ 2     52 x  52 x 128 ->   26 x  26 x 128
	var max7 *gorgonia.Node
	//    8 conv    256       3 x 3/ 1     26 x  26 x 128 ->   26 x  26 x 256
	var conv8, bias8, leaky8 *gorgonia.Node
	//    9 max               2 x 2/ 2     26 x  26 x 256 ->   13 x  13 x 256
	var max9 *gorgonia.Node
	//   10 conv    512       3 x 3/ 1     13 x  13 x 256 ->   13 x  13 x 512
	var conv10, bias10, leaky10 *gorgonia.Node
	//   11 max               2 x 2/ 1     13 x  13 x 512 ->   13 x  13 x 512
	var max11 *gorgonia.Node
	//   12 conv   1024       3 x 3/ 1     13 x  13 x 512 ->   13 x  13 x1024
	var conv12, bias12, leaky12 *gorgonia.Node
	//   13 conv    512       3 x 3/ 1     13 x  13 x1024 ->   13 x  13 x 512
	var conv13, bias13, leaky13 *gorgonia.Node
	//   14 conv    (classesNum+5)*boxesPerCell       1 x 1/ 1     13 x  13 x 512 ->   13 x  13 x (classesNum+5)*boxesPerCell
	var conv14, bias14, leaky14 *gorgonia.Node

	// 	0 conv     16       3 x 3/ 1    416 x 416 x   3 ->  416 x 416 x  16
	if conv0, err = gorgonia.Conv2d(x, tiny.convWeights0, tensor.Shape{3, 3}, []int{1, 1}, []int{1, 1}, []int{1, 1}); err != nil {
		return errors.Wrap(err, "0 conv; convolutional failed")
	}
	bias0 = PrepareBiases(g, conv0.Shape(), tiny.biases, "conv_0", "bias_0")
	if bias0, err = gorgonia.Add(conv0, bias0); err != nil {
		return errors.Wrap(err, "0 conv; bias failed")
	}
	if leaky0, err = gorgonia.LeakyRelu(bias0, 0.1); err != nil {
		return errors.Wrap(err, "0 conv; leaky relu failed")
	}

	//    1 max               2 x 2/ 2    416 x 416 x  16 ->  208 x 208 x  16
	if max1, err = gorgonia.MaxPool2D(leaky0, tensor.Shape{2, 2}, []int{0, 0}, []int{2, 2}); err != nil {
		return errors.Wrap(err, "1 max; max pool failed")
	}

	//    2 conv     32       3 x 3/ 1    208 x 208 x  16 ->  208 x 208 x  32
	if conv2, err = gorgonia.Conv2d(max1, tiny.convWeights2, tensor.Shape{3, 3}, []int{1, 1}, []int{1, 1}, []int{1, 1}); err != nil {
		return errors.Wrap(err, "2 conv; convolutional failed")
	}
	bias2 = PrepareBiases(g, conv2.Shape(), tiny.biases, "conv_2", "bias_2")
	if bias2, err = gorgonia.Add(conv2, bias2); err != nil {
		return errors.Wrap(err, "2 conv; bias failed")
	}
	if leaky2, err = gorgonia.LeakyRelu(bias2, 0.1); err != nil {
		return errors.Wrap(err, "2 conv; leaky relu failed")
	}

	//   3 max               2 x 2/ 2    208 x 208 x  32 ->  104 x 104 x  32
	if max3, err = gorgonia.MaxPool2D(leaky2, tensor.Shape{2, 2}, []int{0, 0}, []int{2, 2}); err != nil {
		return errors.Wrap(err, "3 max; max pool failed")
	}

	//    4 conv     64       3 x 3/ 1    104 x 104 x  32 ->  104 x 104 x  64
	if conv4, err = gorgonia.Conv2d(max3, tiny.convWeights4, tensor.Shape{3, 3}, []int{1, 1}, []int{1, 1}, []int{1, 1}); err != nil {
		return errors.Wrap(err, "4 conv; convolutional failed")
	}
	bias4 = PrepareBiases(g, conv4.Shape(), tiny.biases, "conv_4", "bias_4")
	if bias4, err = gorgonia.Add(conv4, bias4); err != nil {
		return errors.Wrap(err, "4 conv; bias failed")
	}
	if leaky4, err = gorgonia.LeakyRelu(bias4, 0.1); err != nil {
		return errors.Wrap(err, "4 conv; leaky relu failed")
	}

	//   5 max               2 x 2/ 2    104 x 104 x  64 ->   52 x  52 x  64
	if max5, err = gorgonia.MaxPool2D(leaky4, tensor.Shape{2, 2}, []int{0, 0}, []int{2, 2}); err != nil {
		return errors.Wrap(err, "5 max; max pool failed")
	}

	//    6 conv    128       3 x 3/ 1     52 x  52 x  64 ->   52 x  52 x 128
	if conv6, err = gorgonia.Conv2d(max5, tiny.convWeights6, tensor.Shape{3, 3}, []int{1, 1}, []int{1, 1}, []int{1, 1}); err != nil {
		return errors.Wrap(err, "6 conv; convolutional failed")
	}
	bias6 = PrepareBiases(g, conv6.Shape(), tiny.biases, "conv_6", "bias_6")
	if bias6, err = gorgonia.Add(conv6, bias6); err != nil {
		return errors.Wrap(err, "6 conv; bias failed")
	}
	if leaky6, err = gorgonia.LeakyRelu(bias6, 0.1); err != nil {
		return errors.Wrap(err, "6 conv; leaky relu failed")
	}

	//    7 max               2 x 2/ 2     52 x  52 x 128 ->   26 x  26 x 128
	if max7, err = gorgonia.MaxPool2D(leaky6, tensor.Shape{2, 2}, []int{0, 0}, []int{2, 2}); err != nil {
		return errors.Wrap(err, "7 max; max pool failed")
	}

	//    8 conv    256       3 x 3/ 1     26 x  26 x 128 ->   26 x  26 x 256
	if conv8, err = gorgonia.Conv2d(max7, tiny.convWeights8, tensor.Shape{3, 3}, []int{1, 1}, []int{1, 1}, []int{1, 1}); err != nil {
		return errors.Wrap(err, "8 conv; convolutional failed")
	}
	bias8 = PrepareBiases(g, conv8.Shape(), tiny.biases, "conv_8", "bias_8")
	if bias8, err = gorgonia.Add(conv8, bias8); err != nil {
		return errors.Wrap(err, "8 conv; bias failed")
	}
	if leaky8, err = gorgonia.LeakyRelu(bias8, 0.1); err != nil {
		return errors.Wrap(err, "8 conv; leaky relu failed")
	}

	//    9 max               2 x 2/ 2     26 x  26 x 256 ->   13 x  13 x 256
	if max9, err = gorgonia.MaxPool2D(leaky8, tensor.Shape{2, 2}, []int{0, 0}, []int{2, 2}); err != nil {
		return errors.Wrap(err, "9 max; max pool failed")
	}

	//   10 conv    512       3 x 3/ 1     13 x  13 x 256 ->   13 x  13 x 512
	if conv10, err = gorgonia.Conv2d(max9, tiny.convWeights10, tensor.Shape{3, 3}, []int{1, 1}, []int{1, 1}, []int{1, 1}); err != nil {
		return errors.Wrap(err, "10 conv; convolutional failed")
	}
	bias10 = PrepareBiases(g, conv10.Shape(), tiny.biases, "conv_10", "bias_10")
	if bias10, err = gorgonia.Add(conv10, bias10); err != nil {
		return errors.Wrap(err, "10 conv; bias failed")
	}
	if leaky10, err = gorgonia.LeakyRelu(bias10, 0.1); err != nil {
		return errors.Wrap(err, "10 conv; leaky relu failed")
	}

	//   11 max               2 x 2/ 1     13 x  13 x 512 ->   13 x  13 x 512
	// we have special padding and stride here
	if max11, err = gorgonia.MaxPool2D(leaky10, tensor.Shape{2, 2}, []int{1, 0, 1, 0}, []int{1, 1}); err != nil {
		return errors.Wrap(err, "11 max; max pool failed")
	}

	//   12 conv   1024       3 x 3/ 1     13 x  13 x 512 ->   13 x  13 x1024
	if conv12, err = gorgonia.Conv2d(max11, tiny.convWeights12, tensor.Shape{3, 3}, []int{1, 1}, []int{1, 1}, []int{1, 1}); err != nil {
		return errors.Wrap(err, "12 conv; convolutional failed")
	}
	bias12 = PrepareBiases(g, conv12.Shape(), tiny.biases, "conv_12", "bias_12")
	if bias12, err = gorgonia.Add(conv12, bias12); err != nil {
		return errors.Wrap(err, "12 conv; bias failed")
	}
	if leaky12, err = gorgonia.LeakyRelu(bias12, 0.1); err != nil {
		return errors.Wrap(err, "12 conv; leaky relu failed")
	}

	//   13 conv    512       3 x 3/ 1     13 x  13 x1024 ->   13 x  13 x 512
	if conv13, err = gorgonia.Conv2d(leaky12, tiny.convWeights13, tensor.Shape{3, 3}, []int{1, 1}, []int{1, 1}, []int{1, 1}); err != nil {
		return errors.Wrap(err, "13 conv; convolutional failed")
	}
	bias13 = PrepareBiases(g, conv13.Shape(), tiny.biases, "conv_13", "bias_13")
	if bias13, err = gorgonia.Add(conv13, bias13); err != nil {
		return errors.Wrap(err, "13 conv; bias failed")
	}
	if leaky13, err = gorgonia.LeakyRelu(bias13, 0.1); err != nil {
		return errors.Wrap(err, "13 conv; leaky relu failed")
	}

	//   14 conv    (classesNum+5)*boxesPerCell       1 x 1/ 1     13 x  13 x 512 ->   13 x  13 x (classesNum+5)*boxesPerCell
	// we have special kernel size, padding and alpha parameter in LeakyReLU here
	if conv14, err = gorgonia.Conv2d(leaky13, tiny.convWeights14, tensor.Shape{1, 1}, []int{0, 0}, []int{1, 1}, []int{1, 1}); err != nil {
		return errors.Wrap(err, "14 conv; convolutional failed")
	}
	bias14 = PrepareBiases(g, conv14.Shape(), tiny.biases, "conv_14", "bias_14")
	if bias14, err = gorgonia.Add(conv14, bias14); err != nil {
		return errors.Wrap(err, "14 conv; bias failed")
	}
	if leaky14, err = gorgonia.LeakyRelu(bias14, 1.0); err != nil {
		return errors.Wrap(err, "14 conv; leaky relu failed")
	}
	out := leaky14
	tiny.out = out

	return nil
}
