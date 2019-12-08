package gorgonia

import (
	"math"
	"time"

	rng "github.com/leesper/go_rng"
	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

// This file provides several weight initialization utility functions.
// It uses the rng package by leesper

// InitWFn is a type of helper function to help initialize weights vector/matrices.
// It generates the backing required for the tensors.
//
// It's typically used in closures
type InitWFn func(dt tensor.Dtype, s ...int) interface{}

// Zeroes creates an InitWfn that populates a Value with... zeroes. I don't know what you expected.
func Zeroes() InitWFn {
	f := func(dt tensor.Dtype, s ...int) interface{} {
		size := tensor.Shape(s).TotalSize()
		switch dt {
		case tensor.Float64:
			return make([]float64, size)
		case tensor.Float32:
			return make([]float32, size)
		case tensor.Int:
			return make([]int, size)
		default:
			err := errors.Errorf(nyiTypeFail, "Zeroes", dt)
			panic(err)
		}
	}
	return f
}

// Ones creates an InitWfn that populates a Value with ones. See Zeroes() for more explanation.
func Ones() InitWFn {
	return func(dt tensor.Dtype, s ...int) interface{} { return ones(dt, s...).Data() }
}

// RangedFrom creates an InitWFn that populates a Value starting with the provided start, increamenting the number for each element in the value by 1
func RangedFrom(start int) InitWFn {
	f := func(dt tensor.Dtype, s ...int) interface{} {
		size := tensor.Shape(s).TotalSize()
		return tensor.Range(dt, start, start+size)
	}
	return f
}

// ValuesOf creates an InitWrn that populates a value with val. This function will cause a panic if val's type is incompatible with the values type.
func ValuesOf(val interface{}) InitWFn {
	f := func(dt tensor.Dtype, s ...int) interface{} {
		size := tensor.Shape(s).TotalSize()

		switch dt {
		case tensor.Float64:
			v := val.(float64)
			retVal := make([]float64, size)
			for i := range retVal {
				retVal[i] = v
			}
			return retVal
		case tensor.Float32:
			v := val.(float32)
			retVal := make([]float32, size)
			for i := range retVal {
				retVal[i] = v
			}
			return retVal
		case tensor.Int:
			v := val.(int)
			retVal := make([]int, size)
			for i := range retVal {
				retVal[i] = v
			}
			return retVal
		default:
			err := errors.Errorf(nyiTypeFail, "Zeroes", dt)
			panic(err)
		}
	}
	return f
}

// Gaussian creates a InitWFn with the specified parameters.
// Example Usage:
//		w := NewMatrix(g, Float64, WithName("w"), WithShape(2,2), WithInit(Gaussian(0, 1)))
// This will create a backing slice of []float64, with the length of 4, and its values are drawn from a gaussian distro
func Gaussian(mean, stdev float64) InitWFn {
	f := func(dt tensor.Dtype, s ...int) interface{} {
		switch dt {
		case tensor.Float64:
			return Gaussian64(mean, stdev, s...)
		case tensor.Float32:
			return Gaussian32(mean, stdev, s...)
		default:
			err := errors.Errorf(nyiTypeFail, "Gaussian init", dt)
			panic(err)
		}
	}
	return f
}

// Uniform creates a InitWFn with the specified parameters.
// Example Usage:
//		w := NewMatrix(g, Float64, WithName("w"), WithShape(2,2), WithInit(Uniform(-1, 1)))
// This will create a backing slice of []float64, with the length of 4, and its values are drawn from a uniform distro
func Uniform(low, high float64) InitWFn {
	f := func(dt tensor.Dtype, s ...int) interface{} {
		switch dt {
		case tensor.Float64:
			return Uniform64(low, high, s...)
		case tensor.Float32:
			return Uniform32(low, high, s...)
		default:
			err := errors.Errorf(nyiTypeFail, "Uniform init", dt)
			panic(err)
		}
	}
	return f
}

// GlorotN creates a InitWFn that populates a Value with weights normally sampled using Glorot et al.'s algorithm
func GlorotN(gain float64) InitWFn {
	f := func(dt tensor.Dtype, s ...int) interface{} {
		switch dt {
		case tensor.Float64:
			return GlorotEtAlN64(gain, s...)
		case tensor.Float32:
			return GlorotEtAlN32(gain, s...)
		default:
			err := errors.Errorf(nyiTypeFail, "GlorotN", dt)
			panic(err)
		}
	}
	return f
}

// GlorotU creates a InitWFn that populates a Value with weights uniformly sampled using Glorot et al.'s algorithm
func GlorotU(gain float64) InitWFn {
	f := func(dt tensor.Dtype, s ...int) interface{} {
		switch dt {
		case tensor.Float64:
			return GlorotEtAlU64(gain, s...)
		case tensor.Float32:
			return GlorotEtAlU32(gain, s...)
		default:
			err := errors.Errorf(nyiTypeFail, "GlorotU", dt)
			panic(err)
		}
	}
	return f
}

func HeN(gain float64) InitWFn {
	f := func(dt tensor.Dtype, s ...int) interface{} {
		switch dt {
		case tensor.Float64:
			return HeEtAlN64(gain, s...)
		default:
			err := errors.Errorf(nyiTypeFail, "HeNormal", dt)
			panic(err)
		}
	}
	return f
}

func HeU(gain float64) InitWFn {
	f := func(dt tensor.Dtype, s ...int) interface{} {
		switch dt {
		case tensor.Float64:
			return HeEtAlU64(gain, s...)
		default:
			err := errors.Errorf(nyiTypeFail, "HeUniform", dt)
			panic(err)
		}
	}
	return f
}

// Gaussian64 returns a []float64 drawn from a gaussian distribution as defined by the mean and stdev
func Gaussian64(mean, stdev float64, s ...int) []float64 {
	size := tensor.Shape(s).TotalSize()

	rand := rng.NewGaussianGenerator(time.Now().UnixNano())
	retVal := make([]float64, size)
	for i := range retVal {
		retVal[i] = rand.Gaussian(mean, stdev)
	}
	return retVal
}

// Gaussian32 returns a []float32 drawn from a gaussian distribution as defined by the mean and stdev
func Gaussian32(mean, stdev float64, s ...int) []float32 {
	size := tensor.Shape(s).TotalSize()

	rand := rng.NewGaussianGenerator(time.Now().UnixNano())
	retVal := make([]float32, size)
	for i := range retVal {
		retVal[i] = float32(rand.Gaussian(mean, stdev))
	}
	return retVal
}

// Uniform64 returns a []float64 drawn from a uniform distribution between [low, high) that is provided
func Uniform64(low, high float64, s ...int) []float64 {
	size := tensor.Shape(s).TotalSize()

	rand := rng.NewUniformGenerator(time.Now().UnixNano())
	retVal := make([]float64, size)
	for i := range retVal {
		retVal[i] = rand.Float64Range(low, high)
	}
	return retVal
}

// Uniform32 returns a []float64 drawn from a uniform distribution between [low, high) that is provided
func Uniform32(low, high float64, s ...int) []float32 {
	size := tensor.Shape(s).TotalSize()
	l := float32(low)
	h := float32(high)

	rand := rng.NewUniformGenerator(time.Now().UnixNano())
	retVal := make([]float32, size)
	for i := range retVal {
		retVal[i] = rand.Float32Range(l, h)
	}
	return retVal
}

// Binomial64 returns a []float64 drawn from a binomial distribution given the trial and probability parameters.
func Binomial64(trials, prob float64, s ...int) []float64 {
	size := tensor.Shape(s).TotalSize()
	t := int64(trials)

	rand := rng.NewBinomialGenerator(time.Now().UnixNano())
	retVal := make([]float64, size)
	for i := range retVal {
		retVal[i] = float64(rand.Binomial(t, prob))
	}
	return retVal
}

// Binomial32 returns a []float32 drawn from a binomial distribution given the trial and probability parameters.
func Binomial32(trials, prob float64, s ...int) []float32 {
	size := tensor.Shape(s).TotalSize()
	t := int64(trials)

	rand := rng.NewBinomialGenerator(time.Now().UnixNano())
	retVal := make([]float32, size)
	for i := range retVal {
		retVal[i] = float32(rand.Binomial(t, prob))
	}
	return retVal
}

/* SOPHISTICATED INITIALIZATION STRATEGIES */

// GlorotEtAlN64 returns float64 weights sampled from a normal distribution
// using the methods specified in Glorot et. al (2010).
// See also: http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
func GlorotEtAlN64(gain float64, s ...int) []float64 {
	var n1, n2 int
	fieldSize := 1
	switch len(s) {
	case 0:
		panic("Glorot Uniform only works with Tensors of dimensions >= 1")
	case 1:
		// treat it as a col vec
		n1 = 1
		n2 = s[0]
	default:
		n1, n2 = s[0], s[1]
		for _, v := range s[2:] {
			fieldSize *= v
		}
	}

	size := tensor.Shape(s).TotalSize()
	fanIn := float64((n1 + n2) * fieldSize)

	stdev := gain * math.Sqrt(2.0/fanIn)

	rand := rng.NewGaussianGenerator(time.Now().UnixNano())
	retVal := make([]float64, size)
	for i := range retVal {
		retVal[i] = rand.Gaussian(0.0, stdev)
	}
	return retVal
}

// GlorotEtAlN32 returns float32 weights sampled from a normal distribution
// using the methods specified in Glorot et. al (2010).
// See also: http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
func GlorotEtAlN32(gain float64, s ...int) []float32 {
	f64 := GlorotEtAlN64(gain, s...)
	retVal := make([]float32, len(f64))
	for i, v := range f64 {
		retVal[i] = float32(v)
	}
	return retVal
}

// GlorotEtAlU64 returns float64 weights sampled from a uniform distribution
// using the methods specified in Glorot et. al (2010).
// See also: http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
//
// For best results, use:
// 		1.0 for gain for weights that will be used in linear and/or sigmoid units
//		math.Sqrt(2.0) for gain for weights that will be used in ReLU units
//		math.Sqrt(2.0 / (1+alpha*alpha)) for ReLU that are leaky with alpha
func GlorotEtAlU64(gain float64, s ...int) []float64 {
	var n1, n2 int
	fieldSize := 1
	switch len(s) {
	case 0:
		panic("Glorot Uniform only works with Tensors of dimensions >= 1")
	case 1:
		// treat it as a col vec
		n1 = 1
		n2 = s[0]
	default:
		n1, n2 = s[0], s[1]
		for _, v := range s[2:] {
			fieldSize *= v
		}
	}

	size := tensor.Shape(s).TotalSize()
	fanIn := float64((n1 + n2) * fieldSize)

	stdev := gain * math.Sqrt(2.0/fanIn)
	lo := 0.0 - math.Sqrt(3.0)*stdev
	hi := 0.0 + math.Sqrt(3.0)*stdev

	rand := rng.NewUniformGenerator(time.Now().UnixNano())
	retVal := make([]float64, size)
	for i := range retVal {
		retVal[i] = rand.Float64Range(lo, hi)
	}
	return retVal
}

// GlorotEtAlU32 returns float32 weights sampled from a uniform distribution
// using the methods specified in Glorot et. al (2010).
// See also: http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
//
// For best results, use:
// 		1.0 for gain for weights that will be used in linear and/or sigmoid units
//		math.Sqrt(2.0) for gain for weights that will be used in ReLU units
//		math.Sqrt(2.0 / (1+alpha*alpha)) for ReLU that are leaky with alpha
func GlorotEtAlU32(gain float64, s ...int) []float32 {
	f64 := GlorotEtAlN64(gain, s...)
	retVal := make([]float32, len(f64))
	for i, v := range f64 {
		retVal[i] = float32(v)
	}
	return retVal
}

// HeEtAlN64 returns float64 weights sampled from a normal distro, using the methods
// described in He et al (2015). The formula is:
//		randn(n) * sqrt(2/n)
// See also https://arxiv.org/abs/1502.01852
//
// For best results, use:
// 		1.0 for gain for weights that will be used in linear and/or sigmoid units
//		math.Sqrt(2.0) for gain for weights that will be used in ReLU units
//		math.Sqrt(2.0 / (1+alpha*alpha)) for ReLU that are leaky with alpha
func HeEtAlN64(gain float64, s ...int) []float64 {
	var fanIn float64

	switch len(s) {
	case 0, 1:
		panic("He et al only works with Tensors of dimensions >= 2")
	case 2:
		fanIn = float64(s[0])
	default:
		fanIn = 1.0
		for _, v := range s[1:] {
			fanIn *= float64(v)
		}
	}

	size := tensor.Shape(s).TotalSize()
	stdev := gain * math.Sqrt(1.0/fanIn)

	rand := rng.NewGaussianGenerator(time.Now().UnixNano())
	retVal := make([]float64, size)
	for i := range retVal {
		retVal[i] = rand.Gaussian(0.0, stdev)
	}
	return retVal
}

// HeEtAlU64 returns float64 weights sampled from a uniform distro, using the methods
// described in He et al (2015). The formula is:
//		randn(n) * sqrt(2/n)
// See also https://arxiv.org/abs/1502.01852
//
// For best results, use:
// 		1.0 for gain for weights that will be used in linear and/or sigmoid units
//		math.Sqrt(2.0) for gain for weights that will be used in ReLU units
//		math.Sqrt(2.0 / (1+alpha*alpha)) for ReLU that are leaky with alpha
func HeEtAlU64(gain float64, s ...int) []float64 {
	var fanIn float64

	switch len(s) {
	case 0, 1:
		panic("He et al only works with Tensors of dimensions >= 2")
	case 2:
		fanIn = float64(s[0])
	default:
		fanIn = 1.0
		for _, v := range s[1:] {
			fanIn *= float64(v)
		}
	}

	size := tensor.Shape(s).TotalSize()
	stdev := gain * math.Sqrt(1.0/fanIn)

	lo := 0.0 - math.Sqrt(3.0)*stdev
	hi := 0.0 + math.Sqrt(3.0)*stdev

	rand := rng.NewUniformGenerator(time.Now().UnixNano())
	retVal := make([]float64, size)
	for i := range retVal {
		retVal[i] = rand.Float64Range(lo, hi)
	}
	return retVal
}
