package stdops

import (
	"math/rand"
	"reflect"
	"testing/quick"

	"gorgonia.org/dtype"
	"gorgonia.org/shapes"
	"gorgonia.org/tensor"

	_ "unsafe"
)

var (
	intsliceType reflect.Type
	intType      reflect.Type
)

func init() {
	var a []int
	intsliceType = reflect.TypeOf(a)

	var b int
	intType = reflect.TypeOf(b)
}

/* THIS PORTION IS ADAPTED FROM testutils_test.go OF PACKAGE tensor */

var specializedTypes = []dtype.Dtype{
	dtype.Bool, dtype.Int, dtype.Int8, dtype.Int16, dtype.Int32, dtype.Int64, dtype.Uint, dtype.Uint8, dtype.Uint16, dtype.Uint32, dtype.Uint64, dtype.Float32, dtype.Float64, dtype.Complex64, dtype.Complex128, dtype.String,
}

//go:linkname divmod gorgonia.org/tensor.divmod
func divmod(a, b int) (q, r int)

func factorize(a int) []int {
	if a <= 0 {
		return nil
	}
	// all numbers are divisible by at least 1
	retVal := make([]int, 1)
	retVal[0] = 1

	fill := func(a int, e int) {
		n := len(retVal)
		for i, p := 0, a; i < e; i, p = i+1, p*a {
			for j := 0; j < n; j++ {
				retVal = append(retVal, retVal[j]*p)
			}
		}
	}
	// find factors of 2
	// rightshift by 1 = division by 2
	var e int
	for ; a&1 == 0; e++ {
		a >>= 1
	}
	fill(2, e)

	// find factors of 3 and up
	for next := 3; a > 1; next += 2 {
		if next*next > a {
			next = a
		}
		for e = 0; a%next == 0; e++ {
			a /= next
		}
		if e > 0 {
			fill(next, e)
		}
	}
	return retVal
}

func shuffleInts(a []int, r *rand.Rand) {
	for i := range a {
		j := r.Intn(i + 1)
		a[i], a[j] = a[j], a[i]
	}
}

// tTensor is a wrapped *tensor.Dense so it may implement quick.Generator.
type tTensor struct {
	*tensor.Dense
}

// Generate generates a tTensor. This is adapted from the *Dense.Generate in testutils_test.go in package tensor.
func (t tTensor) Generate(r *rand.Rand, size int) reflect.Value {
	// generate type
	ri := r.Intn(len(specializedTypes))
	of := specializedTypes[ri]
	datatyp := reflect.SliceOf(of.Type)
	gendat, _ := quick.Value(datatyp, r)
	// generate dims
	var scalar bool
	var s shapes.Shape
	dims := r.Intn(5) // dims4 is the max we'll generate even though we can handle much more
	l := gendat.Len()

	// generate shape based on inputs
	switch {
	case dims == 0 || l == 0:
		scalar = true
		gendat, _ = quick.Value(of.Type, r)
	case dims == 1:
		s = shapes.Shape{gendat.Len()}
	default:
		factors := factorize(l)
		s = shapes.Shape(tensor.BorrowInts(dims))
		// fill with 1s so that we can get a non-zero TotalSize
		for i := 0; i < len(s); i++ {
			s[i] = 1
		}

		for i := 0; i < dims; i++ {
			j := rand.Intn(len(factors))
			s[i] = factors[j]
			size := s.TotalSize()
			if q, r := divmod(l, size); r != 0 {
				factors = factorize(r)
			} else if size != l {
				if i < dims-2 {
					factors = factorize(q)
				} else if i == dims-2 {
					s[i+1] = q
					break
				}
			} else {
				break
			}
		}
		shuffleInts(s, r)
	}

	/*
		// generate flags
		flag := tensor.MemoryFlag(r.Intn(4))

		// generate order
		order := tensor.DataOrder(r.Intn(4))
	*/

	var v *tensor.Dense
	if scalar {
		v = tensor.New(tensor.FromScalar(gendat.Interface()))
	} else {
		v = tensor.New(tensor.Of(of), tensor.WithShape(s...), tensor.WithBacking(gendat.Interface()))
	}
	tensor.WithEngine(tensor.StdEng{})(v)

	// generate engine

	/*
		v.flag = flag
		v.AP.o = order

		// generate engine
		oeint := r.Intn(2)
		eint := r.Intn(4)
		switch eint {
		case 0:
			v.e = StdEng{}
			if oeint == 0 {
				v.oe = StdEng{}
			} else {
				v.oe = nil
			}
		case 1:
			// check is to prevent panics which Float64Engine will do if asked to allocate memory for non float64s
			if of == Float64 {
				v.e = Float64Engine{}
				if oeint == 0 {
					v.oe = Float64Engine{}
				} else {
					v.oe = nil
				}
			} else {
				v.e = StdEng{}
				if oeint == 0 {
					v.oe = StdEng{}
				} else {
					v.oe = nil
				}
			}
		case 2:
			// check is to prevent panics which Float64Engine will do if asked to allocate memory for non float64s
			if of == Float32 {
				v.e = Float32Engine{}
				if oeint == 0 {
					v.oe = Float32Engine{}
				} else {
					v.oe = nil
				}
			} else {
				v.e = StdEng{}
				if oeint == 0 {
					v.oe = StdEng{}
				} else {
					v.oe = nil
				}
			}
		case 3:
			v.e = dummyEngine(true)
			v.oe = nil
		}
	*/

	return reflect.ValueOf(tTensor{v})
}