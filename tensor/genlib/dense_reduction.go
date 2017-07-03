package main

import (
	"fmt"
	"io"
	"text/template"
)

const reductionHeader = `/*
This file contains code that deals with the reduction of a Tensor by axis.


All of the code in this file is structured in such a way that they're embarassingly parallel.
This message will serve as a reminder until all the code in this file which are embarassingly parallel
has been parallelized

List of functions parallalized:
	<crickets>

A visual explanation for the main reduction algorithm:
			Say you have a (2,3,2,3)-shaped tensor. It looks something like that:

				0  1  2		18 19 20
				3  4  5		21 22 23

				6  7  8		24 25 26
				9 10 11		27 28 29

				12 13 14	30 31 32
				15 16 17	33 34 35

			We'll consider only the first layer (0 - 17), since the same actions can be repeated upon the second layer

			Let's say we want to reduce axis 2. The resulting shape would be (2,3,3) (it's as simple as removing the second axis from the shape).
			This is how the matrix is laid out in the strided slice:

			t.data:
				0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15  16  17
				+   +   +   +   +   +   +   +   +   +    +   +  +   +   +    +   +   +
				|   |   |   |   |   |   |   |   |   |    |   |  |   |   |    |   |   |
				|   |   |   |   |   |   |   |   |   |    |   |  |   |   |    |   |   |
				+---------------------+-+-----------------------+   |   |    |   |   |
				    |   |   |   |   | |     |   |   |    |   |      |   |    |   |   |
				    +--------------------+--+-----------------------+   |    |   |   |
				        |   |   |   | |  |      |   |    |   |          |    |   |   |
				        +-----------------------------------------------+    |   |   |
				            |   |   | |  |      |   |    |   |               |   |   |
				            |   |   | +  +      +   |    |   |               |   |   |
			res.data index  |   |   | 0  1      2   |    |   |               |   |   |
				            |   |   |               |    |   |               |   |   |
				            +----------------------------+-+-----------------+   |   |
				                |   |               |      | |                   |   |
				                +------------------------------------------------+---+
				                    |               |      | |                       |
				                    +------------------------+-----+-----------------+
				                                    |      |       |
				                                    |      |       |
				                                    +      +       +
			res.data indes                          3      4       5

			It's a little difficult to see, but elements (0, 6, 12) from t.data will be written to index 0 of the reduced strided array. This is the listing:
				reduce (t[0], t[6], t[12]) -> res[0]
				reduce (t[1], t[7], t[13]) -> res[1]
				reduce (t[2], t[8], t[14]) -> res[2]
				...

			These are the basic rules:
				size of axis to be reduced  = number of elements to be reduced
				stride of axis to be reduced = how many to skip innerStart
				newStride[0] = expected number of groups within a layer

			The main idea is then this - we loop through the resulting array, and for each index, we find the elements of the original array that is supposed to fit in
			there, and then we reduce it. It is quite self explanatory.
*/


`
const funcHandlingHeader = `func reductionFnType(x interface{}, expectedType reflect.Type) (v reflect.Value, t reflect.Type, err error) {
	v = reflect.ValueOf(x)
	if v.Kind() != reflect.Func {
		err = errors.Errorf(extractionFail, "func(a, a) a", x)
		return
	}
	t = v.Type()
	if t.NumOut() != 1 {
		err = errors.Errorf("Expected one return value in reduction function")
		return
	}
	if t.Out(0) != expectedType {
		err = errors.Errorf("Expected return type of reduction function to be %v. Got %v instead", expectedType, t.Out(0))
		return
	}
	return
}

`

const reduceRaw = `// Reduce recursively applies a function f on the data along the provided axis. A default value has to be provided.
// The provided function must have this signature:
// 		func(T, T) T
// where T is the same as the Dtype of *Dense
func (t *Dense) Reduce(f interface{}, defaultValue interface{}, axis int) (retVal *Dense, err error){
	if axis >= t.Dims() {
		err = errors.Errorf(dimMismatch, axis, t.Dims())
		return
	}

	var newShape Shape
	for i, s := range t.Shape() {
		if i == axis {
			continue
		}
		newShape = append(newShape, s)
	}
	lastAxis := t.Dims() - 1
	retVal = New(Of(t.t), WithShape(newShape...))

	switch axis {
	case 0 :
		err = t.reduce0(retVal, f)
	case lastAxis:
		err = t.reduceLast(retVal , axis, f, defaultValue)
	default:
		err = t.reduceDefault(retVal, axis, f)
	}
	return
}

func (t *Dense) reduce0(retVal *Dense, fn interface{}) (err error) {
	size := t.Shape()[0]
	split := t.len() / size
	copySliced(retVal, 0, split, t, 0, split)
	start := split
		
	var ok bool
	switch t.t.Kind(){
	{{range .Kinds -}}
	{{if isParameterized . -}}
	{{else -}}
	case reflect.{{reflectKind .}}:
		var f func(a, b {{asType .}}) {{asType .}}
		if f, ok = fn.(func(a, b {{asType .}}) {{asType .}}); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b {{asType .}}) {{asType .}}. Got %v instead", fn)
		}

		data := retVal.{{sliceOf .}}
		for i := 0; i < size-1; i++ {
			for j := 0; j < split; j++ {
				data[j] = f(data[j], t.{{getOne .}}(j + start))
			}
			start += split
		}
	{{end -}}
	{{end -}}
	default:
		var f reflect.Value
		var fnT reflect.Type
		if f, fnT, err = reductionFnType(fn, t.t.Type); err != nil {
			return
		}

		args := make([]reflect.Value, 0, fnT.NumIn())
		for i := 0; i < size-1; i++ {
			for j := 0; j < split; j++ {
				args = append(args, reflect.ValueOf(retVal.Get(j)))
				args = append(args, reflect.ValueOf(t.Get(start + j)))
				v := f.Call(args)[0].Interface()
				retVal.Set(j, v)
				args = args[:0]
			}
			start += split
		}
	}
	return nil
}

func (t *Dense) reduceLast(retVal *Dense, axis int, fn interface{}, defaultValue interface{}) error {
	size := t.Shape()[axis]
	var at int
	var ok bool
	switch t.t.Kind() {
	{{range .Kinds -}}
	{{if isParameterized . -}}
	{{else -}}
	case reflect.{{reflectKind .}}:
		var f func(a, b {{asType .}}) {{asType .}}
		if f, ok = fn.(func(a, b {{asType .}}) {{asType .}}); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b {{asType .}}) {{asType .}}. Got %v instead", fn)
		}
		var def {{asType .}}
		if def, ok = defaultValue.({{asType .}}); !ok {
			return errors.Errorf("Expected default value to be {{asType .}}. Got %v of %T instead", defaultValue, defaultValue)
		}
		for start := 0; start <= t.len() - size; start += size {
			r := reduce{{short .}}(f, def, t.{{sliceOf .}}[start:start+size]...)
			retVal.{{setOne .}}(at, r)
			at++
		}	

	{{end -}}
	{{end -}}
	default:
		f, fnT, err := reductionFnType(fn, t.t.Type); 
		if err != nil {
			return err
		}
		def := reflect.ValueOf(defaultValue)
		for start := 0; start < t.len() - size; start += size {
			sliced := t.shallowClone()
			sliced.slice(start, start+size)
			r := reduceRef(f, fnT, def, sliced)
			retVal.Set(at, r)
			at++
		}
	}
	return nil
}

func (t *Dense) reduceDefault(retVal *Dense, axis int, fn interface{}) error {
	size := t.Shape()[axis]
	oStride := t.Strides()[0]
	stride := t.Strides()[axis]
	expected := retVal.Strides()[0]

	var ok bool
	switch t.t.Kind() {
	{{range .Kinds -}}
	{{if isParameterized . -}}
	{{else -}}
	case reflect.{{reflectKind .}}:
		var f func(a, b {{asType .}}) {{asType .}}
		if f, ok = fn.(func(a, b {{asType .}}) {{asType .}}); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b {{asType .}}) {{asType .}}. Got %v instead", fn)
		}
		for i := 0; i < t.Shape()[0]; i++ {
			// this loop can be parallelized!
			start := i * oStride
			tdata := t.{{sliceOf .}}[start : start+oStride]
			rdata := retVal.{{sliceOf .}}
			var innerStart, strideTrack int
			for j := 0; j < expected; j++ {
				for k := 0; k < size; k++ {
					readFrom := innerStart + k*stride
					writeTo := i*expected + j
					rdata[writeTo] = f(rdata[writeTo], tdata[readFrom])
				}
				strideTrack++
				if strideTrack >= stride {
					strideTrack = 0
					innerStart += stride
				}
				innerStart++
			}
		}
	{{end -}}
	{{end -}}
	default:
		f, fnT, err := reductionFnType(fn, t.t.Type); 
		if err != nil {
			return err
		}
		args := make([]reflect.Value, 0, fnT.NumIn())
		for i := 0; i < t.Shape()[0]; i++ {
			// this loop can be parallelized!
			start := i * oStride
			sliced := t.shallowClone()
			sliced.slice(start, start+oStride)

			var innerStart, strideTrack int
			for j := 0; j < expected; j++ {
				for k := 0; k < size; k++ {
					readFrom := innerStart + k*stride
					writeTo := i*expected + j
					args = append(args, reflect.ValueOf(retVal.Get(writeTo)))
					args = append(args, reflect.ValueOf(sliced.Get(readFrom)))
					v := f.Call(args)[0].Interface()
					retVal.Set(writeTo, v)
				}
				strideTrack++
				if strideTrack >= stride {
					strideTrack = 0
					innerStart += stride
				}
				innerStart++
			}
		}	
	}
	return nil
}

{{range .Kinds -}}
{{if isNumber . -}}
// sReduce{{short .}} is a specialization for {{asType .}} reductions, used in methods such as Sum, Prod, Max etc
func (t *Dense) sReduce{{short .}}(axis int, zeroFn func(a, b []{{asType .}}) error, lastFn func([]{{asType .}}) {{asType .}}, defFn func(a, b {{asType .}}) {{asType .}}) (retVal *Dense) {
	if t.IsScalar() {
		return t
	}

	var newShape Shape
	for i, s := range t.Shape() {
		if i == axis {
			continue
		}
		newShape = append(newShape, s)
	}
	retVal = New(Of(t.t), WithShape(newShape...))
	size := t.Shape()[axis]
	lastAxis := t.Dims() - 1
	switch axis {
	case 0:
		// most efficient
		split := t.len() / size
		copy(retVal.{{sliceOf .}}[0:split], t.{{sliceOf .}}[0:split])

		start := split
		for i := 0; i < size - 1 ; i++ {
			if err := zeroFn(retVal.{{sliceOf .}}, t.{{sliceOf .}}[start: start+split]); err != nil {
				panic(err)
			}
			start += split
		}
	case lastAxis:
		// second most efficient
		var at int
		for start := 0; start <= t.len() - size; start += size {
			retVal.{{setOne .}}(at, lastFn(t.{{sliceOf .}}[start: start+size]))
			at++
		}
	default:
		outerSize := t.Shape()[0]
		outerStride := t.Strides()[0]
		stride := t.Strides()[axis]
		expected := retVal.Strides()[0]
		
		for i := 0; i < outerSize; i++ {
			start := i * outerStride
			tdata := t.{{sliceOf .}}[start : start+outerStride]
			rdata := retVal.{{sliceOf .}}
			var innerStart, strideTrack int
			for j := 0; j < expected; j++ {
				for k := 0; k < size; k++ {
					readFrom := innerStart + k*stride
					writeTo := i*expected + j
					a := rdata[writeTo]
					b := tdata[readFrom]
					if k == 0 {
						rdata[writeTo] = b
					} else {
						rdata[writeTo] = defFn(a,b)
					}
				}
				strideTrack++
				if strideTrack >= stride {
					strideTrack = 0
					innerStart += stride
				}
				innerStart++
			}
		}
	}
	return retVal
}
{{end -}}
{{end -}}
`

var (
	reduce *template.Template
)

func init() {
	reduce = template.Must(template.New("reduce").Funcs(funcs).Parse(reduceRaw))
}

func reduction(f io.Writer, generic *ManyKinds) {
	fmt.Fprintln(f, reductionHeader)
	fmt.Fprintln(f, funcHandlingHeader)
	reduce.Execute(f, generic)
}
