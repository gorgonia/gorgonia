package main

import (
	"io"
	"text/template"
)

const onesRaw = `// Ones creates a *Dense with the provided shape and type
func Ones(dt Dtype, shape ...int) *Dense {
	d := recycledDense(dt, shape)
	switch d.t.Kind() {
		{{range .Kinds -}}
		{{if isNumber . -}}
		case reflect.{{reflectKind .}}:
			d.Memset({{asType .}}(1))
		{{end -}}
		{{end -}}
	case reflect.Bool:
		d.Memset(true)
	default:
		// TODO: add a Oner interface
	}
	return d
}
`

const Iraw = `// I creates the identity matrix (usually a square) matrix with 1s across the diagonals, and zeroes elsewhere, like so:
//		Matrix(4,4)
// 		⎡1  0  0  0⎤
// 		⎢0  1  0  0⎥
// 		⎢0  0  1  0⎥
// 		⎣0  0  0  1⎦
// While technically an identity matrix is a square matrix, in attempt to keep feature parity with Numpy,
// the I() function allows you to create non square matrices, as well as an index to start the diagonals.
//
// For example:
//		T = I(Float64, 4, 4, 1)
// Yields:
//		⎡0  1  0  0⎤
//		⎢0  0  1  0⎥
//		⎢0  0  0  1⎥
//		⎣0  0  0  0⎦
//
// The index k can also be a negative number:
// 		T = I(Float64, 4, 4, -1)
// Yields:
// 		⎡0  0  0  0⎤
// 		⎢1  0  0  0⎥
// 		⎢0  1  0  0⎥
// 		⎣0  0  1  0⎦
func I(dt Dtype, r, c, k int) *Dense{
	ret := New(Of(dt), WithShape(r,c))
	i := k
	if k < 0 {
		i = (-k) * c
	}

	var s *Dense
	var err error
	end := c - k
	if end > r {
		s, err = sliceDense(ret, nil)
	} else {
		s, err = sliceDense(ret, rs{0, end, 1})
	}

	if err != nil {
		panic(err)
	}
	var nexts []int
	iter := NewFlatIterator(s.AP)
	nexts, err = iter.Slice(rs{i, s.Size(), c + 1})

	switch s.t.Kind() {
		{{range .Kinds -}}
		{{if isNumber . -}}
		case reflect.{{reflectKind .}}:
			data := s.{{sliceOf .}}
			for _, v := range nexts {
				data[v] = 1
			}
		{{end -}}
		{{end -}}
	}
	// TODO: create Oner interface for custom types
	return ret
}
`

var (
	ones *template.Template
	eye  *template.Template
)

func init() {
	ones = template.Must(template.New("ones").Funcs(funcs).Parse(onesRaw))
	eye = template.Must(template.New("eye").Funcs(funcs).Parse(Iraw))
}

func generateDenseConstructionFns(f io.Writer, generic Kinds) {
	ones.Execute(f, generic)
	eye.Execute(f, generic)
}
