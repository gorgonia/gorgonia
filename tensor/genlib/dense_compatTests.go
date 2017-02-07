package main

import (
	"io"
	"text/template"
)

const compatTestsRaw = `var toMat64Tests = []struct{
	data interface{}
	sliced interface{}
	shape Shape
	dt Dtype
}{
	{{range .Kinds -}}
	{{if isNumber . -}}
	{ Range({{asType . | title | strip}}, 0, 6), []{{asType .}}{0,1,3,4}, Shape{2,3}, {{asType . | title | strip}} },
	{{end -}}
	{{end -}}
}
func TestToMat64(t *testing.T){
	assert := assert.New(t)
	for i, tmt := range toMat64Tests {
		T := New(WithBacking(tmt.data), WithShape(tmt.shape...))
		var m *mat64.Dense
		var err error
		if m, err = ToMat64(T); err != nil {
			t.Errorf("ToMat basic test %d failed : %v", i, err)
			continue
		}
		conv := anyToFloat64s(tmt.data)
		assert.Equal(conv, m.RawMatrix().Data, "i %d from %v", i, tmt.dt)

		if T, err = sliceDense(T, nil, makeRS(0, 2)); err != nil{
			t.Errorf("Slice failed %v", err)
			continue
		}
		if m, err = ToMat64(T); err != nil {
			t.Errorf("ToMat of slice test %d failed : %v", i, err)
			continue
		}
		conv = anyToFloat64s(tmt.sliced)
		assert.Equal(conv, m.RawMatrix().Data, "sliced test %d from %v", i, tmt.dt)
		t.Logf("Done")

		if tmt.dt == Float64 {
			T = New(WithBacking(tmt.data), WithShape(tmt.shape...))
			if m, err = ToMat64(T, UseUnsafe()); err != nil {
				t.Errorf("ToMat64 unsafe test %d failed: %v", i, err)
			}
			conv = anyToFloat64s(tmt.data)
			assert.Equal(conv, m.RawMatrix().Data, "float64 unsafe i %d from %v", i, tmt.dt)
			conv[0] = 1000
			assert.Equal(conv, m.RawMatrix().Data,"float64 unsafe i %d from %v", i, tmt.dt)
			conv[0] = 0 // reset for future tests that use the same backing
		}
	}
	// idiocy test
	T := New(Of(Float64), WithShape(2,3,4))
	_, err := ToMat64(T)
	if err == nil {
		t.Error("Expected an error when trying to convert a 3-T to *mat.Dense")
	}
}

func TestFromMat64(t *testing.T){
	assert := assert.New(t)
	var m *mat64.Dense
	var T *Dense
	var backing []float64


	for i, tmt := range toMat64Tests {
		backing = Range(Float64, 0, 6).([]float64)
		m = mat64.NewDense(2, 3, backing)
		T = FromMat64(m)
		conv := anyToFloat64s(tmt.data)
		assert.Equal(conv, T.float64s(), "test %d: []float64 from %v", i, tmt.dt)
		assert.True(T.Shape().Eq(tmt.shape))

		T = FromMat64(m, As(tmt.dt))
		assert.Equal(tmt.data, T.Data())
		assert.True(T.Shape().Eq(tmt.shape))

		if tmt.dt == Float64{
			backing = Range(Float64, 0, 6).([]float64)
			m = mat64.NewDense(2, 3, backing)
			T = FromMat64(m, UseUnsafe())
			assert.Equal(backing, T.float64s())
			assert.True(T.Shape().Eq(tmt.shape))
			backing[0] = 1000 
			assert.Equal(backing, T.float64s(), "test %d - unsafe float64", i)
		}
	}
}
`

var (
	compatTests *template.Template
)

func init() {
	compatTests = template.Must(template.New("testCompat").Funcs(funcs).Parse(compatTestsRaw))
}

func testCompat(f io.Writer, generic *ManyKinds) {
	compatTests.Execute(f, generic)
}
