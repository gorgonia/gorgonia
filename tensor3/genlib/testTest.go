package main

import (
	"fmt"
	"io"
	"text/template"
)

const testHeaderRaw = `func randomBool() bool {
	i := rand.Intn(11)
	return i > 5
}

// from : https://stackoverflow.com/a/31832326/3426066
const letterBytes = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
const (
    letterIdxBits = 6                    // 6 bits to represent a letter index
    letterIdxMask = 1<<letterIdxBits - 1 // All 1-bits, as many as letterIdxBits
    letterIdxMax  = 63 / letterIdxBits   // # of letter indices fitting in 63 bits
)
var src = rand.NewSource(time.Now().UnixNano())

func randomString() string {
	n := rand.Intn(10)
    b := make([]byte, n)
    // A src.Int63() generates 63 random bits, enough for letterIdxMax characters!
    for i, cache, remain := n-1, src.Int63(), letterIdxMax; i >= 0; {
        if remain == 0 {
            cache, remain = src.Int63(), letterIdxMax
        }
        if idx := int(cache & letterIdxMask); idx < len(letterBytes) {
            b[i] = letterBytes[idx]
            i--
        }
        cache >>= letterIdxBits
        remain--
    }

    return string(b)
}

`

const testQCRaw = `type QCDense{{short .}} struct {
	*Dense 
}
func (q *QCDense{{short .}}) D() *Dense{return q.Dense}
func (*QCDense{{short .}}) Generate(r *rand.Rand, size int) reflect.Value {
	s := make([]{{asType .}}, size)
	for i := range s {
		{{if hasPrefix .String "uint" -}}
			s[i] = {{asType .}}(r.Uint32())
		{{else if hasPrefix .String "int" -}}
			s[i] = {{asType .}}(r.Int())
		{{else if eq .String "float64" -}}
			s[i] = r.Float64()
		{{else if eq .String "float32" -}}
			s[i] = r.Float32()
		{{else if eq .String "complex64" -}}
			s[i] = complex(r.Float32(), r.Float32())
		{{else if eq .String "complex128" -}}
			s[i] = complex(r.Float64(), r.Float64())
		{{else if eq .String "bool" -}}
			s[i] = randomBool()
		{{else if eq .String "string" -}}
			s[i] = randomString()
		{{else if eq .String "unsafe.Pointer" -}}
			s[i] = nil
		{{end -}}
	}
	d := recycledDense({{asType . | title | strip}}, Shape{size}, WithBacking(s))
	q := new(QCDense{{short .}})
	q.Dense = d
	return reflect.ValueOf(q)
}
`

var (
	testHeader *template.Template
	testQC     *template.Template
)

func init() {
	testQC = template.Must(template.New("testQCs").Funcs(funcs).Parse(testQCRaw))
	testHeader = template.Must(template.New("TestHeader").Funcs(funcs).Parse(testHeaderRaw))
}

func testtest(f io.Writer, generic *ManyKinds) {
	testHeader.Execute(f, generic)
	fmt.Fprint(f, "\n")
	for _, k := range generic.Kinds {
		if !isParameterized(k) {
			testQC.Execute(f, k)
			fmt.Fprint(f, "\n")
		}
	}
}
