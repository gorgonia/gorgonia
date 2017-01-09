package main

import "text/template"

const transposeRaw = `func (a {{.Name}}) Transpose(oldShape, oldStrides, axes, newStrides []int) {
	size := len(a)
	track := NewBitMap(size)
	track.Set(0)
	track.Set(size - 1) // first and last don't change

	var saved, tmp {{.Of}}
	var i int

	for i = 1 ;; {
		dest := TransposeIndex(i, oldShape, axes, oldStrides, newStrides)
		
		if track.IsSet(i) && track.IsSet(dest) {
			a[i] = saved
			saved = {{.DefaultZero}}

			for i < size && track.IsSet(i) {
				i++
			}

			if i >= size {
				break
			}
			continue
		}

		track.Set(i)
		tmp = a[i]
		a[i] = saved
		saved = tmp

		i = dest
	}
}
`

var (
	transposeTmpl *template.Template
)

func init() {
	transposeTmpl = template.Must(template.New("Transpose").Parse(transposeRaw))
}
