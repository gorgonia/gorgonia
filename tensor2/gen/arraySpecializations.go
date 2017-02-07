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

const incrMapperRaw = `func (a {{.Name}}) MapIncr(fn interface{}) error {
	if f, ok := fn.(func({{.Of}}){{.Of}}); ok{
		for i, v := range a {
			a[i] += f(v)
		}
		return nil
	}
	return errors.Errorf(extractionFail, "func(x {{.Of}}){{.Of}}", fn)
}
`

const iterMapperRaw = `func (a {{.Name}}) IterMap(other Array, it, ot *FlatIterator, fn interface{}, incr bool) (err error) {
	// check noop 
	if other == nil  && ot == nil && fn == nil {
		return nil
	}

	// check types first
	var b []{{.Of}}
	if other != nil {
		if b, err = get{{title .Of}}s(other); err != nil{
			return
		}
	}

	var f func({{.Of}}){{.Of}}
	var ok bool
	if fn != nil {
		if f, ok  = fn.(func({{.Of}}){{.Of}}); !ok {
			return errors.Errorf(extractionFail, "func({{.Of}}){{.Of}}", f)
		}
	}

	switch {
	case other == nil && it == nil && ot == nil && fn != nil:
		// basic case: this is just a.Map(fn)
		return a.Map(f)
	case other == nil && it != nil && ot == nil && fn != nil:
		// basically this is apply function with iterator guidance
		var next int
		for next, err = it.Next(); err == nil; next, err = it.Next(){
			if _, noop := err.(NoOpError); err != nil && !noop{
				return 
			}
			{{if eq .Of "bool" -}}
				a[next] = f(a[next])
			{{else -}}
				if incr {
					a[next] += f(a[next])
				} else {
					a[next] = f(a[next])
				}
			{{end -}}
		}
		return nil
	case other != nil && it == nil && ot == nil:
		// the case where a[i] = b[i]
		if len(a) != len(b) {
			return errors.Errorf(sizeMismatch, len(a), len(b))
		}
		a = a[:len(a)] // optim for BCE
		b = b[:len(a)] // optim for BCE

		{{if eq .Of "bool" -}}
			if fn == nil {
				for i, v := range b {
					a[i] = v
				}
			} else {
				for i, v := range b{
					a[i] = f(v)
				}
			}
			return nil
		{{else -}}
			switch {
			case incr && fn == nil:
				for i, v := range b {
					a[i] += v
				}
				return nil
			case incr && fn != nil:
				for i, v := range b {
					a[i] += f(v)
				}
				return nil
			case !incr && fn == nil:
				for i, v := range b {
						a[i] = v
				}
			case !incr && fn != nil:
				for i, v := range b {
						a[i] = f(v)
				}
			}
		{{end -}}
	case other != nil && it != nil && ot == nil:
		// case where assignment of a = b; where a is guided by it
		var next, j int
		for next, err = it.Next(); err == nil; next, err = it.Next(){
			if _, noop := err.(NoOpError); err != nil && !noop{
				return 
			}

			{{if eq .Of "bool" -}}
				if fn == nil {
					a[next] = b[j]
				} else {
					a[next] = f(b[j])
				}
			{{else -}}
				switch {
				case incr && fn == nil:
					a[next] += b[j]
				case incr && fn != nil:
					a[next] += f(b[j])
				case !incr && fn == nil:
					a[next] = b[j]
				case !incr && fn != nil:
					a[next] = f(b[j])
				}
			{{end}}


			j++
		}
		return nil
	case other != nil && it == nil && ot != nil:
		// case where assignment of a = b; where b is guided by ot
		var next, i int
		for next, err = ot.Next(); err == nil; next, err = ot.Next(){
			if _, noop := err.(NoOpError); err != nil && !noop{
				return 
			}

			{{if eq .Of "bool" -}}
				if fn == nil {
					a[i] = b[next]
				} else{
					a[i] = f(b[next])
				}
			{{else -}}
				switch {
				case incr && fn == nil:
					a[i] += b[next]
				case incr && fn != nil:
					a[i] += f(b[next])
				case !incr && fn == nil:
					a[i] = b[next]
				case !incr && fn != nil:
					a[i] = f(b[next])
				}
			{{end}}

			i++
		}
		return nil
	case other != nil && it != nil && ot != nil:
		// case where assignment of a = b; and both a and b are guided by it and ot respectively
		var i, j int
		for {
			if i, err = it.Next(); err != nil {
				if _, ok := err.(NoOpError);  !ok{
					return err
				}
				err = nil
				break
			}
			if j, err = ot.Next(); err != nil {
				if _, ok := err.(NoOpError); !ok {
					return err
				}
				err = nil
				break
			}

			{{if eq .Of "bool" -}}
				if fn == nil {
					a[i] = b[j]
				} else {
					a[i] = f(b[j])
				}
			{{else -}}
				switch {
				case incr && fn == nil:
					a[i] += b[j]
				case incr && fn != nil:
					a[i] += f(b[j])
				case !incr && fn == nil:
					a[i] = b[j]
				case !incr && fn != nil:
					a[i] = f(b[j])
				}
			{{end}}

		}
		return nil
	case other == nil && ot != nil:
		// error - stupid
		return errors.Errorf("Meaningless state - other is nil, ot is not")
	}
	return
}

`

var (
	transposeTmpl  *template.Template
	incrMapperTmpl *template.Template
	iterMapperTmpl *template.Template
)

func init() {
	transposeTmpl = template.Must(template.New("Transpose").Parse(transposeRaw))
	incrMapperTmpl = template.Must(template.New("IncrMapper").Funcs(funcMap).Parse(incrMapperRaw))
	iterMapperTmpl = template.Must(template.New("IterMapper").Funcs(funcMap).Parse(iterMapperRaw))
}
