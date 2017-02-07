package main

import (
	"fmt"
	"io"
	"text/template"
)

const writeNpyRaw = `
type binaryWriter struct {
	io.Writer
	error
	seq int
}

func (w binaryWriter) w(x interface{}) {
	if w.error != nil {
		return
	}

	binary.Write(w, binary.LittleEndian, x)
	w.seq++
}

func (w binaryWriter) Error() string {
	return fmt.Sprintf("Error at sequence %d : %v", w.seq, w.error.Error())	
}

// WriteNpy writes the *Tensor as a numpy compatible serialized file.
//
// The format is very well documented here:
// http://docs.scipy.org/doc/numpy/neps/npy-format.html
//
// Gorgonia specifically uses Version 1.0, as 65535 bytes should be more than enough for the headers.
// The values are written in little endian order, because let's face it - 
// 90% of the world's computers are running on x86+ processors.
//
// This method does not close the writer. Closing (if needed) is deferred to the caller
func (t *Dense) WriteNpy(w io.Writer) (err error) {
	var npdt string
	if npdt, err = t.t.numpyDtype(); err != nil{
		return 
	}

	header := "{'descr': '<%v', 'fortran_order': False, 'shape': %v}"
	header = fmt.Sprintf(header, npdt, t.Shape())
	padding := 16 - ((10 + len(header)) % 16)
	if padding > 0 {
		header = header + strings.Repeat(" ", padding)
	}
	bw := binaryWriter{Writer: w}
	bw.Write([]byte("\x93NUMPY"))                              // stupid magic
	bw.w(byte(1))             // major version
	bw.w(byte(0))             // minor version
	bw.w(uint16(len(header))) // 4 bytes to denote header length
	if bw.error != nil {
		return bw
	}
	bw.Write([]byte(header))

	bw.seq = 0
	for i := 0; i < t.len(); i++ {
		bw.w(t.Get(i))
	}

	if bw.error != nil {
		return bw
	}
	return nil
}
`

const gobEncodeRaw = `func (t *Dense) GobEncode() (p []byte, err error){
	var buf bytes.Buffer
	encoder := gob.NewEncoder(&buf)
	if err = encoder.Encode(t.t.id()); err != nil {
		return
	}

	if err = encoder.Encode(t.Shape()); err != nil {
		return
	}

	if err = encoder.Encode(t.Strides()); err != nil {
		return
	}

	switch t.t.Kind() {
	{{range .Kinds -}}
	case reflect.{{reflectKind .}}:
		if err = encoder.Encode(t.{{sliceOf .}}); err != nil {
			return
		}
	{{end -}}
	case reflect.String:
		if err = encoder.Encode(t.strings()); err != nil {
			return
		}
	}

	return buf.Bytes(), err
}
`

const gobDecodeRaw = `func (t *Dense) GobDecode(p []byte) (err error){
	buf := bytes.NewBuffer(p)
	decoder := gob.NewDecoder(buf)

	var dt Dtype
	var id int
	if err = decoder.Decode(&id); err != nil {
		return
	}
	if dt, err = fromTypeID(id); err != nil {
		return
	}
	t.t = dt

	var shape Shape
	if err = decoder.Decode(&shape); err != nil {
		return
	}

	var strides []int
	if err = decoder.Decode(&strides); err != nil {
		return
	}
	t.AP = NewAP(shape, strides)

	switch dt.Kind() {
	{{range .Kinds -}}
	case reflect.{{reflectKind . -}}:
		var data []{{asType .}}
		if err = decoder.Decode(&data); err != nil {
			return
		}
		t.fromSlice(data)
	{{end -}}
	case reflect.String:
		var data []string
		if err = decoder.Decode(&data); err != nil {
			return
		}
		t.fromSlice(data)
	}
	t.fix()
	return t.sanity()
}
`

const readNpyRaw = `func (t *Dense) ReadNpy(r io.Reader) (err error){
	var magic [6]byte
	if _, err = r.Read(magic[:]); err != nil {
		return
	}
	if string(magic[:]) != "\x93NUMPY" {
		err = errors.Errorf("Not a numpy file. Got %q as the magic number instead", string(magic[:]))
		return
	}

	var version byte
	if err = binary.Read(r, binary.LittleEndian, &version); err != nil {
		return
	}
	if version != 1 {
		err = types.NewError(types.IOError, "Only version 1 of numpy's serialization is currently supported")
		return
	}

	var minor byte
	if err = binary.Read(r, binary.LittleEndian, &minor); err != nil {
		return
	}
	if minor != 0 {
		err = types.NewError(types.IOError, "Only version 1.0 of numpy's serialization is currently supported")
		return
	}

	var headerLen uint16
	if err = binary.Read(r, binary.LittleEndian, &headerLen); err != nil {
		return
	}

	header := make([]byte, int(headerLen))
	if _, err = r.Read(header); err != nil {
		return
	}

	desc := regexp.MustCompile(` + "`'descr':" + `\s` + "*'([^']*)'`" + `)
	match := desc.FindSubmatch(header)
	if match == nil {
		err = types.NewError(types.IOError, "No dtype information found")
		return
	}

	// TODO: check for endianness. For now we assume everything is little endian
	var dt Dtype
	if dt, err = fromNumpyDtype(string(match[1][1:])); err != nil {
		return
	}
	t.t = dt

	rowOrder := regexp.MustCompile(` + "`'fortran_order':" + `\s` + "*(False|True)`" + `)
	match = rowOrder.FindSubmatch(header)
	if match == nil {
		err = errors.Errorf("No Row Order information found in the numpy file")
		return
	}
	if string(match[1]) != "False" {
		err = errors.Errorf("Cannot yet read from Fortran Ordered Numpy files")
		return
	}

	shpRe := regexp.MustCompile(` + "`'shape':" + `\s*\(([^\(]*)\)` + "`" + `)
	match = shpRe.FindSubmatch(header)
	if match == nil {
		err = errors.Errorf("No shape information found")
		return
	}
	sizesStr := strings.Split(string(match[1]), ",")
	var shape Shape
	for _, s := range sizesStr {
		s = strings.Trim(s, " ")
		if len(s) == 0 {
			break
		}
		var size int
		if size, err = strconv.Atoi(s); err != nil {
			return
		}
		shape = append(shape, size)
	}

	size := shape.TotalSize()
	t.makeArray(size)

	switch t.t.Kind() {
	{{range .Kinds -}}
	case reflect.{{reflectKind .}}:
		data := t.{{sliceOf .}}
		for i := 0; i < size; i++ {
			if err = binary.Read(r, binary.LittleEndian, &data[i]); err != nil{
				return
			}
		}
	{{end -}}
	}
	t.AP = BorrowAP(len(shape))
	t.setShape(shape...)
	t.fix()
	return t.sanity()
}
`

var (
	readNpy   *template.Template
	gobEncode *template.Template
	gobDecode *template.Template
)

func init() {
	readNpy = template.Must(template.New("readNpy").Funcs(funcs).Parse(readNpyRaw))
	gobEncode = template.Must(template.New("gobEncode").Funcs(funcs).Parse(gobEncodeRaw))
	gobDecode = template.Must(template.New("gobDecode").Funcs(funcs).Parse(gobDecodeRaw))
}

func generateDenseIO(f io.Writer, generic *ManyKinds) {
	mk := &ManyKinds{Kinds: filter(generic.Kinds, isNumber)}

	// writes
	fmt.Fprintln(f, writeNpyRaw)
	gobEncode.Execute(f, mk)

	// reads
	readNpy.Execute(f, mk)
	gobDecode.Execute(f, mk)
}
