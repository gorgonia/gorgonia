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

const writeCSVRaw = `// WriteCSV writes the *Dense to a CSV. It accepts an optional string formatting ("%v", "%f", etc...), which controls what is written to the CSV.
func (t *Dense) WriteCSV(w io.Writer, formats ...string) (err error) {
	// checks:
	if !t.IsMatrix() {
		// error
		err = errors.Errorf("Cannot write *Dense to CSV. Expected number of dimensions: <=2, T has got %d dimensions (Shape: %v)", t.Dims(), t.Shape())
		return
	}
	format := "%v"
	if len(formats) > 0{
		format = formats[0]
	}

	cw := csv.NewWriter(w)
	it := NewFlatIterator(t.AP)
	coord := it.Coord()

	// rows := t.Shape()[0]
	cols := t.Shape()[1]
	record := make([]string, 0, cols)
	var i, lastCol int
	for i, err = it.Next(); err == nil; i, err = it.Next() {
		record = append(record, fmt.Sprintf(format, t.Get(i)))
		if lastCol == cols-1 {
			if err = cw.Write(record); err != nil {
				// TODO: wrap errors
				return
			}
			cw.Flush()
			record = record[:0]
		}

		// cleanup
		switch {
		case t.IsRowVec():
			// lastRow = coord[len(coord)-2]
			lastCol = coord[len(coord)-1]
		case t.IsColVec():
			// lastRow = coord[len(coord)-1]
			lastCol = coord[len(coord)-2]
		case t.IsVector():
			lastCol = coord[len(coord)-1]
		default:
			// lastRow = coord[len(coord)-2]
			lastCol = coord[len(coord)-1]
		}
	}
	return nil
}

`

const gobEncodeRaw = `// GobEncode implements gob.GobEncoder
func (t *Dense) GobEncode() (p []byte, err error){
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

const gobDecodeRaw = `// GobDecode implements gob.GobDecoder
func (t *Dense) GobDecode(p []byte) (err error){
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

const readNpyRaw = `// ReadNpy reads NumPy formatted files into a *Dense
func (t *Dense) ReadNpy(r io.Reader) (err error){
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
		err = errors.New("Only verion 1.0 of numpy's serialization format is currently supported (65535 bytes ought to be enough for a header)")
		return
	}

	var minor byte
	if err = binary.Read(r, binary.LittleEndian, &minor); err != nil {
		return
	}
	if minor != 0 {
		err = errors.New("Only verion 1.0 of numpy's serialization format is currently supported (65535 bytes ought to be enough for a header)")
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
		err = errors.New("No dtype information in npy file")
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
		err = errors.New("No Row Order information found in the numpy file")
		return
	}
	if string(match[1]) != "False" {
		err = errors.New("Cannot yet read from Fortran Ordered Numpy files")
		return
	}

	shpRe := regexp.MustCompile(` + "`'shape':" + `\s*\(([^\(]*)\)` + "`" + `)
	match = shpRe.FindSubmatch(header)
	if match == nil {
		err = errors.New("No shape information found in npy file")
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

const readCSVRaw = `// convFromStrs conversts a []string to a slice of the Dtype provided
func convFromStrs(to Dtype, record []string) (interface{}, error) {
	var err error
	switch to.Kind() {
		{{range .Kinds -}}
		{{if isNumber . -}}
		{{if isOrd . -}}
	case reflect.{{reflectKind .}}:
		retVal := make([]{{asType .}}, len(record))
		for i, v := range record {
			{{if eq .String "float64" -}}
				if retVal[i], err = strconv.ParseFloat(v, 64); err != nil {
					return nil, err
				}
			{{else if eq .String "float32" -}}
				var f float64
				if f, err = strconv.ParseFloat(v, 32); err != nil {
					return nil, err
				}
				retVal[i] = float32(f)
			{{else if hasPrefix .String "int" -}}
				var i64 int64
				if i64, err = strconv.ParseInt(v, 10, {{bitSizeOf .}}); err != nil {
					return nil, err
				}
				retVal[i] = {{asType .}}(i64)
			{{else if hasPrefix .String "uint" -}}
				var u uint64
				if u, err = strconv.ParseUint(v, 10, {{bitSizeOf .}}); err != nil {
					return nil, err
				}
				retVal[i] = {{asType .}}(u)
			{{end -}}
		}
		return retVal, nil
		{{end -}}
		{{end -}}
		{{end -}}
	default:
		return nil,errors.Errorf(methodNYI, "convFromStrs", to)
	}
}

// ReadCSV reads a CSV into a *Dense. It will override the underlying data.
//
// BUG(chewxy): reading CSV doesn't handle CSVs with different columns per row yet.
func (t *Dense) ReadCSV(r io.Reader, opts ...FuncOpt) (err error) {
	fo := parseFuncOpts(opts...)
	as := fo.t
	if fo.t.Type == nil {
		as = Float64
	}

	cr := csv.NewReader(r)

	var record []string
	var row interface{}
	var rows, cols int

	switch as.Kind() {
		{{range .Kinds -}}
		{{if isNumber . -}}
		{{if isOrd . -}}
	case reflect.{{reflectKind .}}:
		backing := make([]{{asType .}}, 0)
		for {
			record, err = cr.Read()
			if err == io.EOF{
				break
			}

			if err != nil {
				return
			}

			if row, err = convFromStrs({{asType . | strip | title}}, record); err != nil {
				return
			}
			backing = append(backing, row.([]{{asType .}})...)
			cols = len(record)
			rows++
		}
		t.fromSlice(backing)
		t.AP = new(AP)
		t.AP.SetShape(rows, cols)
		return nil
		{{end -}}
		{{end -}}
		{{end -}}
	case reflect.String:
		backing := make([]string, 0)
		for {
			record, err = cr.Read()
			if err == io.EOF{
				break
			}

			if err != nil {
				return
			}
			backing = append(backing, record...)
			cols = len(record)
			rows++
		}
		t.fromSlice(backing)
		t.AP = new(AP)
		t.AP.SetShape(rows, cols)
		return nil
	default:
		return errors.Errorf("%v not yet handled", as)
	}
	return errors.Errorf("not yet handled")
}
`

var (
	readNpy   *template.Template
	gobEncode *template.Template
	gobDecode *template.Template
	readCSV   *template.Template
)

func init() {
	readNpy = template.Must(template.New("readNpy").Funcs(funcs).Parse(readNpyRaw))
	readCSV = template.Must(template.New("readCSV").Funcs(funcs).Parse(readCSVRaw))
	gobEncode = template.Must(template.New("gobEncode").Funcs(funcs).Parse(gobEncodeRaw))
	gobDecode = template.Must(template.New("gobDecode").Funcs(funcs).Parse(gobDecodeRaw))
}

func generateDenseIO(f io.Writer, generic *ManyKinds) {
	mk := &ManyKinds{Kinds: filter(generic.Kinds, isNumber)}

	// writes
	fmt.Fprintln(f, writeNpyRaw)
	fmt.Fprintln(f, "\n")
	fmt.Fprintln(f, writeCSVRaw)
	fmt.Fprintln(f, "\n")
	gobEncode.Execute(f, mk)

	// reads
	readNpy.Execute(f, mk)
	gobDecode.Execute(f, mk)
	readCSV.Execute(f, mk)
}
