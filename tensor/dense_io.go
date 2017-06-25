package tensor

import (
	"bytes"
	"encoding/binary"
	"encoding/csv"
	"encoding/gob"
	"fmt"
	"io"
	"reflect"
	"regexp"
	"strconv"
	"strings"

	"github.com/pkg/errors"
)

/*
GENERATED FILE. DO NOT EDIT
*/

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
// If tensor is masked, invalid values are replaced by the default fill value.
func (t *Dense) WriteNpy(w io.Writer) (err error) {
	var npdt string
	if npdt, err = t.t.numpyDtype(); err != nil {
		return
	}

	header := "{'descr': '<%v', 'fortran_order': False, 'shape': %v}"
	header = fmt.Sprintf(header, npdt, t.Shape())
	padding := 16 - ((10 + len(header)) % 16)
	if padding > 0 {
		header = header + strings.Repeat(" ", padding)
	}
	bw := binaryWriter{Writer: w}
	bw.Write([]byte("\x93NUMPY")) // stupid magic
	bw.w(byte(1))                 // major version
	bw.w(byte(0))                 // minor version
	bw.w(uint16(len(header)))     // 4 bytes to denote header length
	if bw.error != nil {
		return bw
	}
	bw.Write([]byte(header))

	bw.seq = 0
	if t.IsMasked() {
		fillval := t.FillValue()
		it := FlatMaskedIteratorFromDense(t)
		for i, err := it.Next(); err == nil; i, err = it.Next() {
			if t.mask[i] {
				bw.w(fillval)
			} else {
				bw.w(t.Get(i))
			}
		}
	} else {
		for i := 0; i < t.len(); i++ {
			bw.w(t.Get(i))
		}
	}

	if bw.error != nil {
		return bw
	}
	return nil
}

// WriteCSV writes the *Dense to a CSV. It accepts an optional string formatting ("%v", "%f", etc...), which controls what is written to the CSV.
// If tensor is masked, invalid values are replaced by the default fill value.
func (t *Dense) WriteCSV(w io.Writer, formats ...string) (err error) {
	// checks:
	if !t.IsMatrix() {
		// error
		err = errors.Errorf("Cannot write *Dense to CSV. Expected number of dimensions: <=2, T has got %d dimensions (Shape: %v)", t.Dims(), t.Shape())
		return
	}
	format := "%v"
	if len(formats) > 0 {
		format = formats[0]
	}

	cw := csv.NewWriter(w)
	it := IteratorFromDense(t)
	coord := it.Coord()

	// rows := t.Shape()[0]
	cols := t.Shape()[1]
	record := make([]string, 0, cols)
	var i, k, lastCol int
	isMasked := t.IsMasked()
	fillval := t.FillValue()
	fillstr := fmt.Sprintf(format, fillval)
	for i, err = it.Next(); err == nil; i, err = it.Next() {
		record = append(record, fmt.Sprintf(format, t.Get(i)))
		if isMasked {
			if t.mask[i] {
				record[k] = fillstr
			}
			k++
		}
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

// GobEncode implements gob.GobEncoder
func (t *Dense) GobEncode() (p []byte, err error) {
	var buf bytes.Buffer
	encoder := gob.NewEncoder(&buf)

	if err = encoder.Encode(t.Shape()); err != nil {
		return
	}

	if err = encoder.Encode(t.Strides()); err != nil {
		return
	}

	if err = encoder.Encode(t.AP.o); err != nil {
		return
	}

	if err = encoder.Encode(t.AP.Δ); err != nil {
		return
	}

	if err = encoder.Encode(t.mask); err != nil {
		return
	}

	data := t.Data()
	if err = encoder.Encode(&data); err != nil {
		return
	}

	return buf.Bytes(), err
}

// ReadNpy reads NumPy formatted files into a *Dense
func (t *Dense) ReadNpy(r io.Reader) (err error) {
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

	desc := regexp.MustCompile(`'descr':\s*'([^']*)'`)
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

	rowOrder := regexp.MustCompile(`'fortran_order':\s*(False|True)`)
	match = rowOrder.FindSubmatch(header)
	if match == nil {
		err = errors.New("No Row Order information found in the numpy file")
		return
	}
	if string(match[1]) != "False" {
		err = errors.New("Cannot yet read from Fortran Ordered Numpy files")
		return
	}

	shpRe := regexp.MustCompile(`'shape':\s*\(([^\(]*)\)`)
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
	case reflect.Int:
		data := t.ints()
		for i := 0; i < size; i++ {
			if err = binary.Read(r, binary.LittleEndian, &data[i]); err != nil {
				return
			}
		}
	case reflect.Int8:
		data := t.int8s()
		for i := 0; i < size; i++ {
			if err = binary.Read(r, binary.LittleEndian, &data[i]); err != nil {
				return
			}
		}
	case reflect.Int16:
		data := t.int16s()
		for i := 0; i < size; i++ {
			if err = binary.Read(r, binary.LittleEndian, &data[i]); err != nil {
				return
			}
		}
	case reflect.Int32:
		data := t.int32s()
		for i := 0; i < size; i++ {
			if err = binary.Read(r, binary.LittleEndian, &data[i]); err != nil {
				return
			}
		}
	case reflect.Int64:
		data := t.int64s()
		for i := 0; i < size; i++ {
			if err = binary.Read(r, binary.LittleEndian, &data[i]); err != nil {
				return
			}
		}
	case reflect.Uint:
		data := t.uints()
		for i := 0; i < size; i++ {
			if err = binary.Read(r, binary.LittleEndian, &data[i]); err != nil {
				return
			}
		}
	case reflect.Uint8:
		data := t.uint8s()
		for i := 0; i < size; i++ {
			if err = binary.Read(r, binary.LittleEndian, &data[i]); err != nil {
				return
			}
		}
	case reflect.Uint16:
		data := t.uint16s()
		for i := 0; i < size; i++ {
			if err = binary.Read(r, binary.LittleEndian, &data[i]); err != nil {
				return
			}
		}
	case reflect.Uint32:
		data := t.uint32s()
		for i := 0; i < size; i++ {
			if err = binary.Read(r, binary.LittleEndian, &data[i]); err != nil {
				return
			}
		}
	case reflect.Uint64:
		data := t.uint64s()
		for i := 0; i < size; i++ {
			if err = binary.Read(r, binary.LittleEndian, &data[i]); err != nil {
				return
			}
		}
	case reflect.Float32:
		data := t.float32s()
		for i := 0; i < size; i++ {
			if err = binary.Read(r, binary.LittleEndian, &data[i]); err != nil {
				return
			}
		}
	case reflect.Float64:
		data := t.float64s()
		for i := 0; i < size; i++ {
			if err = binary.Read(r, binary.LittleEndian, &data[i]); err != nil {
				return
			}
		}
	case reflect.Complex64:
		data := t.complex64s()
		for i := 0; i < size; i++ {
			if err = binary.Read(r, binary.LittleEndian, &data[i]); err != nil {
				return
			}
		}
	case reflect.Complex128:
		data := t.complex128s()
		for i := 0; i < size; i++ {
			if err = binary.Read(r, binary.LittleEndian, &data[i]); err != nil {
				return
			}
		}
	}
	t.AP = BorrowAP(len(shape))
	t.setShape(shape...)
	t.fix()
	return t.sanity()
}

// GobDecode implements gob.GobDecoder
func (t *Dense) GobDecode(p []byte) (err error) {
	buf := bytes.NewBuffer(p)
	decoder := gob.NewDecoder(buf)

	var shape Shape
	if err = decoder.Decode(&shape); err != nil {
		return
	}

	var strides []int
	if err = decoder.Decode(&strides); err != nil {
		return
	}

	var o DataOrder
	var tr Triangle
	if err = decoder.Decode(&o); err == nil {
		if err = decoder.Decode(&tr); err != nil {
			return
		}
	}

	t.AP = NewAP(shape, strides)
	t.AP.o = o
	t.AP.Δ = tr

	var mask []bool
	if err = decoder.Decode(&mask); err != nil {
		return
	}

	var data interface{}
	if err = decoder.Decode(&data); err != nil {
		return
	}
	t.fromSlice(data)
	t.addMask(mask)
	t.fix()
	return t.sanity()
}

// convFromStrs conversts a []string to a slice of the Dtype provided
func convFromStrs(to Dtype, record []string) (interface{}, error) {
	var err error
	switch to.Kind() {
	case reflect.Int:
		retVal := make([]int, len(record))
		for i, v := range record {
			var i64 int64
			if i64, err = strconv.ParseInt(v, 10, 0); err != nil {
				return nil, err
			}
			retVal[i] = int(i64)
		}
		return retVal, nil
	case reflect.Int8:
		retVal := make([]int8, len(record))
		for i, v := range record {
			var i64 int64
			if i64, err = strconv.ParseInt(v, 10, 8); err != nil {
				return nil, err
			}
			retVal[i] = int8(i64)
		}
		return retVal, nil
	case reflect.Int16:
		retVal := make([]int16, len(record))
		for i, v := range record {
			var i64 int64
			if i64, err = strconv.ParseInt(v, 10, 16); err != nil {
				return nil, err
			}
			retVal[i] = int16(i64)
		}
		return retVal, nil
	case reflect.Int32:
		retVal := make([]int32, len(record))
		for i, v := range record {
			var i64 int64
			if i64, err = strconv.ParseInt(v, 10, 32); err != nil {
				return nil, err
			}
			retVal[i] = int32(i64)
		}
		return retVal, nil
	case reflect.Int64:
		retVal := make([]int64, len(record))
		for i, v := range record {
			var i64 int64
			if i64, err = strconv.ParseInt(v, 10, 64); err != nil {
				return nil, err
			}
			retVal[i] = int64(i64)
		}
		return retVal, nil
	case reflect.Uint:
		retVal := make([]uint, len(record))
		for i, v := range record {
			var u uint64
			if u, err = strconv.ParseUint(v, 10, 0); err != nil {
				return nil, err
			}
			retVal[i] = uint(u)
		}
		return retVal, nil
	case reflect.Uint8:
		retVal := make([]uint8, len(record))
		for i, v := range record {
			var u uint64
			if u, err = strconv.ParseUint(v, 10, 8); err != nil {
				return nil, err
			}
			retVal[i] = uint8(u)
		}
		return retVal, nil
	case reflect.Uint16:
		retVal := make([]uint16, len(record))
		for i, v := range record {
			var u uint64
			if u, err = strconv.ParseUint(v, 10, 16); err != nil {
				return nil, err
			}
			retVal[i] = uint16(u)
		}
		return retVal, nil
	case reflect.Uint32:
		retVal := make([]uint32, len(record))
		for i, v := range record {
			var u uint64
			if u, err = strconv.ParseUint(v, 10, 32); err != nil {
				return nil, err
			}
			retVal[i] = uint32(u)
		}
		return retVal, nil
	case reflect.Uint64:
		retVal := make([]uint64, len(record))
		for i, v := range record {
			var u uint64
			if u, err = strconv.ParseUint(v, 10, 64); err != nil {
				return nil, err
			}
			retVal[i] = uint64(u)
		}
		return retVal, nil
	case reflect.Float32:
		retVal := make([]float32, len(record))
		for i, v := range record {
			var f float64
			if f, err = strconv.ParseFloat(v, 32); err != nil {
				return nil, err
			}
			retVal[i] = float32(f)
		}
		return retVal, nil
	case reflect.Float64:
		retVal := make([]float64, len(record))
		for i, v := range record {
			if retVal[i], err = strconv.ParseFloat(v, 64); err != nil {
				return nil, err
			}
		}
		return retVal, nil
	default:
		return nil, errors.Errorf(methodNYI, "convFromStrs", to)
	}
}

// ReadCSV reads a CSV into a *Dense. It will override the underlying data.
//
// BUG(chewxy): reading CSV doesn't handle CSVs with different columns per row yet.
func (t *Dense) ReadCSV(r io.Reader, opts ...FuncOpt) (err error) {
	fo := ParseFuncOpts(opts...)
	as := fo.As()
	if as.Type == nil {
		as = Float64
	}

	cr := csv.NewReader(r)

	var record []string
	var row interface{}
	var rows, cols int

	switch as.Kind() {
	case reflect.Int:
		var backing []int
		for {
			record, err = cr.Read()
			if err == io.EOF {
				break
			}

			if err != nil {
				return
			}

			if row, err = convFromStrs(Int, record); err != nil {
				return
			}
			backing = append(backing, row.([]int)...)
			cols = len(record)
			rows++
		}
		t.fromSlice(backing)
		t.AP = new(AP)
		t.AP.SetShape(rows, cols)
		return nil
	case reflect.Int8:
		var backing []int8
		for {
			record, err = cr.Read()
			if err == io.EOF {
				break
			}

			if err != nil {
				return
			}

			if row, err = convFromStrs(Int8, record); err != nil {
				return
			}
			backing = append(backing, row.([]int8)...)
			cols = len(record)
			rows++
		}
		t.fromSlice(backing)
		t.AP = new(AP)
		t.AP.SetShape(rows, cols)
		return nil
	case reflect.Int16:
		var backing []int16
		for {
			record, err = cr.Read()
			if err == io.EOF {
				break
			}

			if err != nil {
				return
			}

			if row, err = convFromStrs(Int16, record); err != nil {
				return
			}
			backing = append(backing, row.([]int16)...)
			cols = len(record)
			rows++
		}
		t.fromSlice(backing)
		t.AP = new(AP)
		t.AP.SetShape(rows, cols)
		return nil
	case reflect.Int32:
		var backing []int32
		for {
			record, err = cr.Read()
			if err == io.EOF {
				break
			}

			if err != nil {
				return
			}

			if row, err = convFromStrs(Int32, record); err != nil {
				return
			}
			backing = append(backing, row.([]int32)...)
			cols = len(record)
			rows++
		}
		t.fromSlice(backing)
		t.AP = new(AP)
		t.AP.SetShape(rows, cols)
		return nil
	case reflect.Int64:
		var backing []int64
		for {
			record, err = cr.Read()
			if err == io.EOF {
				break
			}

			if err != nil {
				return
			}

			if row, err = convFromStrs(Int64, record); err != nil {
				return
			}
			backing = append(backing, row.([]int64)...)
			cols = len(record)
			rows++
		}
		t.fromSlice(backing)
		t.AP = new(AP)
		t.AP.SetShape(rows, cols)
		return nil
	case reflect.Uint:
		var backing []uint
		for {
			record, err = cr.Read()
			if err == io.EOF {
				break
			}

			if err != nil {
				return
			}

			if row, err = convFromStrs(Uint, record); err != nil {
				return
			}
			backing = append(backing, row.([]uint)...)
			cols = len(record)
			rows++
		}
		t.fromSlice(backing)
		t.AP = new(AP)
		t.AP.SetShape(rows, cols)
		return nil
	case reflect.Uint8:
		var backing []uint8
		for {
			record, err = cr.Read()
			if err == io.EOF {
				break
			}

			if err != nil {
				return
			}

			if row, err = convFromStrs(Uint8, record); err != nil {
				return
			}
			backing = append(backing, row.([]uint8)...)
			cols = len(record)
			rows++
		}
		t.fromSlice(backing)
		t.AP = new(AP)
		t.AP.SetShape(rows, cols)
		return nil
	case reflect.Uint16:
		var backing []uint16
		for {
			record, err = cr.Read()
			if err == io.EOF {
				break
			}

			if err != nil {
				return
			}

			if row, err = convFromStrs(Uint16, record); err != nil {
				return
			}
			backing = append(backing, row.([]uint16)...)
			cols = len(record)
			rows++
		}
		t.fromSlice(backing)
		t.AP = new(AP)
		t.AP.SetShape(rows, cols)
		return nil
	case reflect.Uint32:
		var backing []uint32
		for {
			record, err = cr.Read()
			if err == io.EOF {
				break
			}

			if err != nil {
				return
			}

			if row, err = convFromStrs(Uint32, record); err != nil {
				return
			}
			backing = append(backing, row.([]uint32)...)
			cols = len(record)
			rows++
		}
		t.fromSlice(backing)
		t.AP = new(AP)
		t.AP.SetShape(rows, cols)
		return nil
	case reflect.Uint64:
		var backing []uint64
		for {
			record, err = cr.Read()
			if err == io.EOF {
				break
			}

			if err != nil {
				return
			}

			if row, err = convFromStrs(Uint64, record); err != nil {
				return
			}
			backing = append(backing, row.([]uint64)...)
			cols = len(record)
			rows++
		}
		t.fromSlice(backing)
		t.AP = new(AP)
		t.AP.SetShape(rows, cols)
		return nil
	case reflect.Float32:
		var backing []float32
		for {
			record, err = cr.Read()
			if err == io.EOF {
				break
			}

			if err != nil {
				return
			}

			if row, err = convFromStrs(Float32, record); err != nil {
				return
			}
			backing = append(backing, row.([]float32)...)
			cols = len(record)
			rows++
		}
		t.fromSlice(backing)
		t.AP = new(AP)
		t.AP.SetShape(rows, cols)
		return nil
	case reflect.Float64:
		var backing []float64
		for {
			record, err = cr.Read()
			if err == io.EOF {
				break
			}

			if err != nil {
				return
			}

			if row, err = convFromStrs(Float64, record); err != nil {
				return
			}
			backing = append(backing, row.([]float64)...)
			cols = len(record)
			rows++
		}
		t.fromSlice(backing)
		t.AP = new(AP)
		t.AP.SetShape(rows, cols)
		return nil
	case reflect.String:
		var backing []string
		for {
			record, err = cr.Read()
			if err == io.EOF {
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
