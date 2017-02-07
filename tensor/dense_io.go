package tensor

import (
	"bytes"
	"encoding/binary"
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
	for i := 0; i < t.len(); i++ {
		bw.w(t.Get(i))
	}

	if bw.error != nil {
		return bw
	}
	return nil
}

func (t *Dense) GobEncode() (p []byte, err error) {
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
	case reflect.Int:
		if err = encoder.Encode(t.ints()); err != nil {
			return
		}
	case reflect.Int8:
		if err = encoder.Encode(t.int8s()); err != nil {
			return
		}
	case reflect.Int16:
		if err = encoder.Encode(t.int16s()); err != nil {
			return
		}
	case reflect.Int32:
		if err = encoder.Encode(t.int32s()); err != nil {
			return
		}
	case reflect.Int64:
		if err = encoder.Encode(t.int64s()); err != nil {
			return
		}
	case reflect.Uint:
		if err = encoder.Encode(t.uints()); err != nil {
			return
		}
	case reflect.Uint8:
		if err = encoder.Encode(t.uint8s()); err != nil {
			return
		}
	case reflect.Uint16:
		if err = encoder.Encode(t.uint16s()); err != nil {
			return
		}
	case reflect.Uint32:
		if err = encoder.Encode(t.uint32s()); err != nil {
			return
		}
	case reflect.Uint64:
		if err = encoder.Encode(t.uint64s()); err != nil {
			return
		}
	case reflect.Float32:
		if err = encoder.Encode(t.float32s()); err != nil {
			return
		}
	case reflect.Float64:
		if err = encoder.Encode(t.float64s()); err != nil {
			return
		}
	case reflect.Complex64:
		if err = encoder.Encode(t.complex64s()); err != nil {
			return
		}
	case reflect.Complex128:
		if err = encoder.Encode(t.complex128s()); err != nil {
			return
		}
	case reflect.String:
		if err = encoder.Encode(t.strings()); err != nil {
			return
		}
	}

	return buf.Bytes(), err
}
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
		err = errors.Errorf("No Row Order information found in the numpy file")
		return
	}
	if string(match[1]) != "False" {
		err = errors.Errorf("Cannot yet read from Fortran Ordered Numpy files")
		return
	}

	shpRe := regexp.MustCompile(`'shape':\s*\(([^\(]*)\)`)
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
func (t *Dense) GobDecode(p []byte) (err error) {
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
	case reflect.Int:
		var data []int
		if err = decoder.Decode(&data); err != nil {
			return
		}
		t.fromSlice(data)
	case reflect.Int8:
		var data []int8
		if err = decoder.Decode(&data); err != nil {
			return
		}
		t.fromSlice(data)
	case reflect.Int16:
		var data []int16
		if err = decoder.Decode(&data); err != nil {
			return
		}
		t.fromSlice(data)
	case reflect.Int32:
		var data []int32
		if err = decoder.Decode(&data); err != nil {
			return
		}
		t.fromSlice(data)
	case reflect.Int64:
		var data []int64
		if err = decoder.Decode(&data); err != nil {
			return
		}
		t.fromSlice(data)
	case reflect.Uint:
		var data []uint
		if err = decoder.Decode(&data); err != nil {
			return
		}
		t.fromSlice(data)
	case reflect.Uint8:
		var data []uint8
		if err = decoder.Decode(&data); err != nil {
			return
		}
		t.fromSlice(data)
	case reflect.Uint16:
		var data []uint16
		if err = decoder.Decode(&data); err != nil {
			return
		}
		t.fromSlice(data)
	case reflect.Uint32:
		var data []uint32
		if err = decoder.Decode(&data); err != nil {
			return
		}
		t.fromSlice(data)
	case reflect.Uint64:
		var data []uint64
		if err = decoder.Decode(&data); err != nil {
			return
		}
		t.fromSlice(data)
	case reflect.Float32:
		var data []float32
		if err = decoder.Decode(&data); err != nil {
			return
		}
		t.fromSlice(data)
	case reflect.Float64:
		var data []float64
		if err = decoder.Decode(&data); err != nil {
			return
		}
		t.fromSlice(data)
	case reflect.Complex64:
		var data []complex64
		if err = decoder.Decode(&data); err != nil {
			return
		}
		t.fromSlice(data)
	case reflect.Complex128:
		var data []complex128
		if err = decoder.Decode(&data); err != nil {
			return
		}
		t.fromSlice(data)
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
