package tensor

import (
	"bytes"
	"encoding/binary"
	"encoding/gob"
	"fmt"
	"io"
	"regexp"
	"strconv"
	"strings"

	"github.com/pkg/errors"
)

var dtmap = map[Dtype]string{
	Float64: "<f8",
	Float32: "<f4",
	Int:     "i8",
	Int64:   "i8",
	Int32:   "i4",
	Bool:    "bool",
}

// WriteNpy writes the *Dense as a numpy compatible serialized file.
//
// The format is very well documented here:
// http://docs.scipy.org/doc/numpy/neps/npy-format.html
//
// Gorgonia specifically uses Version 2.0. The floats are written in little endian order,
// because let's face it - 90% of the world's computers are running on x86+ processors
// This method does not close the writer. Closing (optional) is deferred to the caller
func (t *Dense) WriteNpy(w io.Writer) error {
	npType, ok := dtmap[t.t]
	if !ok {
		return errors.Errorf(methodNYI, "WriteNpy", t.t)
	}
	// prep header
	// <f8 indicates that this is a little endian float64.
	header := "{'descr': '%s', 'fortran_order': False, 'shape': %v}"
	header = fmt.Sprintf(header, npType, t.Shape())
	padding := 16 - ((10 + len(header)) % 16)
	if padding > 0 {
		header = header + strings.Repeat(" ", padding)
	}

	w.Write([]byte("\x93NUMPY"))                              // stupid magic
	binary.Write(w, binary.LittleEndian, byte(1))             // major version
	binary.Write(w, binary.LittleEndian, byte(0))             // minor version
	binary.Write(w, binary.LittleEndian, uint16(len(header))) // 4 bytes to denote header length
	w.Write([]byte(header))

	switch at := t.data.(type) {
	case f64s:
		for _, v := range at {
			binary.Write(w, binary.LittleEndian, v)
		}
	case f32s:
		for _, v := range at {
			binary.Write(w, binary.LittleEndian, v)
		}
	case ints:
		for _, v := range at {
			binary.Write(w, binary.LittleEndian, int64(v))
		}
	case i64s:
		for _, v := range at {
			binary.Write(w, binary.LittleEndian, v)
		}
	case i32s:
		for _, v := range at {
			binary.Write(w, binary.LittleEndian, v)
		}
	case u8s:
		for _, v := range at {
			binary.Write(w, binary.LittleEndian, v)
		}
	case bs:
		for _, v := range at {
			binary.Write(w, binary.LittleEndian, v)
		}
	}
	return nil
}

func (t *Dense) GobEncode() (p []byte, err error) {
	var buf bytes.Buffer
	encoder := gob.NewEncoder(&buf)

	if err = encoder.Encode(t.Shape()); err != nil {
		return
	}

	if err = encoder.Encode(t.Strides()); err != nil {
		return
	}

	if err = encoder.Encode(t.data); err != nil {
		return
	}

	p = buf.Bytes()
	return
}

/* READ SHIT */

func (t *Dense) ReadNpy(r io.Reader) (err error) {
	var magic [6]byte
	if _, err = r.Read(magic[:]); err != nil {
		return
	}
	if string(magic[:]) != "\x93NUMPY" {
		err = errors.Errorf("IO Error: Not a numpy file. Got %q as the magic number instead", string(magic[:]))
		return
	}

	var version byte
	if err = binary.Read(r, binary.LittleEndian, &version); err != nil {
		return
	}
	if version != 1 {
		err = errors.Errorf("IO Error: Only version 1 of numpy's serialization is currently supported")
		return
	}

	var minor byte
	if err = binary.Read(r, binary.LittleEndian, &minor); err != nil {
		return
	}
	if minor != 0 {
		err = errors.Errorf("IO Error: Only version 1.0 of numpy's serialization is currently supported")
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
		err = errors.Errorf("IO Error No dtype information found")
		return
	}

	if string(match[1]) != "<f8" {
		err = errors.Errorf(dtypeMismatch, "Float64", string(match[1])) // the reason is because the error message itself will actually be used to handle errors
		return
	}

	rowOrder := regexp.MustCompile(`'fortran_order':\s*(False|True)`)
	match = rowOrder.FindSubmatch(header)
	if match == nil {
		err = errors.Errorf("IO Error: No row order information found")
		return
	}
	if string(match[1]) != "False" {
		err = errors.Errorf(methodNYI, "ReadNPY", "Fortran-Ordered files")
		return
	}

	shpRe := regexp.MustCompile(`'shape':\s*\(([^\(]*)\)`)
	match = shpRe.FindSubmatch(header)
	if match == nil {
		err = errors.Errorf("IO Error: No shape information found")
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
	var arr Array
	switch t.t {
	case Float64:
		data := make(f64s, size)
		for i := 0; i < size; i++ {
			if err = binary.Read(r, binary.LittleEndian, &data[i]); err != nil {
				return
			}
		}
		arr = data
	case Float32:
		data := make(f32s, size)
		for i := 0; i < size; i++ {
			if err = binary.Read(r, binary.LittleEndian, &data[i]); err != nil {
				return
			}
		}
		arr = data
	case Int:
		data := make(ints, size)
		for i := 0; i < size; i++ {
			if err = binary.Read(r, binary.LittleEndian, &data[i]); err != nil {
				return
			}
		}
		arr = data
	case Int64:
		data := make(i64s, size)
		for i := 0; i < size; i++ {
			if err = binary.Read(r, binary.LittleEndian, &data[i]); err != nil {
				return
			}
		}
		arr = data
	case Int32:
		data := make(i32s, size)
		for i := 0; i < size; i++ {
			if err = binary.Read(r, binary.LittleEndian, &data[i]); err != nil {
				return
			}
		}
		arr = data
	case Byte:
		data := make(u8s, size)
		for i := 0; i < size; i++ {
			if err = binary.Read(r, binary.LittleEndian, &data[i]); err != nil {
				return
			}
		}
		arr = data
	case Bool:
		data := make(bs, size)
		for i := 0; i < size; i++ {
			if err = binary.Read(r, binary.LittleEndian, &data[i]); err != nil {
				return
			}
		}
		arr = data
	}

	if t.AP == nil {
		t.AP = new(AP)
	}

	t.setShape(shape...)
	t.data = arr
	t.fix()
	return t.sanity()
}

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

	var data Array
	if err = decoder.Decode(&data); err != nil {
		return
	}

	t.AP = NewAP(shape, strides)
	t.data = data
	t.fix()
	return t.sanity()
}
