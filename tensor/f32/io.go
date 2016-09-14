package tensorf32

import (
	"bytes"
	"encoding/binary"
	"encoding/gob"
	"fmt"
	"io"
	"regexp"
	"strconv"
	"strings"

	"github.com/chewxy/gorgonia/tensor/types"
)

// WriteNpy writes the *Tensor as a numpy compatible serialized file.
//
// The format is very well documented here:
// http://docs.scipy.org/doc/numpy/neps/npy-format.html
//
// Gorgonia specifically uses Version 2.0. The floats are written in little endian order,
// because let's face it - 90% of the world's computers are running on x86+ processors
// This method does not close the writer. Closing (optional) is deferred to the caller
func (t *Tensor) WriteNpy(w io.Writer) {
	// prep header
	// <f8 indicates that this is a little endian float32.
	header := "{'descr': '<f4', 'fortran_order': False, 'shape': %v}"
	header = fmt.Sprintf(header, t.Shape())
	padding := 16 - ((10 + len(header)) % 16)
	if padding > 0 {
		header = header + strings.Repeat(" ", padding)
	}

	w.Write([]byte("\x93NUMPY"))                              // stupid magic
	binary.Write(w, binary.LittleEndian, byte(1))             // major version
	binary.Write(w, binary.LittleEndian, byte(0))             // minor version
	binary.Write(w, binary.LittleEndian, uint16(len(header))) // 4 bytes to denote header length
	w.Write([]byte(header))

	for _, v := range t.data {
		binary.Write(w, binary.LittleEndian, v)
	}
}

func (t *Tensor) GobEncode() (p []byte, err error) {
	var buf bytes.Buffer
	encoder := gob.NewEncoder(&buf)

	if err = encoder.Encode(t.Shape()); err != nil {
		return
	}

	if err = encoder.Encode(t.data); err != nil {
		return
	}

	p = buf.Bytes()
	return
}

// MarshalJSON implements the JSONMarshaller interface
func (t *Tensor) MarshalJSON() (p []byte, err error) {
	var buf bytes.Buffer
	buf.WriteString("{\"Shape\": [")
	for i, s := range t.Shape() {
		fmt.Fprintf(&buf, "%d", s)
		if i < len(t.Shape())-1 {
			buf.WriteString(",")
		}
	}
	buf.WriteString("], \"data\": [")
	for i, v := range t.data {
		fmt.Fprintf(&buf, "%g", v)
		if i < len(t.data)-1 {
			buf.WriteString(", ")
		}
	}
	buf.WriteString("]}")
	return buf.Bytes(), nil
}

/* READ SHIT */

func (t *Tensor) ReadNpy(r io.Reader) (err error) {
	var magic [6]byte
	if _, err = r.Read(magic[:]); err != nil {
		return
	}
	if string(magic[:]) != "\x93NUMPY" {
		err = types.NewError(types.IOError, "Not a numpy file. Got %q as the magic number instead", string(magic[:]))
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

	desc := regexp.MustCompile(`'descr':\s*'([^']*)'`)
	match := desc.FindSubmatch(header)
	if match == nil {
		err = types.NewError(types.IOError, "No dtype information found")
		return
	}

	if string(match[1]) != "<f4" {
		err = types.NewError(types.DtypeMismatch, string(match[1])) // the reason is because the error message itself will actually be used to handle errors
		return
	}

	rowOrder := regexp.MustCompile(`'fortran_order':\s*(False|True)`)
	match = rowOrder.FindSubmatch(header)
	if match == nil {
		err = types.NewError(types.IOError, "No row order information found")
		return
	}
	if string(match[1]) != "False" {
		err = types.NewError(types.NotYetImplemented, "Cannot yet read from fortranorder files")
		return
	}

	shpRe := regexp.MustCompile(`'shape':\s*\(([^\(]*)\)`)
	match = shpRe.FindSubmatch(header)
	if match == nil {
		err = types.NewError(types.IOError, "No shape information found")
		return
	}
	sizesStr := strings.Split(string(match[1]), ",")

	var shape types.Shape
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
	data := make([]float32, size)

	for i := 0; i < size; i++ {
		if err = binary.Read(r, binary.LittleEndian, &data[i]); err != nil {
			return
		}
	}

	if t.AP == nil {
		t.AP = new(types.AP)
	}

	t.setShape(shape...)
	t.data = data
	t.fix()
	return t.sanity()
}

func (t *Tensor) GobDecode(p []byte) (err error) {
	buf := bytes.NewBuffer(p)
	decoder := gob.NewDecoder(buf)

	var shape types.Shape
	if err = decoder.Decode(&shape); err != nil {
		return
	}

	var data []float32
	if err = decoder.Decode(&data); err != nil {
		return
	}

	if t.AP == nil {
		t.AP = new(types.AP)
	}

	t.data = data
	t.setShape(shape...)
	t.fix()
	return t.sanity()
}
