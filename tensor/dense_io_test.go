package tensor

import (
	"bytes"
	"encoding/gob"
	"os"
	"os/exec"
	"testing"

	"github.com/alecthomas/assert"
)

/*
GENERATED FILE. DO NOT EDIT
*/

func TestSaveLoadNumpy(t *testing.T) {
	if os.Getenv("TRAVISTEST") == "true" {
		t.Skip("skipping test; This is being run on TravisCI")
	}

	assert := assert.New(t)
	T := New(WithShape(2, 2), WithBacking([]float64{1, 5, 10, -1}))
	f, _ := os.OpenFile("test.npy", os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0644)
	T.WriteNpy(f)
	f.Close()

	script := "import numpy as np\nx = np.load('test.npy')\nprint(x)"

	cmd := exec.Command("python2")
	stdin, err := cmd.StdinPipe()
	if err != nil {
		t.Error(err)
	}

	go func() {
		defer stdin.Close()
		stdin.Write([]byte(script))
	}()

	buf := new(bytes.Buffer)
	cmd.Stdout = buf

	if err = cmd.Start(); err != nil {
		t.Error(err)
	}

	if err := cmd.Wait(); err != nil {
		t.Error(err)
	}

	expected := "[[  1.   5.]\n [ 10.  -1.]]\n"

	if buf.String() != expected {
		t.Errorf("Did not successfully read numpy file, \n%q\n%q", buf.String(), expected)
	}

	// cleanup
	err = os.Remove("test.npy")
	if err != nil {
		t.Error(err)
	}

	// ok now to test if it can read
	T2 := new(Dense)
	buf = new(bytes.Buffer)
	T.WriteNpy(buf)
	if err = T2.ReadNpy(buf); err != nil {
		t.Fatal(err)
	}
	assert.Equal(T.Shape(), T2.Shape())
	assert.Equal(T.Strides(), T2.Strides())
	assert.Equal(T.Data(), T2.Data())

	// try with masked array. masked elements should be filled with default value
	T.ResetMask(false)
	T.mask[0] = true
	T3 := new(Dense)
	buf = new(bytes.Buffer)
	T.WriteNpy(buf)
	if err = T3.ReadNpy(buf); err != nil {
		t.Fatal(err)
	}
	assert.Equal(T.Shape(), T3.Shape())
	assert.Equal(T.Strides(), T3.Strides())
	data := T.Float64s()
	data[0] = T.FillValue().(float64)
	assert.Equal(data, T3.Data())
}

func TestSaveLoadCSV(t *testing.T) {
	assert := assert.New(t)
	for _, gtd := range serializationTestData {
		if _, ok := gtd.([]complex64); ok {
			continue
		}
		if _, ok := gtd.([]complex128); ok {
			continue
		}

		buf := new(bytes.Buffer)

		T := New(WithShape(2, 2), WithBacking(gtd))
		if err := T.WriteCSV(buf); err != nil {
			t.Error(err)
		}

		T2 := new(Dense)
		if err := T2.ReadCSV(buf, As(T.t)); err != nil {
			t.Error(err)
		}

		assert.Equal(T.Shape(), T2.Shape(), "Test: %v", gtd)
		assert.Equal(T.Data(), T2.Data())

	}

	T := New(WithShape(2, 2), WithBacking([]float64{1, 5, 10, -1}))
	f, _ := os.OpenFile("test.csv", os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0644)
	T.WriteCSV(f)
	f.Close()

	// cleanup
	err := os.Remove("test.csv")
	if err != nil {
		t.Error(err)
	}

	// try with masked array. masked elements should be filled with default value
	T.ResetMask(false)
	T.mask[0] = true
	f, _ = os.OpenFile("test.csv", os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0644)
	T.WriteCSV(f)
	f.Close()

	// cleanup again
	err = os.Remove("test.csv")
	if err != nil {
		t.Error(err)
	}

}

var serializationTestData = []interface{}{
	[]int{1, 5, 10, -1},
	[]int8{1, 5, 10, -1},
	[]int16{1, 5, 10, -1},
	[]int32{1, 5, 10, -1},
	[]int64{1, 5, 10, -1},
	[]uint{1, 5, 10, 255},
	[]uint8{1, 5, 10, 255},
	[]uint16{1, 5, 10, 255},
	[]uint32{1, 5, 10, 255},
	[]uint64{1, 5, 10, 255},
	[]float32{1, 5, 10, -1},
	[]float64{1, 5, 10, -1},
	[]complex64{1, 5, 10, -1},
	[]complex128{1, 5, 10, -1},
	[]string{"hello", "world", "hello", "世界"},
}

func TestDense_GobEncodeDecode(t *testing.T) {
	assert := assert.New(t)
	var err error
	for _, gtd := range serializationTestData {
		buf := new(bytes.Buffer)
		encoder := gob.NewEncoder(buf)
		decoder := gob.NewDecoder(buf)

		T := New(WithShape(2, 2), WithBacking(gtd))
		if err = encoder.Encode(T); err != nil {
			t.Errorf("Error while encoding %v: %v", gtd, err)
			continue
		}

		T2 := new(Dense)
		if err = decoder.Decode(T2); err != nil {
			t.Errorf("Error while decoding %v: %v", gtd, err)
			continue
		}

		assert.Equal(T.Shape(), T2.Shape())
		assert.Equal(T.Strides(), T2.Strides())
		assert.Equal(T.Data(), T2.Data())

		// try with masked array. masked elements should be filled with default value
		buf = new(bytes.Buffer)
		encoder = gob.NewEncoder(buf)
		decoder = gob.NewDecoder(buf)

		T.ResetMask(false)
		T.mask[0] = true
		assert.True(T.IsMasked())
		if err = encoder.Encode(T); err != nil {
			t.Errorf("Error while encoding %v: %v", gtd, err)
			continue
		}

		T3 := new(Dense)
		if err = decoder.Decode(T3); err != nil {
			t.Errorf("Error while decoding %v: %v", gtd, err)
			continue
		}

		assert.Equal(T.Shape(), T3.Shape())
		assert.Equal(T.Strides(), T3.Strides())
		assert.Equal(T.Data(), T3.Data())
		assert.Equal(T.mask, T3.mask)

	}

}
