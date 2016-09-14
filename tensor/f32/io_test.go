package tensorf32

import (
	"bytes"
	"encoding/gob"
	"encoding/json"
	"os"
	"os/exec"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestGobShite(t *testing.T) {
	assert := assert.New(t)
	buf := new(bytes.Buffer)
	encoder := gob.NewEncoder(buf)
	decoder := gob.NewDecoder(buf)

	T := NewTensor(WithShape(2, 2), WithBacking([]float32{1, 5, 10, -1}))
	err := encoder.Encode(T)
	if err != nil {
		t.Error(err)
	}

	T2 := new(Tensor)
	if err = decoder.Decode(T2); err != nil {
		t.Error(err)
	}

	assert.Equal(T.Shape(), T2.Shape())
	assert.Equal(T.Strides(), T2.Strides())
	assert.Equal(T.data, T2.data)
}

func TestSaveLoadNumpy(t *testing.T) {
	assert := assert.New(t)
	T := NewTensor(WithShape(2, 2), WithBacking([]float32{1, 5, 10, -1}))
	f, _ := os.OpenFile("test.npy", os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0644)
	T.WriteNpy(f)
	f.Close()

	script := `import numpy as np
x = np.load('test.npy')
print(x)`

	cmd := exec.Command("python")
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

	expected := `[[  1.   5.]
 [ 10.  -1.]]
`

	if buf.String() != expected {
		t.Errorf("Did not successfully read numpy file, \n%q\n%q", buf.String(), expected)
	}

	// cleanup
	err = os.Remove("test.npy")
	if err != nil {
		t.Error(err)
	}

	// ok now to test if it can read
	T2 := new(Tensor)
	buf = new(bytes.Buffer)
	T.WriteNpy(buf)
	err = T2.ReadNpy(buf)
	if err != nil {
		t.Error(err)
	}
	assert.Equal(T.Shape(), T2.Shape())
	assert.Equal(T.Strides(), T2.Strides())
	assert.Equal(T.data, T2.data)
}

func TestWriteJSON(t *testing.T) {
	assert := assert.New(t)
	T := NewTensor(WithShape(2, 2), WithBacking([]float32{1, 5, 10, -1}))

	p, err := json.Marshal(T)
	if err != nil {
		t.Error(err)
	}

	expected := `{"Shape":[2,2],"data":[1,5,10,-1]}`
	assert.Equal(expected, string(p))
}
