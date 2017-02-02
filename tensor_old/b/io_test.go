package tensorb

import (
	"bytes"
	"encoding/gob"
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

	T := NewTensor(WithShape(2, 2), WithBacking([]bool{true, true, false, false}))
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
	if os.Getenv("TRAVISTEST") == "true" {
		t.Skip("skipping test; This is being run on TravisCI")
	}

	assert := assert.New(t)
	T := NewTensor(WithShape(2, 2), WithBacking([]bool{true, false, true, false}))
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

	expected := `[[ True False]
 [ True False]]
`

	if buf.String() != expected {
		t.Errorf("Did not successfully read numpy file, \n%q\n%q", buf.String(), expected)
	}

	// // cleanup
	// err = os.Remove("test.npy")
	// if err != nil {
	// 	t.Error(err)
	// }

	// ok now to test if it can read
	T2 := new(Tensor)
	buf = new(bytes.Buffer)
	T.WriteNpy(buf)
	err = T2.ReadNpy(buf)
	if err != nil {
		t.Fatal(err)
	}
	assert.Equal(T.Shape(), T2.Shape())
	assert.Equal(T.Strides(), T2.Strides())
	assert.Equal(T.data, T2.data)
}
