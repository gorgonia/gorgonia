// +build !cuda

package gorgonia

import "testing"

func TestDeviceCPU(t *testing.T) {
	if CPU.IsGPU() {
		t.Fail()
	}
	a, err := CPU.Alloc(nil, 0)
	if a != nil || err != nil {
		t.Fail()
	}
	err = CPU.Free(nil, nil, 0)
	if err != nil {
		t.Fail()
	}
}
