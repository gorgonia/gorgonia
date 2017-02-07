package tensor

import "testing"

/*
GENERATED FILE. DO NOT EDIT
*/

/* MapTest */

func invFloat64(x float64) float64 {
	return -x
}

func Test_f64s_Map(t *testing.T) {
	a := f64s{0, 1, 2, 3, 4}
	b := f64s{0, 1, 2, 3, 4}
	if err := a.Map(invFloat64); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != invFloat64(b[i]) {
			t.Fatal("inverse function not applied")
		}
	}

	if err := a.Map(invFloat32); err == nil {
		t.Error("Expected an error!")
	}
}

func invFloat32(x float32) float32 {
	return -x
}

func Test_f32s_Map(t *testing.T) {
	a := f32s{0, 1, 2, 3, 4}
	b := f32s{0, 1, 2, 3, 4}
	if err := a.Map(invFloat32); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != invFloat32(b[i]) {
			t.Fatal("inverse function not applied")
		}
	}

	if err := a.Map(invFloat64); err == nil {
		t.Error("Expected an error!")
	}
}

func invInt(x int) int {
	return -x
}

func Test_ints_Map(t *testing.T) {
	a := ints{0, 1, 2, 3, 4}
	b := ints{0, 1, 2, 3, 4}
	if err := a.Map(invInt); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != invInt(b[i]) {
			t.Fatal("inverse function not applied")
		}
	}

	if err := a.Map(invFloat64); err == nil {
		t.Error("Expected an error!")
	}
}

func invInt64(x int64) int64 {
	return -x
}

func Test_i64s_Map(t *testing.T) {
	a := i64s{0, 1, 2, 3, 4}
	b := i64s{0, 1, 2, 3, 4}
	if err := a.Map(invInt64); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != invInt64(b[i]) {
			t.Fatal("inverse function not applied")
		}
	}

	if err := a.Map(invFloat64); err == nil {
		t.Error("Expected an error!")
	}
}

func invInt32(x int32) int32 {
	return -x
}

func Test_i32s_Map(t *testing.T) {
	a := i32s{0, 1, 2, 3, 4}
	b := i32s{0, 1, 2, 3, 4}
	if err := a.Map(invInt32); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != invInt32(b[i]) {
			t.Fatal("inverse function not applied")
		}
	}

	if err := a.Map(invFloat64); err == nil {
		t.Error("Expected an error!")
	}
}

func invByte(x byte) byte {
	return -x
}

func Test_u8s_Map(t *testing.T) {
	a := u8s{0, 1, 2, 3, 4}
	b := u8s{0, 1, 2, 3, 4}
	if err := a.Map(invByte); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != invByte(b[i]) {
			t.Fatal("inverse function not applied")
		}
	}

	if err := a.Map(invFloat64); err == nil {
		t.Error("Expected an error!")
	}
}

func invBool(x bool) bool {
	return !x
}

func Test_bs_Map(t *testing.T) {
	a := bs{true, false, true, false, true}
	b := bs{true, false, true, false, true}
	if err := a.Map(invBool); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != invBool(b[i]) {
			t.Fatal("inverse function not applied")
		}
	}

	if err := a.Map(invFloat64); err == nil {
		t.Error("Expected an error!")
	}
}
