package tensor

import "testing"

func TestF64s_Memset(t *testing.T) {
	size := 5
	a := makeArray(Float64, size).(f64s)
	sets := []interface{}{
		float64(1), float32(2), int(3), int64(4), int32(5), byte(6),
	}

	correct := make(f64s, size)
	for i, set := range sets {
		for j := range correct {
			correct[j] = float64(i + 1)
		}

		if err := a.Memset(set); err != nil {
			t.Error(err)
			continue
		}
	}

	if err := a.Memset("Hello"); err == nil {
		t.Error("Expected an error")
	}

}

func TestF32s_Memset(t *testing.T) {
	size := 5
	a := makeArray(Float32, size).(f32s)
	sets := []interface{}{
		float64(1), float32(2), int(3), int64(4), int32(5), byte(6),
	}

	correct := make(f32s, size)
	for i, set := range sets {
		for j := range correct {
			correct[j] = float32(i + 1)
		}

		if err := a.Memset(set); err != nil {
			t.Error(err)
			continue
		}
	}

	if err := a.Memset("Hello"); err == nil {
		t.Error("Expected an error")
	}
}

func TestInts_Memset(t *testing.T) {
	size := 5
	a := makeArray(Int, size).(ints)
	sets := []interface{}{
		float64(1), float32(2), int(3), int64(4), int32(5), byte(6),
	}

	correct := make(ints, size)
	for i, set := range sets {
		for j := range correct {
			correct[j] = int(i + 1)
		}

		if err := a.Memset(set); err != nil {
			t.Error(err)
			continue
		}
	}

	if err := a.Memset("Hello"); err == nil {
		t.Error("Expected an error")
	}
}

func TestI64s_Memset(t *testing.T) {
	size := 5
	a := makeArray(Int64, size).(i64s)
	sets := []interface{}{
		float64(1), float32(2), int(3), int64(4), int32(5), byte(6),
	}

	correct := make(i64s, size)
	for i, set := range sets {
		for j := range correct {
			correct[j] = int64(i + 1)
		}

		if err := a.Memset(set); err != nil {
			t.Error(err)
			continue
		}
	}

	if err := a.Memset("Hello"); err == nil {
		t.Error("Expected an error")
	}
}

func TestI32s_Memset(t *testing.T) {
	size := 5
	a := makeArray(Int32, size).(i32s)
	sets := []interface{}{
		float64(1), float32(2), int(3), int64(4), int32(5), byte(6),
	}

	correct := make(i32s, size)
	for i, set := range sets {
		for j := range correct {
			correct[j] = int32(i + 1)
		}

		if err := a.Memset(set); err != nil {
			t.Error(err)
			continue
		}
	}

	if err := a.Memset("Hello"); err == nil {
		t.Error("Expected an error")
	}
}

func TestU8s_Memset(t *testing.T) {
	size := 5
	a := makeArray(Byte, size).(u8s)
	sets := []interface{}{
		float64(1), float32(2), int(3), int64(4), int32(5), byte(6),
	}

	correct := make(u8s, size)
	for i, set := range sets {
		for j := range correct {
			correct[j] = byte(i + 1)
		}

		if err := a.Memset(set); err != nil {
			t.Error(err)
			continue
		}
	}

	sets = []interface{}{
		float64(-1), float32(-2), int(-3), int64(-4), int32(-5),
	}
	for _, set := range sets {
		if err := a.Memset(set); err == nil {
			t.Error("Expected an error when setting negative numbers to slice of bytes")
		}
	}

	if err := a.Memset("Hello"); err == nil {
		t.Error("Expected an error")
	}
}

func TestBs_Memset(t *testing.T) {
	size := 5
	a := makeArray(Bool, size).(bs)
	sets := []interface{}{
		true, false,
	}

	correct := make(bs, size)
	for _, set := range sets {
		for j := range correct {
			correct[j] = set.(bool)
		}

		if err := a.Memset(set); err != nil {
			t.Error(err)
		}
	}

	sets = []interface{}{
		float64(1), float32(2), int(3), int64(4), int32(5), byte(6),
	}
	for _, set := range sets {
		if err := a.Memset(set); err == nil {
			t.Error("Expected an error when setting negative numbers to slice of bytes")
		}
	}
}
