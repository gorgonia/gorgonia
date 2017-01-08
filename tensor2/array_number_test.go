package tensor

import "testing"

/* Add */

func Test_f64s_Add(t *testing.T) {
	var a, b f64s
	var c f64sDummy

	a = f64s{0, 1, 2, 3, 4}
	b = f64s{1, 2, 2, 1, 100}
	c = f64sDummy{1, 2, 2, 1, 100}

	correct := make(f64s, len(a))
	for i, v := range a {
		correct[i] = v + b[i]
	}

	// same type
	if err := a.Add(b); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Add is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// compatible type
	a = f64s{0, 1, 2, 3, 4}
	if err := a.Add(c); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Add is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.Add(b[:3]); err == nil {
		t.Error("Expected an error when performing Add on differing lengths")
	}

	// idiotsville 2
	if err := a.Add(f32s{}); err == nil {
		t.Errorf("Expected an error when performing Add on a non-compatible type")
	}

}

func Test_f32s_Add(t *testing.T) {
	var a, b f32s
	var c f32sDummy

	a = f32s{0, 1, 2, 3, 4}
	b = f32s{1, 2, 2, 1, 100}
	c = f32sDummy{1, 2, 2, 1, 100}

	correct := make(f32s, len(a))
	for i, v := range a {
		correct[i] = v + b[i]
	}

	// same type
	if err := a.Add(b); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Add is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// compatible type
	a = f32s{0, 1, 2, 3, 4}
	if err := a.Add(c); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Add is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.Add(b[:3]); err == nil {
		t.Error("Expected an error when performing Add on differing lengths")
	}

	// idiotsville 2
	if err := a.Add(f64s{}); err == nil {
		t.Errorf("Expected an error when performing Add on a non-compatible type")
	}

}

func Test_ints_Add(t *testing.T) {
	var a, b ints
	var c intsDummy

	a = ints{0, 1, 2, 3, 4}
	b = ints{1, 2, 2, 1, 100}
	c = intsDummy{1, 2, 2, 1, 100}

	correct := make(ints, len(a))
	for i, v := range a {
		correct[i] = v + b[i]
	}

	// same type
	if err := a.Add(b); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Add is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// compatible type
	a = ints{0, 1, 2, 3, 4}
	if err := a.Add(c); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Add is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.Add(b[:3]); err == nil {
		t.Error("Expected an error when performing Add on differing lengths")
	}

	// idiotsville 2
	if err := a.Add(f64s{}); err == nil {
		t.Errorf("Expected an error when performing Add on a non-compatible type")
	}

}

func Test_i64s_Add(t *testing.T) {
	var a, b i64s
	var c i64sDummy

	a = i64s{0, 1, 2, 3, 4}
	b = i64s{1, 2, 2, 1, 100}
	c = i64sDummy{1, 2, 2, 1, 100}

	correct := make(i64s, len(a))
	for i, v := range a {
		correct[i] = v + b[i]
	}

	// same type
	if err := a.Add(b); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Add is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// compatible type
	a = i64s{0, 1, 2, 3, 4}
	if err := a.Add(c); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Add is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.Add(b[:3]); err == nil {
		t.Error("Expected an error when performing Add on differing lengths")
	}

	// idiotsville 2
	if err := a.Add(f64s{}); err == nil {
		t.Errorf("Expected an error when performing Add on a non-compatible type")
	}

}

func Test_i32s_Add(t *testing.T) {
	var a, b i32s
	var c i32sDummy

	a = i32s{0, 1, 2, 3, 4}
	b = i32s{1, 2, 2, 1, 100}
	c = i32sDummy{1, 2, 2, 1, 100}

	correct := make(i32s, len(a))
	for i, v := range a {
		correct[i] = v + b[i]
	}

	// same type
	if err := a.Add(b); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Add is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// compatible type
	a = i32s{0, 1, 2, 3, 4}
	if err := a.Add(c); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Add is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.Add(b[:3]); err == nil {
		t.Error("Expected an error when performing Add on differing lengths")
	}

	// idiotsville 2
	if err := a.Add(f64s{}); err == nil {
		t.Errorf("Expected an error when performing Add on a non-compatible type")
	}

}

func Test_u8s_Add(t *testing.T) {
	var a, b u8s
	var c u8sDummy

	a = u8s{0, 1, 2, 3, 4}
	b = u8s{1, 2, 2, 1, 100}
	c = u8sDummy{1, 2, 2, 1, 100}

	correct := make(u8s, len(a))
	for i, v := range a {
		correct[i] = v + b[i]
	}

	// same type
	if err := a.Add(b); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Add is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// compatible type
	a = u8s{0, 1, 2, 3, 4}
	if err := a.Add(c); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Add is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.Add(b[:3]); err == nil {
		t.Error("Expected an error when performing Add on differing lengths")
	}

	// idiotsville 2
	if err := a.Add(f64s{}); err == nil {
		t.Errorf("Expected an error when performing Add on a non-compatible type")
	}

}

/* Sub */

func Test_f64s_Sub(t *testing.T) {
	var a, b f64s
	var c f64sDummy

	a = f64s{0, 1, 2, 3, 4}
	b = f64s{1, 2, 2, 1, 100}
	c = f64sDummy{1, 2, 2, 1, 100}

	correct := make(f64s, len(a))
	for i, v := range a {
		correct[i] = v - b[i]
	}

	// same type
	if err := a.Sub(b); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Sub is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// compatible type
	a = f64s{0, 1, 2, 3, 4}
	if err := a.Sub(c); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Sub is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.Sub(b[:3]); err == nil {
		t.Error("Expected an error when performing Sub on differing lengths")
	}

	// idiotsville 2
	if err := a.Sub(f32s{}); err == nil {
		t.Errorf("Expected an error when performing Sub on a non-compatible type")
	}

}

func Test_f32s_Sub(t *testing.T) {
	var a, b f32s
	var c f32sDummy

	a = f32s{0, 1, 2, 3, 4}
	b = f32s{1, 2, 2, 1, 100}
	c = f32sDummy{1, 2, 2, 1, 100}

	correct := make(f32s, len(a))
	for i, v := range a {
		correct[i] = v - b[i]
	}

	// same type
	if err := a.Sub(b); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Sub is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// compatible type
	a = f32s{0, 1, 2, 3, 4}
	if err := a.Sub(c); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Sub is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.Sub(b[:3]); err == nil {
		t.Error("Expected an error when performing Sub on differing lengths")
	}

	// idiotsville 2
	if err := a.Sub(f64s{}); err == nil {
		t.Errorf("Expected an error when performing Sub on a non-compatible type")
	}

}

func Test_ints_Sub(t *testing.T) {
	var a, b ints
	var c intsDummy

	a = ints{0, 1, 2, 3, 4}
	b = ints{1, 2, 2, 1, 100}
	c = intsDummy{1, 2, 2, 1, 100}

	correct := make(ints, len(a))
	for i, v := range a {
		correct[i] = v - b[i]
	}

	// same type
	if err := a.Sub(b); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Sub is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// compatible type
	a = ints{0, 1, 2, 3, 4}
	if err := a.Sub(c); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Sub is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.Sub(b[:3]); err == nil {
		t.Error("Expected an error when performing Sub on differing lengths")
	}

	// idiotsville 2
	if err := a.Sub(f64s{}); err == nil {
		t.Errorf("Expected an error when performing Sub on a non-compatible type")
	}

}

func Test_i64s_Sub(t *testing.T) {
	var a, b i64s
	var c i64sDummy

	a = i64s{0, 1, 2, 3, 4}
	b = i64s{1, 2, 2, 1, 100}
	c = i64sDummy{1, 2, 2, 1, 100}

	correct := make(i64s, len(a))
	for i, v := range a {
		correct[i] = v - b[i]
	}

	// same type
	if err := a.Sub(b); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Sub is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// compatible type
	a = i64s{0, 1, 2, 3, 4}
	if err := a.Sub(c); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Sub is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.Sub(b[:3]); err == nil {
		t.Error("Expected an error when performing Sub on differing lengths")
	}

	// idiotsville 2
	if err := a.Sub(f64s{}); err == nil {
		t.Errorf("Expected an error when performing Sub on a non-compatible type")
	}

}

func Test_i32s_Sub(t *testing.T) {
	var a, b i32s
	var c i32sDummy

	a = i32s{0, 1, 2, 3, 4}
	b = i32s{1, 2, 2, 1, 100}
	c = i32sDummy{1, 2, 2, 1, 100}

	correct := make(i32s, len(a))
	for i, v := range a {
		correct[i] = v - b[i]
	}

	// same type
	if err := a.Sub(b); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Sub is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// compatible type
	a = i32s{0, 1, 2, 3, 4}
	if err := a.Sub(c); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Sub is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.Sub(b[:3]); err == nil {
		t.Error("Expected an error when performing Sub on differing lengths")
	}

	// idiotsville 2
	if err := a.Sub(f64s{}); err == nil {
		t.Errorf("Expected an error when performing Sub on a non-compatible type")
	}

}

func Test_u8s_Sub(t *testing.T) {
	var a, b u8s
	var c u8sDummy

	a = u8s{0, 1, 2, 3, 4}
	b = u8s{1, 2, 2, 1, 100}
	c = u8sDummy{1, 2, 2, 1, 100}

	correct := make(u8s, len(a))
	for i, v := range a {
		correct[i] = v - b[i]
	}

	// same type
	if err := a.Sub(b); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Sub is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// compatible type
	a = u8s{0, 1, 2, 3, 4}
	if err := a.Sub(c); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Sub is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.Sub(b[:3]); err == nil {
		t.Error("Expected an error when performing Sub on differing lengths")
	}

	// idiotsville 2
	if err := a.Sub(f64s{}); err == nil {
		t.Errorf("Expected an error when performing Sub on a non-compatible type")
	}

}

/* Mul */

func Test_f64s_Mul(t *testing.T) {
	var a, b f64s
	var c f64sDummy

	a = f64s{0, 1, 2, 3, 4}
	b = f64s{1, 2, 2, 1, 100}
	c = f64sDummy{1, 2, 2, 1, 100}

	correct := make(f64s, len(a))
	for i, v := range a {
		correct[i] = v * b[i]
	}

	// same type
	if err := a.Mul(b); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Mul is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// compatible type
	a = f64s{0, 1, 2, 3, 4}
	if err := a.Mul(c); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Mul is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.Mul(b[:3]); err == nil {
		t.Error("Expected an error when performing Mul on differing lengths")
	}

	// idiotsville 2
	if err := a.Mul(f32s{}); err == nil {
		t.Errorf("Expected an error when performing Mul on a non-compatible type")
	}

}

func Test_f32s_Mul(t *testing.T) {
	var a, b f32s
	var c f32sDummy

	a = f32s{0, 1, 2, 3, 4}
	b = f32s{1, 2, 2, 1, 100}
	c = f32sDummy{1, 2, 2, 1, 100}

	correct := make(f32s, len(a))
	for i, v := range a {
		correct[i] = v * b[i]
	}

	// same type
	if err := a.Mul(b); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Mul is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// compatible type
	a = f32s{0, 1, 2, 3, 4}
	if err := a.Mul(c); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Mul is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.Mul(b[:3]); err == nil {
		t.Error("Expected an error when performing Mul on differing lengths")
	}

	// idiotsville 2
	if err := a.Mul(f64s{}); err == nil {
		t.Errorf("Expected an error when performing Mul on a non-compatible type")
	}

}

func Test_ints_Mul(t *testing.T) {
	var a, b ints
	var c intsDummy

	a = ints{0, 1, 2, 3, 4}
	b = ints{1, 2, 2, 1, 100}
	c = intsDummy{1, 2, 2, 1, 100}

	correct := make(ints, len(a))
	for i, v := range a {
		correct[i] = v * b[i]
	}

	// same type
	if err := a.Mul(b); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Mul is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// compatible type
	a = ints{0, 1, 2, 3, 4}
	if err := a.Mul(c); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Mul is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.Mul(b[:3]); err == nil {
		t.Error("Expected an error when performing Mul on differing lengths")
	}

	// idiotsville 2
	if err := a.Mul(f64s{}); err == nil {
		t.Errorf("Expected an error when performing Mul on a non-compatible type")
	}

}

func Test_i64s_Mul(t *testing.T) {
	var a, b i64s
	var c i64sDummy

	a = i64s{0, 1, 2, 3, 4}
	b = i64s{1, 2, 2, 1, 100}
	c = i64sDummy{1, 2, 2, 1, 100}

	correct := make(i64s, len(a))
	for i, v := range a {
		correct[i] = v * b[i]
	}

	// same type
	if err := a.Mul(b); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Mul is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// compatible type
	a = i64s{0, 1, 2, 3, 4}
	if err := a.Mul(c); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Mul is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.Mul(b[:3]); err == nil {
		t.Error("Expected an error when performing Mul on differing lengths")
	}

	// idiotsville 2
	if err := a.Mul(f64s{}); err == nil {
		t.Errorf("Expected an error when performing Mul on a non-compatible type")
	}

}

func Test_i32s_Mul(t *testing.T) {
	var a, b i32s
	var c i32sDummy

	a = i32s{0, 1, 2, 3, 4}
	b = i32s{1, 2, 2, 1, 100}
	c = i32sDummy{1, 2, 2, 1, 100}

	correct := make(i32s, len(a))
	for i, v := range a {
		correct[i] = v * b[i]
	}

	// same type
	if err := a.Mul(b); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Mul is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// compatible type
	a = i32s{0, 1, 2, 3, 4}
	if err := a.Mul(c); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Mul is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.Mul(b[:3]); err == nil {
		t.Error("Expected an error when performing Mul on differing lengths")
	}

	// idiotsville 2
	if err := a.Mul(f64s{}); err == nil {
		t.Errorf("Expected an error when performing Mul on a non-compatible type")
	}

}

func Test_u8s_Mul(t *testing.T) {
	var a, b u8s
	var c u8sDummy

	a = u8s{0, 1, 2, 3, 4}
	b = u8s{1, 2, 2, 1, 100}
	c = u8sDummy{1, 2, 2, 1, 100}

	correct := make(u8s, len(a))
	for i, v := range a {
		correct[i] = v * b[i]
	}

	// same type
	if err := a.Mul(b); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Mul is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// compatible type
	a = u8s{0, 1, 2, 3, 4}
	if err := a.Mul(c); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Mul is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.Mul(b[:3]); err == nil {
		t.Error("Expected an error when performing Mul on differing lengths")
	}

	// idiotsville 2
	if err := a.Mul(f64s{}); err == nil {
		t.Errorf("Expected an error when performing Mul on a non-compatible type")
	}

}

/* Div */

func Test_f64s_Div(t *testing.T) {
	var a, b f64s
	var c f64sDummy

	a = f64s{0, 1, 2, 3, 4}
	b = f64s{1, 2, 2, 1, 100}
	c = f64sDummy{1, 2, 2, 1, 100}

	correct := make(f64s, len(a))
	for i, v := range a {
		correct[i] = v / b[i]
	}

	// same type
	if err := a.Div(b); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Div is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// compatible type
	a = f64s{0, 1, 2, 3, 4}
	if err := a.Div(c); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Div is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.Div(b[:3]); err == nil {
		t.Error("Expected an error when performing Div on differing lengths")
	}

	// idiotsville 2
	if err := a.Div(f32s{}); err == nil {
		t.Errorf("Expected an error when performing Div on a non-compatible type")
	}

}

func Test_f32s_Div(t *testing.T) {
	var a, b f32s
	var c f32sDummy

	a = f32s{0, 1, 2, 3, 4}
	b = f32s{1, 2, 2, 1, 100}
	c = f32sDummy{1, 2, 2, 1, 100}

	correct := make(f32s, len(a))
	for i, v := range a {
		correct[i] = v / b[i]
	}

	// same type
	if err := a.Div(b); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Div is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// compatible type
	a = f32s{0, 1, 2, 3, 4}
	if err := a.Div(c); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Div is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.Div(b[:3]); err == nil {
		t.Error("Expected an error when performing Div on differing lengths")
	}

	// idiotsville 2
	if err := a.Div(f64s{}); err == nil {
		t.Errorf("Expected an error when performing Div on a non-compatible type")
	}

}

func Test_ints_Div(t *testing.T) {
	var a, b ints
	var c intsDummy

	a = ints{0, 1, 2, 3, 4}
	b = ints{1, 2, 2, 1, 100}
	c = intsDummy{1, 2, 2, 1, 100}

	correct := make(ints, len(a))
	for i, v := range a {
		correct[i] = v / b[i]
	}

	// same type
	if err := a.Div(b); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Div is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// compatible type
	a = ints{0, 1, 2, 3, 4}
	if err := a.Div(c); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Div is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.Div(b[:3]); err == nil {
		t.Error("Expected an error when performing Div on differing lengths")
	}

	// idiotsville 2
	if err := a.Div(f64s{}); err == nil {
		t.Errorf("Expected an error when performing Div on a non-compatible type")
	}

}

func Test_i64s_Div(t *testing.T) {
	var a, b i64s
	var c i64sDummy

	a = i64s{0, 1, 2, 3, 4}
	b = i64s{1, 2, 2, 1, 100}
	c = i64sDummy{1, 2, 2, 1, 100}

	correct := make(i64s, len(a))
	for i, v := range a {
		correct[i] = v / b[i]
	}

	// same type
	if err := a.Div(b); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Div is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// compatible type
	a = i64s{0, 1, 2, 3, 4}
	if err := a.Div(c); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Div is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.Div(b[:3]); err == nil {
		t.Error("Expected an error when performing Div on differing lengths")
	}

	// idiotsville 2
	if err := a.Div(f64s{}); err == nil {
		t.Errorf("Expected an error when performing Div on a non-compatible type")
	}

}

func Test_i32s_Div(t *testing.T) {
	var a, b i32s
	var c i32sDummy

	a = i32s{0, 1, 2, 3, 4}
	b = i32s{1, 2, 2, 1, 100}
	c = i32sDummy{1, 2, 2, 1, 100}

	correct := make(i32s, len(a))
	for i, v := range a {
		correct[i] = v / b[i]
	}

	// same type
	if err := a.Div(b); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Div is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// compatible type
	a = i32s{0, 1, 2, 3, 4}
	if err := a.Div(c); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Div is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.Div(b[:3]); err == nil {
		t.Error("Expected an error when performing Div on differing lengths")
	}

	// idiotsville 2
	if err := a.Div(f64s{}); err == nil {
		t.Errorf("Expected an error when performing Div on a non-compatible type")
	}

}

func Test_u8s_Div(t *testing.T) {
	var a, b u8s
	var c u8sDummy

	a = u8s{0, 1, 2, 3, 4}
	b = u8s{1, 2, 2, 1, 100}
	c = u8sDummy{1, 2, 2, 1, 100}

	correct := make(u8s, len(a))
	for i, v := range a {
		correct[i] = v / b[i]
	}

	// same type
	if err := a.Div(b); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Div is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// compatible type
	a = u8s{0, 1, 2, 3, 4}
	if err := a.Div(c); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Div is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.Div(b[:3]); err == nil {
		t.Error("Expected an error when performing Div on differing lengths")
	}

	// idiotsville 2
	if err := a.Div(f64s{}); err == nil {
		t.Errorf("Expected an error when performing Div on a non-compatible type")
	}

}
