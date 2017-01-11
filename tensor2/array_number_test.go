package tensor

import (
	"math"
	"testing"
)

func prepf64sTest() (f64s, f64s, f64sDummy, float64) {
	a := f64s{0, 1, 2, 3, 4}
	b := f64s{1, 2, 2, 1, 100}
	c := f64sDummy{1, 2, 2, 1, 100}
	return a, b, c, 2
}

func prepf32sTest() (f32s, f32s, f32sDummy, float32) {
	a := f32s{0, 1, 2, 3, 4}
	b := f32s{1, 2, 2, 1, 100}
	c := f32sDummy{1, 2, 2, 1, 100}
	return a, b, c, 2
}

func prepintsTest() (ints, ints, intsDummy, int) {
	a := ints{0, 1, 2, 3, 4}
	b := ints{1, 2, 2, 1, 100}
	c := intsDummy{1, 2, 2, 1, 100}
	return a, b, c, 2
}

func prepi64sTest() (i64s, i64s, i64sDummy, int64) {
	a := i64s{0, 1, 2, 3, 4}
	b := i64s{1, 2, 2, 1, 100}
	c := i64sDummy{1, 2, 2, 1, 100}
	return a, b, c, 2
}

func prepi32sTest() (i32s, i32s, i32sDummy, int32) {
	a := i32s{0, 1, 2, 3, 4}
	b := i32s{1, 2, 2, 1, 100}
	c := i32sDummy{1, 2, 2, 1, 100}
	return a, b, c, 2
}

func prepu8sTest() (u8s, u8s, u8sDummy, byte) {
	a := u8s{0, 1, 2, 3, 4}
	b := u8s{1, 2, 2, 1, 100}
	c := u8sDummy{1, 2, 2, 1, 100}
	return a, b, c, 2
}

/* Add */

func Test_f64s_Add(t *testing.T) {
	a, b, c, _ := prepf64sTest()

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
	a, b, c, _ := prepf32sTest()

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
	a, b, c, _ := prepintsTest()

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
	a, b, c, _ := prepi64sTest()

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
	a, b, c, _ := prepi32sTest()

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
	a, b, c, _ := prepu8sTest()

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
	a, b, c, _ := prepf64sTest()

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
	a, b, c, _ := prepf32sTest()

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
	a, b, c, _ := prepintsTest()

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
	a, b, c, _ := prepi64sTest()

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
	a, b, c, _ := prepi32sTest()

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
	a, b, c, _ := prepu8sTest()

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
	a, b, c, _ := prepf64sTest()

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
	a, b, c, _ := prepf32sTest()

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
	a, b, c, _ := prepintsTest()

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
	a, b, c, _ := prepi64sTest()

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
	a, b, c, _ := prepi32sTest()

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
	a, b, c, _ := prepu8sTest()

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
	a, b, c, _ := prepf64sTest()

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
	a, b, c, _ := prepf32sTest()

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
	a, b, c, _ := prepintsTest()

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

	// additional tests for ScaleInv just for completeness sake
	b = ints{0, 1, 2, 3, 4}
	if err := a.Div(b); err == nil {
		t.Error("Expected an errrorIndices")
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
	a, b, c, _ := prepi64sTest()

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

	// additional tests for ScaleInv just for completeness sake
	b = i64s{0, 1, 2, 3, 4}
	if err := a.Div(b); err == nil {
		t.Error("Expected an errrorIndices")
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
	a, b, c, _ := prepi32sTest()

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

	// additional tests for ScaleInv just for completeness sake
	b = i32s{0, 1, 2, 3, 4}
	if err := a.Div(b); err == nil {
		t.Error("Expected an errrorIndices")
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
	a, b, c, _ := prepu8sTest()

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

	// additional tests for ScaleInv just for completeness sake
	b = u8s{0, 1, 2, 3, 4}
	if err := a.Div(b); err == nil {
		t.Error("Expected an errrorIndices")
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

/* Pow */

func Test_f64s_Pow(t *testing.T) {
	a, b, c, _ := prepf64sTest()

	correct := make(f64s, len(a))
	for i, v := range a {
		correct[i] = float64(math.Pow(float64(v), float64(b[i])))
	}

	// same type
	if err := a.Pow(b); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Pow is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// compatible type
	a = f64s{0, 1, 2, 3, 4}
	if err := a.Pow(c); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Pow is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.Pow(b[:3]); err == nil {
		t.Error("Expected an error when performing Pow on differing lengths")
	}

	// idiotsville 2
	if err := a.Pow(f32s{}); err == nil {
		t.Errorf("Expected an error when performing Pow on a non-compatible type")
	}
}

func Test_f32s_Pow(t *testing.T) {
	a, b, c, _ := prepf32sTest()

	correct := make(f32s, len(a))
	for i, v := range a {
		correct[i] = float32(math.Pow(float64(v), float64(b[i])))
	}

	// same type
	if err := a.Pow(b); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Pow is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// compatible type
	a = f32s{0, 1, 2, 3, 4}
	if err := a.Pow(c); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Pow is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.Pow(b[:3]); err == nil {
		t.Error("Expected an error when performing Pow on differing lengths")
	}

	// idiotsville 2
	if err := a.Pow(f64s{}); err == nil {
		t.Errorf("Expected an error when performing Pow on a non-compatible type")
	}
}

func Test_ints_Pow(t *testing.T) {
	a, b, c, _ := prepintsTest()

	correct := make(ints, len(a))
	for i, v := range a {
		correct[i] = int(math.Pow(float64(v), float64(b[i])))
	}

	// same type
	if err := a.Pow(b); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Pow is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// compatible type
	a = ints{0, 1, 2, 3, 4}
	if err := a.Pow(c); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Pow is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.Pow(b[:3]); err == nil {
		t.Error("Expected an error when performing Pow on differing lengths")
	}

	// idiotsville 2
	if err := a.Pow(f64s{}); err == nil {
		t.Errorf("Expected an error when performing Pow on a non-compatible type")
	}
}

func Test_i64s_Pow(t *testing.T) {
	a, b, c, _ := prepi64sTest()

	correct := make(i64s, len(a))
	for i, v := range a {
		correct[i] = int64(math.Pow(float64(v), float64(b[i])))
	}

	// same type
	if err := a.Pow(b); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Pow is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// compatible type
	a = i64s{0, 1, 2, 3, 4}
	if err := a.Pow(c); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Pow is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.Pow(b[:3]); err == nil {
		t.Error("Expected an error when performing Pow on differing lengths")
	}

	// idiotsville 2
	if err := a.Pow(f64s{}); err == nil {
		t.Errorf("Expected an error when performing Pow on a non-compatible type")
	}
}

func Test_i32s_Pow(t *testing.T) {
	a, b, c, _ := prepi32sTest()

	correct := make(i32s, len(a))
	for i, v := range a {
		correct[i] = int32(math.Pow(float64(v), float64(b[i])))
	}

	// same type
	if err := a.Pow(b); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Pow is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// compatible type
	a = i32s{0, 1, 2, 3, 4}
	if err := a.Pow(c); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Pow is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.Pow(b[:3]); err == nil {
		t.Error("Expected an error when performing Pow on differing lengths")
	}

	// idiotsville 2
	if err := a.Pow(f64s{}); err == nil {
		t.Errorf("Expected an error when performing Pow on a non-compatible type")
	}
}

func Test_u8s_Pow(t *testing.T) {
	a, b, c, _ := prepu8sTest()

	correct := make(u8s, len(a))
	for i, v := range a {
		correct[i] = byte(math.Pow(float64(v), float64(b[i])))
	}

	// same type
	if err := a.Pow(b); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Pow is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// compatible type
	a = u8s{0, 1, 2, 3, 4}
	if err := a.Pow(c); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Pow is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.Pow(b[:3]); err == nil {
		t.Error("Expected an error when performing Pow on differing lengths")
	}

	// idiotsville 2
	if err := a.Pow(f64s{}); err == nil {
		t.Errorf("Expected an error when performing Pow on a non-compatible type")
	}
}

/* Trans */

func Test_f64s_Trans(t *testing.T) {
	a, _, _, b := prepf64sTest()

	correct := make(f64s, len(a))
	for i, v := range a {
		correct[i] = v + b
	}

	if err := a.Trans(b); err != nil {
		t.Fatal(err)
	}
	for i, v := range a {
		// for floats we don't bother checking the incorrect stuff
		if v != correct[i] && i != 0 {
			t.Errorf("Trans is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.Trans(float32(2)); err == nil {
		t.Error("Expected an error when performing Trans on a differing type")
	}
}

func Test_f32s_Trans(t *testing.T) {
	a, _, _, b := prepf32sTest()

	correct := make(f32s, len(a))
	for i, v := range a {
		correct[i] = v + b
	}

	if err := a.Trans(b); err != nil {
		t.Fatal(err)
	}
	for i, v := range a {
		// for floats we don't bother checking the incorrect stuff
		if v != correct[i] && i != 0 {
			t.Errorf("Trans is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.Trans(2.0); err == nil {
		t.Error("Expected an error when performing Trans on a differing type")
	}
}

func Test_ints_Trans(t *testing.T) {
	a, _, _, b := prepintsTest()

	correct := make(ints, len(a))
	for i, v := range a {
		correct[i] = v + b
	}

	if err := a.Trans(b); err != nil {
		t.Fatal(err)
	}
	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Trans is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.Trans(2.0); err == nil {
		t.Error("Expected an error when performing Trans on a differing type")
	}
}

func Test_i64s_Trans(t *testing.T) {
	a, _, _, b := prepi64sTest()

	correct := make(i64s, len(a))
	for i, v := range a {
		correct[i] = v + b
	}

	if err := a.Trans(b); err != nil {
		t.Fatal(err)
	}
	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Trans is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.Trans(2.0); err == nil {
		t.Error("Expected an error when performing Trans on a differing type")
	}
}

func Test_i32s_Trans(t *testing.T) {
	a, _, _, b := prepi32sTest()

	correct := make(i32s, len(a))
	for i, v := range a {
		correct[i] = v + b
	}

	if err := a.Trans(b); err != nil {
		t.Fatal(err)
	}
	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Trans is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.Trans(2.0); err == nil {
		t.Error("Expected an error when performing Trans on a differing type")
	}
}

func Test_u8s_Trans(t *testing.T) {
	a, _, _, b := prepu8sTest()

	correct := make(u8s, len(a))
	for i, v := range a {
		correct[i] = v + b
	}

	if err := a.Trans(b); err != nil {
		t.Fatal(err)
	}
	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Trans is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.Trans(2.0); err == nil {
		t.Error("Expected an error when performing Trans on a differing type")
	}
}

/* TransInv */

func Test_f64s_TransInv(t *testing.T) {
	a, _, _, b := prepf64sTest()

	correct := make(f64s, len(a))
	for i, v := range a {
		correct[i] = v - b
	}

	if err := a.TransInv(b); err != nil {
		t.Fatal(err)
	}
	for i, v := range a {
		// for floats we don't bother checking the incorrect stuff
		if v != correct[i] && i != 0 {
			t.Errorf("TransInv is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.TransInv(float32(2)); err == nil {
		t.Error("Expected an error when performing TransInv on a differing type")
	}
}

func Test_f32s_TransInv(t *testing.T) {
	a, _, _, b := prepf32sTest()

	correct := make(f32s, len(a))
	for i, v := range a {
		correct[i] = v - b
	}

	if err := a.TransInv(b); err != nil {
		t.Fatal(err)
	}
	for i, v := range a {
		// for floats we don't bother checking the incorrect stuff
		if v != correct[i] && i != 0 {
			t.Errorf("TransInv is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.TransInv(2.0); err == nil {
		t.Error("Expected an error when performing TransInv on a differing type")
	}
}

func Test_ints_TransInv(t *testing.T) {
	a, _, _, b := prepintsTest()

	correct := make(ints, len(a))
	for i, v := range a {
		correct[i] = v - b
	}

	if err := a.TransInv(b); err != nil {
		t.Fatal(err)
	}
	for i, v := range a {
		if v != correct[i] {
			t.Errorf("TransInv is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.TransInv(2.0); err == nil {
		t.Error("Expected an error when performing TransInv on a differing type")
	}
}

func Test_i64s_TransInv(t *testing.T) {
	a, _, _, b := prepi64sTest()

	correct := make(i64s, len(a))
	for i, v := range a {
		correct[i] = v - b
	}

	if err := a.TransInv(b); err != nil {
		t.Fatal(err)
	}
	for i, v := range a {
		if v != correct[i] {
			t.Errorf("TransInv is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.TransInv(2.0); err == nil {
		t.Error("Expected an error when performing TransInv on a differing type")
	}
}

func Test_i32s_TransInv(t *testing.T) {
	a, _, _, b := prepi32sTest()

	correct := make(i32s, len(a))
	for i, v := range a {
		correct[i] = v - b
	}

	if err := a.TransInv(b); err != nil {
		t.Fatal(err)
	}
	for i, v := range a {
		if v != correct[i] {
			t.Errorf("TransInv is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.TransInv(2.0); err == nil {
		t.Error("Expected an error when performing TransInv on a differing type")
	}
}

func Test_u8s_TransInv(t *testing.T) {
	a, _, _, b := prepu8sTest()

	correct := make(u8s, len(a))
	for i, v := range a {
		correct[i] = v - b
	}

	if err := a.TransInv(b); err != nil {
		t.Fatal(err)
	}
	for i, v := range a {
		if v != correct[i] {
			t.Errorf("TransInv is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.TransInv(2.0); err == nil {
		t.Error("Expected an error when performing TransInv on a differing type")
	}
}

/* TransInvR */

func Test_f64s_TransInvR(t *testing.T) {
	a, _, _, b := prepf64sTest()

	correct := make(f64s, len(a))
	for i, v := range a {
		correct[i] = b - v
	}

	if err := a.TransInvR(b); err != nil {
		t.Fatal(err)
	}
	for i, v := range a {
		// for floats we don't bother checking the incorrect stuff
		if v != correct[i] && i != 0 {
			t.Errorf("TransInvR is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.TransInvR(float32(2)); err == nil {
		t.Error("Expected an error when performing TransInvR on a differing type")
	}
}

func Test_f32s_TransInvR(t *testing.T) {
	a, _, _, b := prepf32sTest()

	correct := make(f32s, len(a))
	for i, v := range a {
		correct[i] = b - v
	}

	if err := a.TransInvR(b); err != nil {
		t.Fatal(err)
	}
	for i, v := range a {
		// for floats we don't bother checking the incorrect stuff
		if v != correct[i] && i != 0 {
			t.Errorf("TransInvR is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.TransInvR(2.0); err == nil {
		t.Error("Expected an error when performing TransInvR on a differing type")
	}
}

func Test_ints_TransInvR(t *testing.T) {
	a, _, _, b := prepintsTest()

	correct := make(ints, len(a))
	for i, v := range a {
		correct[i] = b - v
	}

	if err := a.TransInvR(b); err != nil {
		t.Fatal(err)
	}
	for i, v := range a {
		if v != correct[i] {
			t.Errorf("TransInvR is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.TransInvR(2.0); err == nil {
		t.Error("Expected an error when performing TransInvR on a differing type")
	}
}

func Test_i64s_TransInvR(t *testing.T) {
	a, _, _, b := prepi64sTest()

	correct := make(i64s, len(a))
	for i, v := range a {
		correct[i] = b - v
	}

	if err := a.TransInvR(b); err != nil {
		t.Fatal(err)
	}
	for i, v := range a {
		if v != correct[i] {
			t.Errorf("TransInvR is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.TransInvR(2.0); err == nil {
		t.Error("Expected an error when performing TransInvR on a differing type")
	}
}

func Test_i32s_TransInvR(t *testing.T) {
	a, _, _, b := prepi32sTest()

	correct := make(i32s, len(a))
	for i, v := range a {
		correct[i] = b - v
	}

	if err := a.TransInvR(b); err != nil {
		t.Fatal(err)
	}
	for i, v := range a {
		if v != correct[i] {
			t.Errorf("TransInvR is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.TransInvR(2.0); err == nil {
		t.Error("Expected an error when performing TransInvR on a differing type")
	}
}

func Test_u8s_TransInvR(t *testing.T) {
	a, _, _, b := prepu8sTest()

	correct := make(u8s, len(a))
	for i, v := range a {
		correct[i] = b - v
	}

	if err := a.TransInvR(b); err != nil {
		t.Fatal(err)
	}
	for i, v := range a {
		if v != correct[i] {
			t.Errorf("TransInvR is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.TransInvR(2.0); err == nil {
		t.Error("Expected an error when performing TransInvR on a differing type")
	}
}

/* Scale */

func Test_f64s_Scale(t *testing.T) {
	a, _, _, b := prepf64sTest()

	correct := make(f64s, len(a))
	for i, v := range a {
		correct[i] = v * b
	}

	if err := a.Scale(b); err != nil {
		t.Fatal(err)
	}
	for i, v := range a {
		// for floats we don't bother checking the incorrect stuff
		if v != correct[i] && i != 0 {
			t.Errorf("Scale is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.Scale(float32(2)); err == nil {
		t.Error("Expected an error when performing Scale on a differing type")
	}
}

func Test_f32s_Scale(t *testing.T) {
	a, _, _, b := prepf32sTest()

	correct := make(f32s, len(a))
	for i, v := range a {
		correct[i] = v * b
	}

	if err := a.Scale(b); err != nil {
		t.Fatal(err)
	}
	for i, v := range a {
		// for floats we don't bother checking the incorrect stuff
		if v != correct[i] && i != 0 {
			t.Errorf("Scale is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.Scale(2.0); err == nil {
		t.Error("Expected an error when performing Scale on a differing type")
	}
}

func Test_ints_Scale(t *testing.T) {
	a, _, _, b := prepintsTest()

	correct := make(ints, len(a))
	for i, v := range a {
		correct[i] = v * b
	}

	if err := a.Scale(b); err != nil {
		t.Fatal(err)
	}
	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Scale is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.Scale(2.0); err == nil {
		t.Error("Expected an error when performing Scale on a differing type")
	}
}

func Test_i64s_Scale(t *testing.T) {
	a, _, _, b := prepi64sTest()

	correct := make(i64s, len(a))
	for i, v := range a {
		correct[i] = v * b
	}

	if err := a.Scale(b); err != nil {
		t.Fatal(err)
	}
	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Scale is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.Scale(2.0); err == nil {
		t.Error("Expected an error when performing Scale on a differing type")
	}
}

func Test_i32s_Scale(t *testing.T) {
	a, _, _, b := prepi32sTest()

	correct := make(i32s, len(a))
	for i, v := range a {
		correct[i] = v * b
	}

	if err := a.Scale(b); err != nil {
		t.Fatal(err)
	}
	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Scale is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.Scale(2.0); err == nil {
		t.Error("Expected an error when performing Scale on a differing type")
	}
}

func Test_u8s_Scale(t *testing.T) {
	a, _, _, b := prepu8sTest()

	correct := make(u8s, len(a))
	for i, v := range a {
		correct[i] = v * b
	}

	if err := a.Scale(b); err != nil {
		t.Fatal(err)
	}
	for i, v := range a {
		if v != correct[i] {
			t.Errorf("Scale is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.Scale(2.0); err == nil {
		t.Error("Expected an error when performing Scale on a differing type")
	}
}

/* ScaleInv */

func Test_f64s_ScaleInv(t *testing.T) {
	a, _, _, b := prepf64sTest()

	correct := make(f64s, len(a))
	for i, v := range a {
		if v == float64(0) {
			correct[i] = 0
			continue
		}
		correct[i] = v / b
	}

	if err := a.ScaleInv(b); err != nil {
		t.Fatal(err)
	}
	for i, v := range a {
		// for floats we don't bother checking the incorrect stuff
		if v != correct[i] && i != 0 {
			t.Errorf("ScaleInv is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.ScaleInv(float32(2)); err == nil {
		t.Error("Expected an error when performing ScaleInv on a differing type")
	}
}

func Test_f32s_ScaleInv(t *testing.T) {
	a, _, _, b := prepf32sTest()

	correct := make(f32s, len(a))
	for i, v := range a {
		if v == float32(0) {
			correct[i] = 0
			continue
		}
		correct[i] = v / b
	}

	if err := a.ScaleInv(b); err != nil {
		t.Fatal(err)
	}
	for i, v := range a {
		// for floats we don't bother checking the incorrect stuff
		if v != correct[i] && i != 0 {
			t.Errorf("ScaleInv is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.ScaleInv(2.0); err == nil {
		t.Error("Expected an error when performing ScaleInv on a differing type")
	}
}

func Test_ints_ScaleInv(t *testing.T) {
	a, _, _, b := prepintsTest()

	correct := make(ints, len(a))
	for i, v := range a {
		if v == int(0) {
			correct[i] = 0
			continue
		}
		correct[i] = v / b
	}

	err := a.ScaleInv(b)
	if err == nil {
		t.Error("Expected error (division by zero)")
	}
	if _, ok := err.(errorIndices); !ok {
		t.Fatal(err)
	}
	for i, v := range a {
		if v != correct[i] {
			t.Errorf("ScaleInv is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.ScaleInv(2.0); err == nil {
		t.Error("Expected an error when performing ScaleInv on a differing type")
	}
}

func Test_i64s_ScaleInv(t *testing.T) {
	a, _, _, b := prepi64sTest()

	correct := make(i64s, len(a))
	for i, v := range a {
		if v == int64(0) {
			correct[i] = 0
			continue
		}
		correct[i] = v / b
	}

	err := a.ScaleInv(b)
	if err == nil {
		t.Error("Expected error (division by zero)")
	}
	if _, ok := err.(errorIndices); !ok {
		t.Fatal(err)
	}
	for i, v := range a {
		if v != correct[i] {
			t.Errorf("ScaleInv is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.ScaleInv(2.0); err == nil {
		t.Error("Expected an error when performing ScaleInv on a differing type")
	}
}

func Test_i32s_ScaleInv(t *testing.T) {
	a, _, _, b := prepi32sTest()

	correct := make(i32s, len(a))
	for i, v := range a {
		if v == int32(0) {
			correct[i] = 0
			continue
		}
		correct[i] = v / b
	}

	err := a.ScaleInv(b)
	if err == nil {
		t.Error("Expected error (division by zero)")
	}
	if _, ok := err.(errorIndices); !ok {
		t.Fatal(err)
	}
	for i, v := range a {
		if v != correct[i] {
			t.Errorf("ScaleInv is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.ScaleInv(2.0); err == nil {
		t.Error("Expected an error when performing ScaleInv on a differing type")
	}
}

func Test_u8s_ScaleInv(t *testing.T) {
	a, _, _, b := prepu8sTest()

	correct := make(u8s, len(a))
	for i, v := range a {
		if v == byte(0) {
			correct[i] = 0
			continue
		}
		correct[i] = v / b
	}

	err := a.ScaleInv(b)
	if err == nil {
		t.Error("Expected error (division by zero)")
	}
	if _, ok := err.(errorIndices); !ok {
		t.Fatal(err)
	}
	for i, v := range a {
		if v != correct[i] {
			t.Errorf("ScaleInv is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.ScaleInv(2.0); err == nil {
		t.Error("Expected an error when performing ScaleInv on a differing type")
	}
}

/* ScaleInvR */

func Test_f64s_ScaleInvR(t *testing.T) {
	a, _, _, b := prepf64sTest()

	correct := make(f64s, len(a))
	for i, v := range a {
		if v == float64(0) {
			correct[i] = 0
			continue
		}
		correct[i] = b / v
	}

	if err := a.ScaleInvR(b); err != nil {
		t.Fatal(err)
	}
	for i, v := range a {
		// for floats we don't bother checking the incorrect stuff
		if v != correct[i] && i != 0 {
			t.Errorf("ScaleInvR is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.ScaleInvR(float32(2)); err == nil {
		t.Error("Expected an error when performing ScaleInvR on a differing type")
	}
}

func Test_f32s_ScaleInvR(t *testing.T) {
	a, _, _, b := prepf32sTest()

	correct := make(f32s, len(a))
	for i, v := range a {
		if v == float32(0) {
			correct[i] = 0
			continue
		}
		correct[i] = b / v
	}

	if err := a.ScaleInvR(b); err != nil {
		t.Fatal(err)
	}
	for i, v := range a {
		// for floats we don't bother checking the incorrect stuff
		if v != correct[i] && i != 0 {
			t.Errorf("ScaleInvR is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.ScaleInvR(2.0); err == nil {
		t.Error("Expected an error when performing ScaleInvR on a differing type")
	}
}

func Test_ints_ScaleInvR(t *testing.T) {
	a, _, _, b := prepintsTest()

	correct := make(ints, len(a))
	for i, v := range a {
		if v == int(0) {
			correct[i] = 0
			continue
		}
		correct[i] = b / v
	}

	err := a.ScaleInvR(b)
	if err == nil {
		t.Error("Expected error (division by zero)")
	}
	if _, ok := err.(errorIndices); !ok {
		t.Fatal(err)
	}
	for i, v := range a {
		if v != correct[i] {
			t.Errorf("ScaleInvR is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.ScaleInvR(2.0); err == nil {
		t.Error("Expected an error when performing ScaleInvR on a differing type")
	}
}

func Test_i64s_ScaleInvR(t *testing.T) {
	a, _, _, b := prepi64sTest()

	correct := make(i64s, len(a))
	for i, v := range a {
		if v == int64(0) {
			correct[i] = 0
			continue
		}
		correct[i] = b / v
	}

	err := a.ScaleInvR(b)
	if err == nil {
		t.Error("Expected error (division by zero)")
	}
	if _, ok := err.(errorIndices); !ok {
		t.Fatal(err)
	}
	for i, v := range a {
		if v != correct[i] {
			t.Errorf("ScaleInvR is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.ScaleInvR(2.0); err == nil {
		t.Error("Expected an error when performing ScaleInvR on a differing type")
	}
}

func Test_i32s_ScaleInvR(t *testing.T) {
	a, _, _, b := prepi32sTest()

	correct := make(i32s, len(a))
	for i, v := range a {
		if v == int32(0) {
			correct[i] = 0
			continue
		}
		correct[i] = b / v
	}

	err := a.ScaleInvR(b)
	if err == nil {
		t.Error("Expected error (division by zero)")
	}
	if _, ok := err.(errorIndices); !ok {
		t.Fatal(err)
	}
	for i, v := range a {
		if v != correct[i] {
			t.Errorf("ScaleInvR is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.ScaleInvR(2.0); err == nil {
		t.Error("Expected an error when performing ScaleInvR on a differing type")
	}
}

func Test_u8s_ScaleInvR(t *testing.T) {
	a, _, _, b := prepu8sTest()

	correct := make(u8s, len(a))
	for i, v := range a {
		if v == byte(0) {
			correct[i] = 0
			continue
		}
		correct[i] = b / v
	}

	err := a.ScaleInvR(b)
	if err == nil {
		t.Error("Expected error (division by zero)")
	}
	if _, ok := err.(errorIndices); !ok {
		t.Fatal(err)
	}
	for i, v := range a {
		if v != correct[i] {
			t.Errorf("ScaleInvR is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.ScaleInvR(2.0); err == nil {
		t.Error("Expected an error when performing ScaleInvR on a differing type")
	}
}

/* PowOf */

func Test_f64s_PowOf(t *testing.T) {
	a, _, _, b := prepf64sTest()

	correct := make(f64s, len(a))
	for i, v := range a {
		correct[i] = float64(math.Pow(float64(v), float64(b)))
	}

	if err := a.PowOf(b); err != nil {
		t.Fatal(err)
	}
	for i, v := range a {
		// for floats we don't bother checking the incorrect stuff
		if v != correct[i] && i != 0 {
			t.Errorf("PowOf is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.PowOf(float32(2)); err == nil {
		t.Error("Expected an error when performing PowOf on a differing type")
	}
}

func Test_f32s_PowOf(t *testing.T) {
	a, _, _, b := prepf32sTest()

	correct := make(f32s, len(a))
	for i, v := range a {
		correct[i] = float32(math.Pow(float64(v), float64(b)))
	}

	if err := a.PowOf(b); err != nil {
		t.Fatal(err)
	}
	for i, v := range a {
		// for floats we don't bother checking the incorrect stuff
		if v != correct[i] && i != 0 {
			t.Errorf("PowOf is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.PowOf(2.0); err == nil {
		t.Error("Expected an error when performing PowOf on a differing type")
	}
}

func Test_ints_PowOf(t *testing.T) {
	a, _, _, b := prepintsTest()

	correct := make(ints, len(a))
	for i, v := range a {
		correct[i] = int(math.Pow(float64(v), float64(b)))
	}

	if err := a.PowOf(b); err != nil {
		t.Fatal(err)
	}
	for i, v := range a {
		if v != correct[i] {
			t.Errorf("PowOf is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.PowOf(2.0); err == nil {
		t.Error("Expected an error when performing PowOf on a differing type")
	}
}

func Test_i64s_PowOf(t *testing.T) {
	a, _, _, b := prepi64sTest()

	correct := make(i64s, len(a))
	for i, v := range a {
		correct[i] = int64(math.Pow(float64(v), float64(b)))
	}

	if err := a.PowOf(b); err != nil {
		t.Fatal(err)
	}
	for i, v := range a {
		if v != correct[i] {
			t.Errorf("PowOf is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.PowOf(2.0); err == nil {
		t.Error("Expected an error when performing PowOf on a differing type")
	}
}

func Test_i32s_PowOf(t *testing.T) {
	a, _, _, b := prepi32sTest()

	correct := make(i32s, len(a))
	for i, v := range a {
		correct[i] = int32(math.Pow(float64(v), float64(b)))
	}

	if err := a.PowOf(b); err != nil {
		t.Fatal(err)
	}
	for i, v := range a {
		if v != correct[i] {
			t.Errorf("PowOf is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.PowOf(2.0); err == nil {
		t.Error("Expected an error when performing PowOf on a differing type")
	}
}

func Test_u8s_PowOf(t *testing.T) {
	a, _, _, b := prepu8sTest()

	correct := make(u8s, len(a))
	for i, v := range a {
		correct[i] = byte(math.Pow(float64(v), float64(b)))
	}

	if err := a.PowOf(b); err != nil {
		t.Fatal(err)
	}
	for i, v := range a {
		if v != correct[i] {
			t.Errorf("PowOf is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.PowOf(2.0); err == nil {
		t.Error("Expected an error when performing PowOf on a differing type")
	}
}

/* PowOfR */

func Test_f64s_PowOfR(t *testing.T) {
	a, _, _, b := prepf64sTest()

	correct := make(f64s, len(a))
	for i, v := range a {
		correct[i] = float64(math.Pow(float64(b), float64(v)))
	}

	if err := a.PowOfR(b); err != nil {
		t.Fatal(err)
	}
	for i, v := range a {
		// for floats we don't bother checking the incorrect stuff
		if v != correct[i] && i != 0 {
			t.Errorf("PowOfR is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.PowOfR(float32(2)); err == nil {
		t.Error("Expected an error when performing PowOfR on a differing type")
	}
}

func Test_f32s_PowOfR(t *testing.T) {
	a, _, _, b := prepf32sTest()

	correct := make(f32s, len(a))
	for i, v := range a {
		correct[i] = float32(math.Pow(float64(b), float64(v)))
	}

	if err := a.PowOfR(b); err != nil {
		t.Fatal(err)
	}
	for i, v := range a {
		// for floats we don't bother checking the incorrect stuff
		if v != correct[i] && i != 0 {
			t.Errorf("PowOfR is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.PowOfR(2.0); err == nil {
		t.Error("Expected an error when performing PowOfR on a differing type")
	}
}

func Test_ints_PowOfR(t *testing.T) {
	a, _, _, b := prepintsTest()

	correct := make(ints, len(a))
	for i, v := range a {
		correct[i] = int(math.Pow(float64(b), float64(v)))
	}

	if err := a.PowOfR(b); err != nil {
		t.Fatal(err)
	}
	for i, v := range a {
		if v != correct[i] {
			t.Errorf("PowOfR is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.PowOfR(2.0); err == nil {
		t.Error("Expected an error when performing PowOfR on a differing type")
	}
}

func Test_i64s_PowOfR(t *testing.T) {
	a, _, _, b := prepi64sTest()

	correct := make(i64s, len(a))
	for i, v := range a {
		correct[i] = int64(math.Pow(float64(b), float64(v)))
	}

	if err := a.PowOfR(b); err != nil {
		t.Fatal(err)
	}
	for i, v := range a {
		if v != correct[i] {
			t.Errorf("PowOfR is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.PowOfR(2.0); err == nil {
		t.Error("Expected an error when performing PowOfR on a differing type")
	}
}

func Test_i32s_PowOfR(t *testing.T) {
	a, _, _, b := prepi32sTest()

	correct := make(i32s, len(a))
	for i, v := range a {
		correct[i] = int32(math.Pow(float64(b), float64(v)))
	}

	if err := a.PowOfR(b); err != nil {
		t.Fatal(err)
	}
	for i, v := range a {
		if v != correct[i] {
			t.Errorf("PowOfR is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.PowOfR(2.0); err == nil {
		t.Error("Expected an error when performing PowOfR on a differing type")
	}
}

func Test_u8s_PowOfR(t *testing.T) {
	a, _, _, b := prepu8sTest()

	correct := make(u8s, len(a))
	for i, v := range a {
		correct[i] = byte(math.Pow(float64(b), float64(v)))
	}

	if err := a.PowOfR(b); err != nil {
		t.Fatal(err)
	}
	for i, v := range a {
		if v != correct[i] {
			t.Errorf("PowOfR is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.PowOfR(2.0); err == nil {
		t.Error("Expected an error when performing PowOfR on a differing type")
	}
}
