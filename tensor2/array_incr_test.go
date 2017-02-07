package tensor

import (
	"math"
	"testing"
)

/*
GENERATED FILE. DO NOT EDIT
*/

/* Add */

func Test_f64s_IncrAdd(t *testing.T) {
	a, b, _, _ := prepf64sTest()
	incr := f64s{100, 100, 100, 100, 100}

	correct := make(f64s, len(a))
	for i, v := range a {
		correct[i] = v + b[i] + incr[i]
	}

	// same type
	if err := a.IncrAdd(b, incr); err != nil {
		t.Error(err)
	}

	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrAdd is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}
}

func Test_f32s_IncrAdd(t *testing.T) {
	a, b, _, _ := prepf32sTest()
	incr := f32s{100, 100, 100, 100, 100}

	correct := make(f32s, len(a))
	for i, v := range a {
		correct[i] = v + b[i] + incr[i]
	}

	// same type
	if err := a.IncrAdd(b, incr); err != nil {
		t.Error(err)
	}

	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrAdd is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}
}

func Test_ints_IncrAdd(t *testing.T) {
	a, b, _, _ := prepintsTest()
	incr := ints{100, 100, 100, 100, 100}

	correct := make(ints, len(a))
	for i, v := range a {
		correct[i] = v + b[i] + incr[i]
	}

	// same type
	if err := a.IncrAdd(b, incr); err != nil {
		t.Error(err)
	}

	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrAdd is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}
}

func Test_i64s_IncrAdd(t *testing.T) {
	a, b, _, _ := prepi64sTest()
	incr := i64s{100, 100, 100, 100, 100}

	correct := make(i64s, len(a))
	for i, v := range a {
		correct[i] = v + b[i] + incr[i]
	}

	// same type
	if err := a.IncrAdd(b, incr); err != nil {
		t.Error(err)
	}

	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrAdd is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}
}

func Test_i32s_IncrAdd(t *testing.T) {
	a, b, _, _ := prepi32sTest()
	incr := i32s{100, 100, 100, 100, 100}

	correct := make(i32s, len(a))
	for i, v := range a {
		correct[i] = v + b[i] + incr[i]
	}

	// same type
	if err := a.IncrAdd(b, incr); err != nil {
		t.Error(err)
	}

	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrAdd is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}
}

func Test_u8s_IncrAdd(t *testing.T) {
	a, b, _, _ := prepu8sTest()
	incr := u8s{100, 100, 100, 100, 100}

	correct := make(u8s, len(a))
	for i, v := range a {
		correct[i] = v + b[i] + incr[i]
	}

	// same type
	if err := a.IncrAdd(b, incr); err != nil {
		t.Error(err)
	}

	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrAdd is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}
}

/* Sub */

func Test_f64s_IncrSub(t *testing.T) {
	a, b, _, _ := prepf64sTest()
	incr := f64s{100, 100, 100, 100, 100}

	correct := make(f64s, len(a))
	for i, v := range a {
		correct[i] = v - b[i] + incr[i]
	}

	// same type
	if err := a.IncrSub(b, incr); err != nil {
		t.Error(err)
	}

	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrSub is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}
}

func Test_f32s_IncrSub(t *testing.T) {
	a, b, _, _ := prepf32sTest()
	incr := f32s{100, 100, 100, 100, 100}

	correct := make(f32s, len(a))
	for i, v := range a {
		correct[i] = v - b[i] + incr[i]
	}

	// same type
	if err := a.IncrSub(b, incr); err != nil {
		t.Error(err)
	}

	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrSub is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}
}

func Test_ints_IncrSub(t *testing.T) {
	a, b, _, _ := prepintsTest()
	incr := ints{100, 100, 100, 100, 100}

	correct := make(ints, len(a))
	for i, v := range a {
		correct[i] = v - b[i] + incr[i]
	}

	// same type
	if err := a.IncrSub(b, incr); err != nil {
		t.Error(err)
	}

	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrSub is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}
}

func Test_i64s_IncrSub(t *testing.T) {
	a, b, _, _ := prepi64sTest()
	incr := i64s{100, 100, 100, 100, 100}

	correct := make(i64s, len(a))
	for i, v := range a {
		correct[i] = v - b[i] + incr[i]
	}

	// same type
	if err := a.IncrSub(b, incr); err != nil {
		t.Error(err)
	}

	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrSub is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}
}

func Test_i32s_IncrSub(t *testing.T) {
	a, b, _, _ := prepi32sTest()
	incr := i32s{100, 100, 100, 100, 100}

	correct := make(i32s, len(a))
	for i, v := range a {
		correct[i] = v - b[i] + incr[i]
	}

	// same type
	if err := a.IncrSub(b, incr); err != nil {
		t.Error(err)
	}

	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrSub is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}
}

func Test_u8s_IncrSub(t *testing.T) {
	a, b, _, _ := prepu8sTest()
	incr := u8s{100, 100, 100, 100, 100}

	correct := make(u8s, len(a))
	for i, v := range a {
		correct[i] = v - b[i] + incr[i]
	}

	// same type
	if err := a.IncrSub(b, incr); err != nil {
		t.Error(err)
	}

	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrSub is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}
}

/* Mul */

func Test_f64s_IncrMul(t *testing.T) {
	a, b, _, _ := prepf64sTest()
	incr := f64s{100, 100, 100, 100, 100}

	correct := make(f64s, len(a))
	for i, v := range a {
		correct[i] = v*b[i] + incr[i]
	}

	// same type
	if err := a.IncrMul(b, incr); err != nil {
		t.Error(err)
	}

	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrMul is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}
}

func Test_f32s_IncrMul(t *testing.T) {
	a, b, _, _ := prepf32sTest()
	incr := f32s{100, 100, 100, 100, 100}

	correct := make(f32s, len(a))
	for i, v := range a {
		correct[i] = v*b[i] + incr[i]
	}

	// same type
	if err := a.IncrMul(b, incr); err != nil {
		t.Error(err)
	}

	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrMul is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}
}

func Test_ints_IncrMul(t *testing.T) {
	a, b, _, _ := prepintsTest()
	incr := ints{100, 100, 100, 100, 100}

	correct := make(ints, len(a))
	for i, v := range a {
		correct[i] = v*b[i] + incr[i]
	}

	// same type
	if err := a.IncrMul(b, incr); err != nil {
		t.Error(err)
	}

	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrMul is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}
}

func Test_i64s_IncrMul(t *testing.T) {
	a, b, _, _ := prepi64sTest()
	incr := i64s{100, 100, 100, 100, 100}

	correct := make(i64s, len(a))
	for i, v := range a {
		correct[i] = v*b[i] + incr[i]
	}

	// same type
	if err := a.IncrMul(b, incr); err != nil {
		t.Error(err)
	}

	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrMul is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}
}

func Test_i32s_IncrMul(t *testing.T) {
	a, b, _, _ := prepi32sTest()
	incr := i32s{100, 100, 100, 100, 100}

	correct := make(i32s, len(a))
	for i, v := range a {
		correct[i] = v*b[i] + incr[i]
	}

	// same type
	if err := a.IncrMul(b, incr); err != nil {
		t.Error(err)
	}

	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrMul is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}
}

func Test_u8s_IncrMul(t *testing.T) {
	a, b, _, _ := prepu8sTest()
	incr := u8s{100, 100, 100, 100, 100}

	correct := make(u8s, len(a))
	for i, v := range a {
		correct[i] = v*b[i] + incr[i]
	}

	// same type
	if err := a.IncrMul(b, incr); err != nil {
		t.Error(err)
	}

	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrMul is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}
}

/* Div */

func Test_f64s_IncrDiv(t *testing.T) {
	a, b, _, _ := prepf64sTest()
	incr := f64s{100, 100, 100, 100, 100}

	correct := make(f64s, len(a))
	for i, v := range a {
		correct[i] = v/b[i] + incr[i]
	}

	// same type
	if err := a.IncrDiv(b, incr); err != nil {
		t.Error(err)
	}

	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrDiv is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}
}

func Test_f32s_IncrDiv(t *testing.T) {
	a, b, _, _ := prepf32sTest()
	incr := f32s{100, 100, 100, 100, 100}

	correct := make(f32s, len(a))
	for i, v := range a {
		correct[i] = v/b[i] + incr[i]
	}

	// same type
	if err := a.IncrDiv(b, incr); err != nil {
		t.Error(err)
	}

	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrDiv is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}
}

func Test_ints_IncrDiv(t *testing.T) {
	a, b, _, _ := prepintsTest()
	incr := ints{100, 100, 100, 100, 100}

	correct := make(ints, len(a))
	for i, v := range a {
		correct[i] = v/b[i] + incr[i]
	}

	// same type
	if err := a.IncrDiv(b, incr); err != nil {
		t.Error(err)
	}

	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrDiv is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}
}

func Test_i64s_IncrDiv(t *testing.T) {
	a, b, _, _ := prepi64sTest()
	incr := i64s{100, 100, 100, 100, 100}

	correct := make(i64s, len(a))
	for i, v := range a {
		correct[i] = v/b[i] + incr[i]
	}

	// same type
	if err := a.IncrDiv(b, incr); err != nil {
		t.Error(err)
	}

	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrDiv is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}
}

func Test_i32s_IncrDiv(t *testing.T) {
	a, b, _, _ := prepi32sTest()
	incr := i32s{100, 100, 100, 100, 100}

	correct := make(i32s, len(a))
	for i, v := range a {
		correct[i] = v/b[i] + incr[i]
	}

	// same type
	if err := a.IncrDiv(b, incr); err != nil {
		t.Error(err)
	}

	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrDiv is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}
}

func Test_u8s_IncrDiv(t *testing.T) {
	a, b, _, _ := prepu8sTest()
	incr := u8s{100, 100, 100, 100, 100}

	correct := make(u8s, len(a))
	for i, v := range a {
		correct[i] = v/b[i] + incr[i]
	}

	// same type
	if err := a.IncrDiv(b, incr); err != nil {
		t.Error(err)
	}

	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrDiv is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}
}

/* Pow */

func Test_f64s_IncrPow(t *testing.T) {
	a, b, _, _ := prepf64sTest()
	incr := f64s{100, 100, 100, 100, 100}

	correct := make(f64s, len(a))
	for i, v := range a {
		correct[i] = float64(math.Pow(float64(v), float64(b[i]))) + incr[i]
	}

	// same type
	if err := a.IncrPow(b, incr); err != nil {
		t.Error(err)
	}

	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrPow is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}
}

func Test_f32s_IncrPow(t *testing.T) {
	a, b, _, _ := prepf32sTest()
	incr := f32s{100, 100, 100, 100, 100}

	correct := make(f32s, len(a))
	for i, v := range a {
		correct[i] = float32(math.Pow(float64(v), float64(b[i]))) + incr[i]
	}

	// same type
	if err := a.IncrPow(b, incr); err != nil {
		t.Error(err)
	}

	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrPow is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}
}

func Test_ints_IncrPow(t *testing.T) {
	a, b, _, _ := prepintsTest()
	incr := ints{100, 100, 100, 100, 100}

	correct := make(ints, len(a))
	for i, v := range a {
		correct[i] = int(math.Pow(float64(v), float64(b[i]))) + incr[i]
	}

	// same type
	if err := a.IncrPow(b, incr); err != nil {
		t.Error(err)
	}

	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrPow is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}
}

func Test_i64s_IncrPow(t *testing.T) {
	a, b, _, _ := prepi64sTest()
	incr := i64s{100, 100, 100, 100, 100}

	correct := make(i64s, len(a))
	for i, v := range a {
		correct[i] = int64(math.Pow(float64(v), float64(b[i]))) + incr[i]
	}

	// same type
	if err := a.IncrPow(b, incr); err != nil {
		t.Error(err)
	}

	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrPow is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}
}

func Test_i32s_IncrPow(t *testing.T) {
	a, b, _, _ := prepi32sTest()
	incr := i32s{100, 100, 100, 100, 100}

	correct := make(i32s, len(a))
	for i, v := range a {
		correct[i] = int32(math.Pow(float64(v), float64(b[i]))) + incr[i]
	}

	// same type
	if err := a.IncrPow(b, incr); err != nil {
		t.Error(err)
	}

	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrPow is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}
}

func Test_u8s_IncrPow(t *testing.T) {
	a, b, _, _ := prepu8sTest()
	incr := u8s{100, 100, 100, 100, 100}

	correct := make(u8s, len(a))
	for i, v := range a {
		correct[i] = byte(math.Pow(float64(v), float64(b[i]))) + incr[i]
	}

	// same type
	if err := a.IncrPow(b, incr); err != nil {
		t.Error(err)
	}

	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrPow is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}
}

/* Trans */

func Test_f64s_IncrTrans(t *testing.T) {
	a, _, _, b := prepf64sTest()
	incr := f64s{100, 100, 100, 100, 100}

	correct := make(f64s, len(a))
	for i, v := range a {
		correct[i] = v + b + incr[i]
	}

	if err := a.IncrTrans(b, incr); err != nil {
		t.Fatal(err)
	}
	for i, v := range incr {
		// for floats we don't bother checking the incorrect stuff
		if v != correct[i] && i != 0 {
			t.Errorf("IncrTrans is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.Trans(float32(2)); err == nil {
		t.Error("Expected an error when performing Trans on a differing type")
	}
}

func Test_f32s_IncrTrans(t *testing.T) {
	a, _, _, b := prepf32sTest()
	incr := f32s{100, 100, 100, 100, 100}

	correct := make(f32s, len(a))
	for i, v := range a {
		correct[i] = v + b + incr[i]
	}

	if err := a.IncrTrans(b, incr); err != nil {
		t.Fatal(err)
	}
	for i, v := range incr {
		// for floats we don't bother checking the incorrect stuff
		if v != correct[i] && i != 0 {
			t.Errorf("IncrTrans is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.Trans(2.0); err == nil {
		t.Error("Expected an error when performing Trans on a differing type")
	}
}

func Test_ints_IncrTrans(t *testing.T) {
	a, _, _, b := prepintsTest()
	incr := ints{100, 100, 100, 100, 100}

	correct := make(ints, len(a))
	for i, v := range a {
		correct[i] = v + b + incr[i]
	}

	if err := a.IncrTrans(b, incr); err != nil {
		t.Fatal(err)
	}
	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrTrans is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.Trans(2.0); err == nil {
		t.Error("Expected an error when performing Trans on a differing type")
	}
}

func Test_i64s_IncrTrans(t *testing.T) {
	a, _, _, b := prepi64sTest()
	incr := i64s{100, 100, 100, 100, 100}

	correct := make(i64s, len(a))
	for i, v := range a {
		correct[i] = v + b + incr[i]
	}

	if err := a.IncrTrans(b, incr); err != nil {
		t.Fatal(err)
	}
	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrTrans is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.Trans(2.0); err == nil {
		t.Error("Expected an error when performing Trans on a differing type")
	}
}

func Test_i32s_IncrTrans(t *testing.T) {
	a, _, _, b := prepi32sTest()
	incr := i32s{100, 100, 100, 100, 100}

	correct := make(i32s, len(a))
	for i, v := range a {
		correct[i] = v + b + incr[i]
	}

	if err := a.IncrTrans(b, incr); err != nil {
		t.Fatal(err)
	}
	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrTrans is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.Trans(2.0); err == nil {
		t.Error("Expected an error when performing Trans on a differing type")
	}
}

func Test_u8s_IncrTrans(t *testing.T) {
	a, _, _, b := prepu8sTest()
	incr := u8s{100, 100, 100, 100, 100}

	correct := make(u8s, len(a))
	for i, v := range a {
		correct[i] = v + b + incr[i]
	}

	if err := a.IncrTrans(b, incr); err != nil {
		t.Fatal(err)
	}
	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrTrans is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.Trans(2.0); err == nil {
		t.Error("Expected an error when performing Trans on a differing type")
	}
}

/* TransInv */

func Test_f64s_IncrTransInv(t *testing.T) {
	a, _, _, b := prepf64sTest()
	incr := f64s{100, 100, 100, 100, 100}

	correct := make(f64s, len(a))
	for i, v := range a {
		correct[i] = v - b + incr[i]
	}

	if err := a.IncrTransInv(b, incr); err != nil {
		t.Fatal(err)
	}
	for i, v := range incr {
		// for floats we don't bother checking the incorrect stuff
		if v != correct[i] && i != 0 {
			t.Errorf("IncrTransInv is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.TransInv(float32(2)); err == nil {
		t.Error("Expected an error when performing TransInv on a differing type")
	}
}

func Test_f32s_IncrTransInv(t *testing.T) {
	a, _, _, b := prepf32sTest()
	incr := f32s{100, 100, 100, 100, 100}

	correct := make(f32s, len(a))
	for i, v := range a {
		correct[i] = v - b + incr[i]
	}

	if err := a.IncrTransInv(b, incr); err != nil {
		t.Fatal(err)
	}
	for i, v := range incr {
		// for floats we don't bother checking the incorrect stuff
		if v != correct[i] && i != 0 {
			t.Errorf("IncrTransInv is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.TransInv(2.0); err == nil {
		t.Error("Expected an error when performing TransInv on a differing type")
	}
}

func Test_ints_IncrTransInv(t *testing.T) {
	a, _, _, b := prepintsTest()
	incr := ints{100, 100, 100, 100, 100}

	correct := make(ints, len(a))
	for i, v := range a {
		correct[i] = v - b + incr[i]
	}

	if err := a.IncrTransInv(b, incr); err != nil {
		t.Fatal(err)
	}
	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrTransInv is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.TransInv(2.0); err == nil {
		t.Error("Expected an error when performing TransInv on a differing type")
	}
}

func Test_i64s_IncrTransInv(t *testing.T) {
	a, _, _, b := prepi64sTest()
	incr := i64s{100, 100, 100, 100, 100}

	correct := make(i64s, len(a))
	for i, v := range a {
		correct[i] = v - b + incr[i]
	}

	if err := a.IncrTransInv(b, incr); err != nil {
		t.Fatal(err)
	}
	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrTransInv is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.TransInv(2.0); err == nil {
		t.Error("Expected an error when performing TransInv on a differing type")
	}
}

func Test_i32s_IncrTransInv(t *testing.T) {
	a, _, _, b := prepi32sTest()
	incr := i32s{100, 100, 100, 100, 100}

	correct := make(i32s, len(a))
	for i, v := range a {
		correct[i] = v - b + incr[i]
	}

	if err := a.IncrTransInv(b, incr); err != nil {
		t.Fatal(err)
	}
	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrTransInv is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.TransInv(2.0); err == nil {
		t.Error("Expected an error when performing TransInv on a differing type")
	}
}

func Test_u8s_IncrTransInv(t *testing.T) {
	a, _, _, b := prepu8sTest()
	incr := u8s{100, 100, 100, 100, 100}

	correct := make(u8s, len(a))
	for i, v := range a {
		correct[i] = v - b + incr[i]
	}

	if err := a.IncrTransInv(b, incr); err != nil {
		t.Fatal(err)
	}
	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrTransInv is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.TransInv(2.0); err == nil {
		t.Error("Expected an error when performing TransInv on a differing type")
	}
}

/* TransInvR */

func Test_f64s_IncrTransInvR(t *testing.T) {
	a, _, _, b := prepf64sTest()
	incr := f64s{100, 100, 100, 100, 100}

	correct := make(f64s, len(a))
	for i, v := range a {
		correct[i] = b - v + incr[i]
	}

	if err := a.IncrTransInvR(b, incr); err != nil {
		t.Fatal(err)
	}
	for i, v := range incr {
		// for floats we don't bother checking the incorrect stuff
		if v != correct[i] && i != 0 {
			t.Errorf("IncrTransInvR is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.TransInvR(float32(2)); err == nil {
		t.Error("Expected an error when performing TransInvR on a differing type")
	}
}

func Test_f32s_IncrTransInvR(t *testing.T) {
	a, _, _, b := prepf32sTest()
	incr := f32s{100, 100, 100, 100, 100}

	correct := make(f32s, len(a))
	for i, v := range a {
		correct[i] = b - v + incr[i]
	}

	if err := a.IncrTransInvR(b, incr); err != nil {
		t.Fatal(err)
	}
	for i, v := range incr {
		// for floats we don't bother checking the incorrect stuff
		if v != correct[i] && i != 0 {
			t.Errorf("IncrTransInvR is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.TransInvR(2.0); err == nil {
		t.Error("Expected an error when performing TransInvR on a differing type")
	}
}

func Test_ints_IncrTransInvR(t *testing.T) {
	a, _, _, b := prepintsTest()
	incr := ints{100, 100, 100, 100, 100}

	correct := make(ints, len(a))
	for i, v := range a {
		correct[i] = b - v + incr[i]
	}

	if err := a.IncrTransInvR(b, incr); err != nil {
		t.Fatal(err)
	}
	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrTransInvR is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.TransInvR(2.0); err == nil {
		t.Error("Expected an error when performing TransInvR on a differing type")
	}
}

func Test_i64s_IncrTransInvR(t *testing.T) {
	a, _, _, b := prepi64sTest()
	incr := i64s{100, 100, 100, 100, 100}

	correct := make(i64s, len(a))
	for i, v := range a {
		correct[i] = b - v + incr[i]
	}

	if err := a.IncrTransInvR(b, incr); err != nil {
		t.Fatal(err)
	}
	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrTransInvR is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.TransInvR(2.0); err == nil {
		t.Error("Expected an error when performing TransInvR on a differing type")
	}
}

func Test_i32s_IncrTransInvR(t *testing.T) {
	a, _, _, b := prepi32sTest()
	incr := i32s{100, 100, 100, 100, 100}

	correct := make(i32s, len(a))
	for i, v := range a {
		correct[i] = b - v + incr[i]
	}

	if err := a.IncrTransInvR(b, incr); err != nil {
		t.Fatal(err)
	}
	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrTransInvR is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.TransInvR(2.0); err == nil {
		t.Error("Expected an error when performing TransInvR on a differing type")
	}
}

func Test_u8s_IncrTransInvR(t *testing.T) {
	a, _, _, b := prepu8sTest()
	incr := u8s{100, 100, 100, 100, 100}

	correct := make(u8s, len(a))
	for i, v := range a {
		correct[i] = b - v + incr[i]
	}

	if err := a.IncrTransInvR(b, incr); err != nil {
		t.Fatal(err)
	}
	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrTransInvR is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.TransInvR(2.0); err == nil {
		t.Error("Expected an error when performing TransInvR on a differing type")
	}
}

/* Scale */

func Test_f64s_IncrScale(t *testing.T) {
	a, _, _, b := prepf64sTest()
	incr := f64s{100, 100, 100, 100, 100}

	correct := make(f64s, len(a))
	for i, v := range a {
		correct[i] = v*b + incr[i]
	}

	if err := a.IncrScale(b, incr); err != nil {
		t.Fatal(err)
	}
	for i, v := range incr {
		// for floats we don't bother checking the incorrect stuff
		if v != correct[i] && i != 0 {
			t.Errorf("IncrScale is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.Scale(float32(2)); err == nil {
		t.Error("Expected an error when performing Scale on a differing type")
	}
}

func Test_f32s_IncrScale(t *testing.T) {
	a, _, _, b := prepf32sTest()
	incr := f32s{100, 100, 100, 100, 100}

	correct := make(f32s, len(a))
	for i, v := range a {
		correct[i] = v*b + incr[i]
	}

	if err := a.IncrScale(b, incr); err != nil {
		t.Fatal(err)
	}
	for i, v := range incr {
		// for floats we don't bother checking the incorrect stuff
		if v != correct[i] && i != 0 {
			t.Errorf("IncrScale is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.Scale(2.0); err == nil {
		t.Error("Expected an error when performing Scale on a differing type")
	}
}

func Test_ints_IncrScale(t *testing.T) {
	a, _, _, b := prepintsTest()
	incr := ints{100, 100, 100, 100, 100}

	correct := make(ints, len(a))
	for i, v := range a {
		correct[i] = v*b + incr[i]
	}

	if err := a.IncrScale(b, incr); err != nil {
		t.Fatal(err)
	}
	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrScale is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.Scale(2.0); err == nil {
		t.Error("Expected an error when performing Scale on a differing type")
	}
}

func Test_i64s_IncrScale(t *testing.T) {
	a, _, _, b := prepi64sTest()
	incr := i64s{100, 100, 100, 100, 100}

	correct := make(i64s, len(a))
	for i, v := range a {
		correct[i] = v*b + incr[i]
	}

	if err := a.IncrScale(b, incr); err != nil {
		t.Fatal(err)
	}
	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrScale is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.Scale(2.0); err == nil {
		t.Error("Expected an error when performing Scale on a differing type")
	}
}

func Test_i32s_IncrScale(t *testing.T) {
	a, _, _, b := prepi32sTest()
	incr := i32s{100, 100, 100, 100, 100}

	correct := make(i32s, len(a))
	for i, v := range a {
		correct[i] = v*b + incr[i]
	}

	if err := a.IncrScale(b, incr); err != nil {
		t.Fatal(err)
	}
	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrScale is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.Scale(2.0); err == nil {
		t.Error("Expected an error when performing Scale on a differing type")
	}
}

func Test_u8s_IncrScale(t *testing.T) {
	a, _, _, b := prepu8sTest()
	incr := u8s{100, 100, 100, 100, 100}

	correct := make(u8s, len(a))
	for i, v := range a {
		correct[i] = v*b + incr[i]
	}

	if err := a.IncrScale(b, incr); err != nil {
		t.Fatal(err)
	}
	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrScale is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.Scale(2.0); err == nil {
		t.Error("Expected an error when performing Scale on a differing type")
	}
}

/* ScaleInv */

func Test_f64s_IncrScaleInv(t *testing.T) {
	a, _, _, b := prepf64sTest()
	incr := f64s{100, 100, 100, 100, 100}

	correct := make(f64s, len(a))
	for i, v := range a {
		if v == float64(0) {
			correct[i] = 0
			continue
		}

		correct[i] = v/b + incr[i]
	}

	if err := a.IncrScaleInv(b, incr); err != nil {
		t.Fatal(err)
	}
	for i, v := range incr {
		// for floats we don't bother checking the incorrect stuff
		if v != correct[i] && i != 0 {
			t.Errorf("IncrScaleInv is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.ScaleInv(float32(2)); err == nil {
		t.Error("Expected an error when performing ScaleInv on a differing type")
	}
}

func Test_f32s_IncrScaleInv(t *testing.T) {
	a, _, _, b := prepf32sTest()
	incr := f32s{100, 100, 100, 100, 100}

	correct := make(f32s, len(a))
	for i, v := range a {
		if v == float32(0) {
			correct[i] = 0
			continue
		}

		correct[i] = v/b + incr[i]
	}

	if err := a.IncrScaleInv(b, incr); err != nil {
		t.Fatal(err)
	}
	for i, v := range incr {
		// for floats we don't bother checking the incorrect stuff
		if v != correct[i] && i != 0 {
			t.Errorf("IncrScaleInv is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.ScaleInv(2.0); err == nil {
		t.Error("Expected an error when performing ScaleInv on a differing type")
	}
}

func Test_ints_IncrScaleInv(t *testing.T) {
	a, _, _, b := prepintsTest()
	incr := ints{100, 100, 100, 100, 100}

	correct := make(ints, len(a))
	for i, v := range a {
		if v == int(0) {
			correct[i] = 0
			continue
		}

		correct[i] = v/b + incr[i]
	}

	err := a.IncrScaleInv(b, incr)
	if err == nil {
		t.Error("Expected error (division by zero)")
	}
	if _, ok := err.(errorIndices); !ok {
		t.Fatal(err)
	}
	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrScaleInv is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.ScaleInv(2.0); err == nil {
		t.Error("Expected an error when performing ScaleInv on a differing type")
	}
}

func Test_i64s_IncrScaleInv(t *testing.T) {
	a, _, _, b := prepi64sTest()
	incr := i64s{100, 100, 100, 100, 100}

	correct := make(i64s, len(a))
	for i, v := range a {
		if v == int64(0) {
			correct[i] = 0
			continue
		}

		correct[i] = v/b + incr[i]
	}

	err := a.IncrScaleInv(b, incr)
	if err == nil {
		t.Error("Expected error (division by zero)")
	}
	if _, ok := err.(errorIndices); !ok {
		t.Fatal(err)
	}
	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrScaleInv is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.ScaleInv(2.0); err == nil {
		t.Error("Expected an error when performing ScaleInv on a differing type")
	}
}

func Test_i32s_IncrScaleInv(t *testing.T) {
	a, _, _, b := prepi32sTest()
	incr := i32s{100, 100, 100, 100, 100}

	correct := make(i32s, len(a))
	for i, v := range a {
		if v == int32(0) {
			correct[i] = 0
			continue
		}

		correct[i] = v/b + incr[i]
	}

	err := a.IncrScaleInv(b, incr)
	if err == nil {
		t.Error("Expected error (division by zero)")
	}
	if _, ok := err.(errorIndices); !ok {
		t.Fatal(err)
	}
	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrScaleInv is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.ScaleInv(2.0); err == nil {
		t.Error("Expected an error when performing ScaleInv on a differing type")
	}
}

func Test_u8s_IncrScaleInv(t *testing.T) {
	a, _, _, b := prepu8sTest()
	incr := u8s{100, 100, 100, 100, 100}

	correct := make(u8s, len(a))
	for i, v := range a {
		if v == byte(0) {
			correct[i] = 0
			continue
		}

		correct[i] = v/b + incr[i]
	}

	err := a.IncrScaleInv(b, incr)
	if err == nil {
		t.Error("Expected error (division by zero)")
	}
	if _, ok := err.(errorIndices); !ok {
		t.Fatal(err)
	}
	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrScaleInv is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.ScaleInv(2.0); err == nil {
		t.Error("Expected an error when performing ScaleInv on a differing type")
	}
}

/* ScaleInvR */

func Test_f64s_IncrScaleInvR(t *testing.T) {
	a, _, _, b := prepf64sTest()
	incr := f64s{100, 100, 100, 100, 100}

	correct := make(f64s, len(a))
	for i, v := range a {
		if v == float64(0) {
			correct[i] = 0
			continue
		}

		correct[i] = b/v + incr[i]
	}

	if err := a.IncrScaleInvR(b, incr); err != nil {
		t.Fatal(err)
	}
	for i, v := range incr {
		// for floats we don't bother checking the incorrect stuff
		if v != correct[i] && i != 0 {
			t.Errorf("IncrScaleInvR is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.ScaleInvR(float32(2)); err == nil {
		t.Error("Expected an error when performing ScaleInvR on a differing type")
	}
}

func Test_f32s_IncrScaleInvR(t *testing.T) {
	a, _, _, b := prepf32sTest()
	incr := f32s{100, 100, 100, 100, 100}

	correct := make(f32s, len(a))
	for i, v := range a {
		if v == float32(0) {
			correct[i] = 0
			continue
		}

		correct[i] = b/v + incr[i]
	}

	if err := a.IncrScaleInvR(b, incr); err != nil {
		t.Fatal(err)
	}
	for i, v := range incr {
		// for floats we don't bother checking the incorrect stuff
		if v != correct[i] && i != 0 {
			t.Errorf("IncrScaleInvR is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.ScaleInvR(2.0); err == nil {
		t.Error("Expected an error when performing ScaleInvR on a differing type")
	}
}

func Test_ints_IncrScaleInvR(t *testing.T) {
	a, _, _, b := prepintsTest()
	incr := ints{100, 100, 100, 100, 100}

	correct := make(ints, len(a))
	for i, v := range a {
		if v == int(0) {
			correct[i] = 0
			continue
		}

		correct[i] = b/v + incr[i]
	}

	err := a.IncrScaleInvR(b, incr)
	if err == nil {
		t.Error("Expected error (division by zero)")
	}
	if _, ok := err.(errorIndices); !ok {
		t.Fatal(err)
	}
	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrScaleInvR is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.ScaleInvR(2.0); err == nil {
		t.Error("Expected an error when performing ScaleInvR on a differing type")
	}
}

func Test_i64s_IncrScaleInvR(t *testing.T) {
	a, _, _, b := prepi64sTest()
	incr := i64s{100, 100, 100, 100, 100}

	correct := make(i64s, len(a))
	for i, v := range a {
		if v == int64(0) {
			correct[i] = 0
			continue
		}

		correct[i] = b/v + incr[i]
	}

	err := a.IncrScaleInvR(b, incr)
	if err == nil {
		t.Error("Expected error (division by zero)")
	}
	if _, ok := err.(errorIndices); !ok {
		t.Fatal(err)
	}
	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrScaleInvR is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.ScaleInvR(2.0); err == nil {
		t.Error("Expected an error when performing ScaleInvR on a differing type")
	}
}

func Test_i32s_IncrScaleInvR(t *testing.T) {
	a, _, _, b := prepi32sTest()
	incr := i32s{100, 100, 100, 100, 100}

	correct := make(i32s, len(a))
	for i, v := range a {
		if v == int32(0) {
			correct[i] = 0
			continue
		}

		correct[i] = b/v + incr[i]
	}

	err := a.IncrScaleInvR(b, incr)
	if err == nil {
		t.Error("Expected error (division by zero)")
	}
	if _, ok := err.(errorIndices); !ok {
		t.Fatal(err)
	}
	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrScaleInvR is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.ScaleInvR(2.0); err == nil {
		t.Error("Expected an error when performing ScaleInvR on a differing type")
	}
}

func Test_u8s_IncrScaleInvR(t *testing.T) {
	a, _, _, b := prepu8sTest()
	incr := u8s{100, 100, 100, 100, 100}

	correct := make(u8s, len(a))
	for i, v := range a {
		if v == byte(0) {
			correct[i] = 0
			continue
		}

		correct[i] = b/v + incr[i]
	}

	err := a.IncrScaleInvR(b, incr)
	if err == nil {
		t.Error("Expected error (division by zero)")
	}
	if _, ok := err.(errorIndices); !ok {
		t.Fatal(err)
	}
	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrScaleInvR is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.ScaleInvR(2.0); err == nil {
		t.Error("Expected an error when performing ScaleInvR on a differing type")
	}
}

/* PowOf */

func Test_f64s_IncrPowOf(t *testing.T) {
	a, _, _, b := prepf64sTest()
	incr := f64s{100, 100, 100, 100, 100}

	correct := make(f64s, len(a))
	for i, v := range a {
		correct[i] = float64(math.Pow(float64(v), float64(b))) + incr[i]
	}

	if err := a.IncrPowOf(b, incr); err != nil {
		t.Fatal(err)
	}
	for i, v := range incr {
		// for floats we don't bother checking the incorrect stuff
		if v != correct[i] && i != 0 {
			t.Errorf("IncrPowOf is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.PowOf(float32(2)); err == nil {
		t.Error("Expected an error when performing PowOf on a differing type")
	}
}

func Test_f32s_IncrPowOf(t *testing.T) {
	a, _, _, b := prepf32sTest()
	incr := f32s{100, 100, 100, 100, 100}

	correct := make(f32s, len(a))
	for i, v := range a {
		correct[i] = float32(math.Pow(float64(v), float64(b))) + incr[i]
	}

	if err := a.IncrPowOf(b, incr); err != nil {
		t.Fatal(err)
	}
	for i, v := range incr {
		// for floats we don't bother checking the incorrect stuff
		if v != correct[i] && i != 0 {
			t.Errorf("IncrPowOf is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.PowOf(2.0); err == nil {
		t.Error("Expected an error when performing PowOf on a differing type")
	}
}

func Test_ints_IncrPowOf(t *testing.T) {
	a, _, _, b := prepintsTest()
	incr := ints{100, 100, 100, 100, 100}

	correct := make(ints, len(a))
	for i, v := range a {
		correct[i] = int(math.Pow(float64(v), float64(b))) + incr[i]
	}

	if err := a.IncrPowOf(b, incr); err != nil {
		t.Fatal(err)
	}
	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrPowOf is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.PowOf(2.0); err == nil {
		t.Error("Expected an error when performing PowOf on a differing type")
	}
}

func Test_i64s_IncrPowOf(t *testing.T) {
	a, _, _, b := prepi64sTest()
	incr := i64s{100, 100, 100, 100, 100}

	correct := make(i64s, len(a))
	for i, v := range a {
		correct[i] = int64(math.Pow(float64(v), float64(b))) + incr[i]
	}

	if err := a.IncrPowOf(b, incr); err != nil {
		t.Fatal(err)
	}
	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrPowOf is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.PowOf(2.0); err == nil {
		t.Error("Expected an error when performing PowOf on a differing type")
	}
}

func Test_i32s_IncrPowOf(t *testing.T) {
	a, _, _, b := prepi32sTest()
	incr := i32s{100, 100, 100, 100, 100}

	correct := make(i32s, len(a))
	for i, v := range a {
		correct[i] = int32(math.Pow(float64(v), float64(b))) + incr[i]
	}

	if err := a.IncrPowOf(b, incr); err != nil {
		t.Fatal(err)
	}
	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrPowOf is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.PowOf(2.0); err == nil {
		t.Error("Expected an error when performing PowOf on a differing type")
	}
}

func Test_u8s_IncrPowOf(t *testing.T) {
	a, _, _, b := prepu8sTest()
	incr := u8s{100, 100, 100, 100, 100}

	correct := make(u8s, len(a))
	for i, v := range a {
		correct[i] = byte(math.Pow(float64(v), float64(b))) + incr[i]
	}

	if err := a.IncrPowOf(b, incr); err != nil {
		t.Fatal(err)
	}
	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrPowOf is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.PowOf(2.0); err == nil {
		t.Error("Expected an error when performing PowOf on a differing type")
	}
}

/* PowOfR */

func Test_f64s_IncrPowOfR(t *testing.T) {
	a, _, _, b := prepf64sTest()
	incr := f64s{100, 100, 100, 100, 100}

	correct := make(f64s, len(a))
	for i, v := range a {
		correct[i] = float64(math.Pow(float64(b), float64(v))) + incr[i]
	}

	if err := a.IncrPowOfR(b, incr); err != nil {
		t.Fatal(err)
	}
	for i, v := range incr {
		// for floats we don't bother checking the incorrect stuff
		if v != correct[i] && i != 0 {
			t.Errorf("IncrPowOfR is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.PowOfR(float32(2)); err == nil {
		t.Error("Expected an error when performing PowOfR on a differing type")
	}
}

func Test_f32s_IncrPowOfR(t *testing.T) {
	a, _, _, b := prepf32sTest()
	incr := f32s{100, 100, 100, 100, 100}

	correct := make(f32s, len(a))
	for i, v := range a {
		correct[i] = float32(math.Pow(float64(b), float64(v))) + incr[i]
	}

	if err := a.IncrPowOfR(b, incr); err != nil {
		t.Fatal(err)
	}
	for i, v := range incr {
		// for floats we don't bother checking the incorrect stuff
		if v != correct[i] && i != 0 {
			t.Errorf("IncrPowOfR is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.PowOfR(2.0); err == nil {
		t.Error("Expected an error when performing PowOfR on a differing type")
	}
}

func Test_ints_IncrPowOfR(t *testing.T) {
	a, _, _, b := prepintsTest()
	incr := ints{100, 100, 100, 100, 100}

	correct := make(ints, len(a))
	for i, v := range a {
		correct[i] = int(math.Pow(float64(b), float64(v))) + incr[i]
	}

	if err := a.IncrPowOfR(b, incr); err != nil {
		t.Fatal(err)
	}
	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrPowOfR is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.PowOfR(2.0); err == nil {
		t.Error("Expected an error when performing PowOfR on a differing type")
	}
}

func Test_i64s_IncrPowOfR(t *testing.T) {
	a, _, _, b := prepi64sTest()
	incr := i64s{100, 100, 100, 100, 100}

	correct := make(i64s, len(a))
	for i, v := range a {
		correct[i] = int64(math.Pow(float64(b), float64(v))) + incr[i]
	}

	if err := a.IncrPowOfR(b, incr); err != nil {
		t.Fatal(err)
	}
	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrPowOfR is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.PowOfR(2.0); err == nil {
		t.Error("Expected an error when performing PowOfR on a differing type")
	}
}

func Test_i32s_IncrPowOfR(t *testing.T) {
	a, _, _, b := prepi32sTest()
	incr := i32s{100, 100, 100, 100, 100}

	correct := make(i32s, len(a))
	for i, v := range a {
		correct[i] = int32(math.Pow(float64(b), float64(v))) + incr[i]
	}

	if err := a.IncrPowOfR(b, incr); err != nil {
		t.Fatal(err)
	}
	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrPowOfR is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.PowOfR(2.0); err == nil {
		t.Error("Expected an error when performing PowOfR on a differing type")
	}
}

func Test_u8s_IncrPowOfR(t *testing.T) {
	a, _, _, b := prepu8sTest()
	incr := u8s{100, 100, 100, 100, 100}

	correct := make(u8s, len(a))
	for i, v := range a {
		correct[i] = byte(math.Pow(float64(b), float64(v))) + incr[i]
	}

	if err := a.IncrPowOfR(b, incr); err != nil {
		t.Fatal(err)
	}
	for i, v := range incr {
		if v != correct[i] {
			t.Errorf("IncrPowOfR is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.PowOfR(2.0); err == nil {
		t.Error("Expected an error when performing PowOfR on a differing type")
	}
}
