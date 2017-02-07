package tensor

import "testing"

/*
GENERATED FILE. DO NOT EDIT
*/

/* ElEq */

func Test_f64s_ElEq(t *testing.T) {
	var a, b f64s
	var c f64sDummy
	var res Array
	var err error

	a = f64s{0, 1, 2, 3, 4}
	b = f64s{1, 2, 2, 1, 100}
	c = f64sDummy{1, 2, 2, 1, 100}

	correct := make(bs, len(a))
	correctSame := make(f64s, len(a))
	for i, v := range a {
		correct[i] = v == b[i]

		if v == b[i] {
			correctSame[i] = float64(1)
		} else {
			correctSame[i] = float64(0)
		}
	}

	// return bools
	if res, err = a.ElEq(b, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("ElEq is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = f64s{0, 1, 2, 3, 4}
	if res, err = a.ElEq(b, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(f64s) {
		if v != correctSame[i] {
			t.Errorf("ElEq is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// now for compatible types

	// return bools
	a = f64s{0, 1, 2, 3, 4}
	if res, err = a.ElEq(c, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("ElEq is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = f64s{0, 1, 2, 3, 4}
	if res, err = a.ElEq(c, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(f64s) {
		if v != correctSame[i] {
			t.Errorf("ElEq is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// stupids # 1 : differing length
	if _, err := a.ElEq(b[:3], true); err == nil {
		t.Errorf("Expected an error when performing ElEq on differing lengths")
	}

	// stupids #2 : different types (which are checked before lengths)
	if _, err := a.ElEq(f32s{}, true); err == nil {
		t.Errorf("Expected an error when performing ElEq on a non-compatible type")
	}
}

func Test_f32s_ElEq(t *testing.T) {
	var a, b f32s
	var c f32sDummy
	var res Array
	var err error

	a = f32s{0, 1, 2, 3, 4}
	b = f32s{1, 2, 2, 1, 100}
	c = f32sDummy{1, 2, 2, 1, 100}

	correct := make(bs, len(a))
	correctSame := make(f32s, len(a))
	for i, v := range a {
		correct[i] = v == b[i]

		if v == b[i] {
			correctSame[i] = float32(1)
		} else {
			correctSame[i] = float32(0)
		}
	}

	// return bools
	if res, err = a.ElEq(b, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("ElEq is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = f32s{0, 1, 2, 3, 4}
	if res, err = a.ElEq(b, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(f32s) {
		if v != correctSame[i] {
			t.Errorf("ElEq is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// now for compatible types

	// return bools
	a = f32s{0, 1, 2, 3, 4}
	if res, err = a.ElEq(c, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("ElEq is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = f32s{0, 1, 2, 3, 4}
	if res, err = a.ElEq(c, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(f32s) {
		if v != correctSame[i] {
			t.Errorf("ElEq is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// stupids # 1 : differing length
	if _, err := a.ElEq(b[:3], true); err == nil {
		t.Errorf("Expected an error when performing ElEq on differing lengths")
	}

	// stupids #2 : different types (which are checked before lengths)
	if _, err := a.ElEq(f64s{}, true); err == nil {
		t.Errorf("Expected an error when performing ElEq on a non-compatible type")
	}
}

func Test_ints_ElEq(t *testing.T) {
	var a, b ints
	var c intsDummy
	var res Array
	var err error

	a = ints{0, 1, 2, 3, 4}
	b = ints{1, 2, 2, 1, 100}
	c = intsDummy{1, 2, 2, 1, 100}

	correct := make(bs, len(a))
	correctSame := make(ints, len(a))
	for i, v := range a {
		correct[i] = v == b[i]

		if v == b[i] {
			correctSame[i] = int(1)
		} else {
			correctSame[i] = int(0)
		}
	}

	// return bools
	if res, err = a.ElEq(b, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("ElEq is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = ints{0, 1, 2, 3, 4}
	if res, err = a.ElEq(b, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(ints) {
		if v != correctSame[i] {
			t.Errorf("ElEq is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// now for compatible types

	// return bools
	a = ints{0, 1, 2, 3, 4}
	if res, err = a.ElEq(c, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("ElEq is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = ints{0, 1, 2, 3, 4}
	if res, err = a.ElEq(c, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(ints) {
		if v != correctSame[i] {
			t.Errorf("ElEq is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// stupids # 1 : differing length
	if _, err := a.ElEq(b[:3], true); err == nil {
		t.Errorf("Expected an error when performing ElEq on differing lengths")
	}

	// stupids #2 : different types (which are checked before lengths)
	if _, err := a.ElEq(f64s{}, true); err == nil {
		t.Errorf("Expected an error when performing ElEq on a non-compatible type")
	}
}

func Test_i64s_ElEq(t *testing.T) {
	var a, b i64s
	var c i64sDummy
	var res Array
	var err error

	a = i64s{0, 1, 2, 3, 4}
	b = i64s{1, 2, 2, 1, 100}
	c = i64sDummy{1, 2, 2, 1, 100}

	correct := make(bs, len(a))
	correctSame := make(i64s, len(a))
	for i, v := range a {
		correct[i] = v == b[i]

		if v == b[i] {
			correctSame[i] = int64(1)
		} else {
			correctSame[i] = int64(0)
		}
	}

	// return bools
	if res, err = a.ElEq(b, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("ElEq is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = i64s{0, 1, 2, 3, 4}
	if res, err = a.ElEq(b, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(i64s) {
		if v != correctSame[i] {
			t.Errorf("ElEq is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// now for compatible types

	// return bools
	a = i64s{0, 1, 2, 3, 4}
	if res, err = a.ElEq(c, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("ElEq is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = i64s{0, 1, 2, 3, 4}
	if res, err = a.ElEq(c, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(i64s) {
		if v != correctSame[i] {
			t.Errorf("ElEq is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// stupids # 1 : differing length
	if _, err := a.ElEq(b[:3], true); err == nil {
		t.Errorf("Expected an error when performing ElEq on differing lengths")
	}

	// stupids #2 : different types (which are checked before lengths)
	if _, err := a.ElEq(f64s{}, true); err == nil {
		t.Errorf("Expected an error when performing ElEq on a non-compatible type")
	}
}

func Test_i32s_ElEq(t *testing.T) {
	var a, b i32s
	var c i32sDummy
	var res Array
	var err error

	a = i32s{0, 1, 2, 3, 4}
	b = i32s{1, 2, 2, 1, 100}
	c = i32sDummy{1, 2, 2, 1, 100}

	correct := make(bs, len(a))
	correctSame := make(i32s, len(a))
	for i, v := range a {
		correct[i] = v == b[i]

		if v == b[i] {
			correctSame[i] = int32(1)
		} else {
			correctSame[i] = int32(0)
		}
	}

	// return bools
	if res, err = a.ElEq(b, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("ElEq is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = i32s{0, 1, 2, 3, 4}
	if res, err = a.ElEq(b, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(i32s) {
		if v != correctSame[i] {
			t.Errorf("ElEq is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// now for compatible types

	// return bools
	a = i32s{0, 1, 2, 3, 4}
	if res, err = a.ElEq(c, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("ElEq is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = i32s{0, 1, 2, 3, 4}
	if res, err = a.ElEq(c, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(i32s) {
		if v != correctSame[i] {
			t.Errorf("ElEq is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// stupids # 1 : differing length
	if _, err := a.ElEq(b[:3], true); err == nil {
		t.Errorf("Expected an error when performing ElEq on differing lengths")
	}

	// stupids #2 : different types (which are checked before lengths)
	if _, err := a.ElEq(f64s{}, true); err == nil {
		t.Errorf("Expected an error when performing ElEq on a non-compatible type")
	}
}

func Test_u8s_ElEq(t *testing.T) {
	var a, b u8s
	var c u8sDummy
	var res Array
	var err error

	a = u8s{0, 1, 2, 3, 4}
	b = u8s{1, 2, 2, 1, 100}
	c = u8sDummy{1, 2, 2, 1, 100}

	correct := make(bs, len(a))
	correctSame := make(u8s, len(a))
	for i, v := range a {
		correct[i] = v == b[i]

		if v == b[i] {
			correctSame[i] = byte(1)
		} else {
			correctSame[i] = byte(0)
		}
	}

	// return bools
	if res, err = a.ElEq(b, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("ElEq is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = u8s{0, 1, 2, 3, 4}
	if res, err = a.ElEq(b, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(u8s) {
		if v != correctSame[i] {
			t.Errorf("ElEq is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// now for compatible types

	// return bools
	a = u8s{0, 1, 2, 3, 4}
	if res, err = a.ElEq(c, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("ElEq is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = u8s{0, 1, 2, 3, 4}
	if res, err = a.ElEq(c, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(u8s) {
		if v != correctSame[i] {
			t.Errorf("ElEq is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// stupids # 1 : differing length
	if _, err := a.ElEq(b[:3], true); err == nil {
		t.Errorf("Expected an error when performing ElEq on differing lengths")
	}

	// stupids #2 : different types (which are checked before lengths)
	if _, err := a.ElEq(f64s{}, true); err == nil {
		t.Errorf("Expected an error when performing ElEq on a non-compatible type")
	}
}

func Test_bs_ElEq(t *testing.T) {
	var a, b bs
	var c bsDummy
	var res Array
	var err error

	a = bs{true, false, true, false, true}
	b = bs{true, true, true, false, false}
	c = bsDummy{true, true, true, false, false}

	correct := make(bs, len(a))
	correctSame := make(bs, len(a))
	for i, v := range a {
		correct[i] = v == b[i]

		if v == b[i] {
			correctSame[i] = true
		} else {
			correctSame[i] = false
		}
	}

	// return bools
	if res, err = a.ElEq(b, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("ElEq is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = bs{true, false, true, false, true}
	if res, err = a.ElEq(b, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correctSame[i] {
			t.Errorf("ElEq is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// now for compatible types

	// return bools
	a = bs{true, false, true, false, true}
	if res, err = a.ElEq(c, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("ElEq is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = bs{true, false, true, false, true}
	if res, err = a.ElEq(c, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correctSame[i] {
			t.Errorf("ElEq is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// stupids # 1 : differing length
	if _, err := a.ElEq(b[:3], true); err == nil {
		t.Errorf("Expected an error when performing ElEq on differing lengths")
	}

	// stupids #2 : different types (which are checked before lengths)
	if _, err := a.ElEq(f64s{}, true); err == nil {
		t.Errorf("Expected an error when performing ElEq on a non-compatible type")
	}
}

/* Gt */

func Test_f64s_Gt(t *testing.T) {
	var a, b f64s
	var c f64sDummy
	var res Array
	var err error

	a = f64s{0, 1, 2, 3, 4}
	b = f64s{1, 2, 2, 1, 100}
	c = f64sDummy{1, 2, 2, 1, 100}

	correct := make(bs, len(a))
	correctSame := make(f64s, len(a))
	for i, v := range a {
		correct[i] = v > b[i]

		if v > b[i] {
			correctSame[i] = float64(1)
		} else {
			correctSame[i] = float64(0)
		}
	}

	// return bools
	if res, err = a.Gt(b, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("Gt is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = f64s{0, 1, 2, 3, 4}
	if res, err = a.Gt(b, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(f64s) {
		if v != correctSame[i] {
			t.Errorf("Gt is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// now for compatible types

	// return bools
	a = f64s{0, 1, 2, 3, 4}
	if res, err = a.Gt(c, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("Gt is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = f64s{0, 1, 2, 3, 4}
	if res, err = a.Gt(c, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(f64s) {
		if v != correctSame[i] {
			t.Errorf("Gt is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// stupids # 1 : differing length
	if _, err := a.Gt(b[:3], true); err == nil {
		t.Errorf("Expected an error when performing Gt on differing lengths")
	}

	// stupids #2 : different types (which are checked before lengths)
	if _, err := a.Gt(f32s{}, true); err == nil {
		t.Errorf("Expected an error when performing Gt on a non-compatible type")
	}
}

func Test_f32s_Gt(t *testing.T) {
	var a, b f32s
	var c f32sDummy
	var res Array
	var err error

	a = f32s{0, 1, 2, 3, 4}
	b = f32s{1, 2, 2, 1, 100}
	c = f32sDummy{1, 2, 2, 1, 100}

	correct := make(bs, len(a))
	correctSame := make(f32s, len(a))
	for i, v := range a {
		correct[i] = v > b[i]

		if v > b[i] {
			correctSame[i] = float32(1)
		} else {
			correctSame[i] = float32(0)
		}
	}

	// return bools
	if res, err = a.Gt(b, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("Gt is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = f32s{0, 1, 2, 3, 4}
	if res, err = a.Gt(b, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(f32s) {
		if v != correctSame[i] {
			t.Errorf("Gt is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// now for compatible types

	// return bools
	a = f32s{0, 1, 2, 3, 4}
	if res, err = a.Gt(c, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("Gt is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = f32s{0, 1, 2, 3, 4}
	if res, err = a.Gt(c, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(f32s) {
		if v != correctSame[i] {
			t.Errorf("Gt is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// stupids # 1 : differing length
	if _, err := a.Gt(b[:3], true); err == nil {
		t.Errorf("Expected an error when performing Gt on differing lengths")
	}

	// stupids #2 : different types (which are checked before lengths)
	if _, err := a.Gt(f64s{}, true); err == nil {
		t.Errorf("Expected an error when performing Gt on a non-compatible type")
	}
}

func Test_ints_Gt(t *testing.T) {
	var a, b ints
	var c intsDummy
	var res Array
	var err error

	a = ints{0, 1, 2, 3, 4}
	b = ints{1, 2, 2, 1, 100}
	c = intsDummy{1, 2, 2, 1, 100}

	correct := make(bs, len(a))
	correctSame := make(ints, len(a))
	for i, v := range a {
		correct[i] = v > b[i]

		if v > b[i] {
			correctSame[i] = int(1)
		} else {
			correctSame[i] = int(0)
		}
	}

	// return bools
	if res, err = a.Gt(b, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("Gt is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = ints{0, 1, 2, 3, 4}
	if res, err = a.Gt(b, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(ints) {
		if v != correctSame[i] {
			t.Errorf("Gt is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// now for compatible types

	// return bools
	a = ints{0, 1, 2, 3, 4}
	if res, err = a.Gt(c, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("Gt is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = ints{0, 1, 2, 3, 4}
	if res, err = a.Gt(c, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(ints) {
		if v != correctSame[i] {
			t.Errorf("Gt is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// stupids # 1 : differing length
	if _, err := a.Gt(b[:3], true); err == nil {
		t.Errorf("Expected an error when performing Gt on differing lengths")
	}

	// stupids #2 : different types (which are checked before lengths)
	if _, err := a.Gt(f64s{}, true); err == nil {
		t.Errorf("Expected an error when performing Gt on a non-compatible type")
	}
}

func Test_i64s_Gt(t *testing.T) {
	var a, b i64s
	var c i64sDummy
	var res Array
	var err error

	a = i64s{0, 1, 2, 3, 4}
	b = i64s{1, 2, 2, 1, 100}
	c = i64sDummy{1, 2, 2, 1, 100}

	correct := make(bs, len(a))
	correctSame := make(i64s, len(a))
	for i, v := range a {
		correct[i] = v > b[i]

		if v > b[i] {
			correctSame[i] = int64(1)
		} else {
			correctSame[i] = int64(0)
		}
	}

	// return bools
	if res, err = a.Gt(b, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("Gt is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = i64s{0, 1, 2, 3, 4}
	if res, err = a.Gt(b, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(i64s) {
		if v != correctSame[i] {
			t.Errorf("Gt is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// now for compatible types

	// return bools
	a = i64s{0, 1, 2, 3, 4}
	if res, err = a.Gt(c, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("Gt is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = i64s{0, 1, 2, 3, 4}
	if res, err = a.Gt(c, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(i64s) {
		if v != correctSame[i] {
			t.Errorf("Gt is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// stupids # 1 : differing length
	if _, err := a.Gt(b[:3], true); err == nil {
		t.Errorf("Expected an error when performing Gt on differing lengths")
	}

	// stupids #2 : different types (which are checked before lengths)
	if _, err := a.Gt(f64s{}, true); err == nil {
		t.Errorf("Expected an error when performing Gt on a non-compatible type")
	}
}

func Test_i32s_Gt(t *testing.T) {
	var a, b i32s
	var c i32sDummy
	var res Array
	var err error

	a = i32s{0, 1, 2, 3, 4}
	b = i32s{1, 2, 2, 1, 100}
	c = i32sDummy{1, 2, 2, 1, 100}

	correct := make(bs, len(a))
	correctSame := make(i32s, len(a))
	for i, v := range a {
		correct[i] = v > b[i]

		if v > b[i] {
			correctSame[i] = int32(1)
		} else {
			correctSame[i] = int32(0)
		}
	}

	// return bools
	if res, err = a.Gt(b, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("Gt is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = i32s{0, 1, 2, 3, 4}
	if res, err = a.Gt(b, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(i32s) {
		if v != correctSame[i] {
			t.Errorf("Gt is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// now for compatible types

	// return bools
	a = i32s{0, 1, 2, 3, 4}
	if res, err = a.Gt(c, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("Gt is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = i32s{0, 1, 2, 3, 4}
	if res, err = a.Gt(c, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(i32s) {
		if v != correctSame[i] {
			t.Errorf("Gt is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// stupids # 1 : differing length
	if _, err := a.Gt(b[:3], true); err == nil {
		t.Errorf("Expected an error when performing Gt on differing lengths")
	}

	// stupids #2 : different types (which are checked before lengths)
	if _, err := a.Gt(f64s{}, true); err == nil {
		t.Errorf("Expected an error when performing Gt on a non-compatible type")
	}
}

func Test_u8s_Gt(t *testing.T) {
	var a, b u8s
	var c u8sDummy
	var res Array
	var err error

	a = u8s{0, 1, 2, 3, 4}
	b = u8s{1, 2, 2, 1, 100}
	c = u8sDummy{1, 2, 2, 1, 100}

	correct := make(bs, len(a))
	correctSame := make(u8s, len(a))
	for i, v := range a {
		correct[i] = v > b[i]

		if v > b[i] {
			correctSame[i] = byte(1)
		} else {
			correctSame[i] = byte(0)
		}
	}

	// return bools
	if res, err = a.Gt(b, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("Gt is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = u8s{0, 1, 2, 3, 4}
	if res, err = a.Gt(b, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(u8s) {
		if v != correctSame[i] {
			t.Errorf("Gt is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// now for compatible types

	// return bools
	a = u8s{0, 1, 2, 3, 4}
	if res, err = a.Gt(c, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("Gt is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = u8s{0, 1, 2, 3, 4}
	if res, err = a.Gt(c, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(u8s) {
		if v != correctSame[i] {
			t.Errorf("Gt is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// stupids # 1 : differing length
	if _, err := a.Gt(b[:3], true); err == nil {
		t.Errorf("Expected an error when performing Gt on differing lengths")
	}

	// stupids #2 : different types (which are checked before lengths)
	if _, err := a.Gt(f64s{}, true); err == nil {
		t.Errorf("Expected an error when performing Gt on a non-compatible type")
	}
}

/* Gte */

func Test_f64s_Gte(t *testing.T) {
	var a, b f64s
	var c f64sDummy
	var res Array
	var err error

	a = f64s{0, 1, 2, 3, 4}
	b = f64s{1, 2, 2, 1, 100}
	c = f64sDummy{1, 2, 2, 1, 100}

	correct := make(bs, len(a))
	correctSame := make(f64s, len(a))
	for i, v := range a {
		correct[i] = v >= b[i]

		if v >= b[i] {
			correctSame[i] = float64(1)
		} else {
			correctSame[i] = float64(0)
		}
	}

	// return bools
	if res, err = a.Gte(b, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("Gte is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = f64s{0, 1, 2, 3, 4}
	if res, err = a.Gte(b, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(f64s) {
		if v != correctSame[i] {
			t.Errorf("Gte is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// now for compatible types

	// return bools
	a = f64s{0, 1, 2, 3, 4}
	if res, err = a.Gte(c, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("Gte is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = f64s{0, 1, 2, 3, 4}
	if res, err = a.Gte(c, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(f64s) {
		if v != correctSame[i] {
			t.Errorf("Gte is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// stupids # 1 : differing length
	if _, err := a.Gte(b[:3], true); err == nil {
		t.Errorf("Expected an error when performing Gte on differing lengths")
	}

	// stupids #2 : different types (which are checked before lengths)
	if _, err := a.Gte(f32s{}, true); err == nil {
		t.Errorf("Expected an error when performing Gte on a non-compatible type")
	}
}

func Test_f32s_Gte(t *testing.T) {
	var a, b f32s
	var c f32sDummy
	var res Array
	var err error

	a = f32s{0, 1, 2, 3, 4}
	b = f32s{1, 2, 2, 1, 100}
	c = f32sDummy{1, 2, 2, 1, 100}

	correct := make(bs, len(a))
	correctSame := make(f32s, len(a))
	for i, v := range a {
		correct[i] = v >= b[i]

		if v >= b[i] {
			correctSame[i] = float32(1)
		} else {
			correctSame[i] = float32(0)
		}
	}

	// return bools
	if res, err = a.Gte(b, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("Gte is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = f32s{0, 1, 2, 3, 4}
	if res, err = a.Gte(b, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(f32s) {
		if v != correctSame[i] {
			t.Errorf("Gte is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// now for compatible types

	// return bools
	a = f32s{0, 1, 2, 3, 4}
	if res, err = a.Gte(c, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("Gte is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = f32s{0, 1, 2, 3, 4}
	if res, err = a.Gte(c, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(f32s) {
		if v != correctSame[i] {
			t.Errorf("Gte is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// stupids # 1 : differing length
	if _, err := a.Gte(b[:3], true); err == nil {
		t.Errorf("Expected an error when performing Gte on differing lengths")
	}

	// stupids #2 : different types (which are checked before lengths)
	if _, err := a.Gte(f64s{}, true); err == nil {
		t.Errorf("Expected an error when performing Gte on a non-compatible type")
	}
}

func Test_ints_Gte(t *testing.T) {
	var a, b ints
	var c intsDummy
	var res Array
	var err error

	a = ints{0, 1, 2, 3, 4}
	b = ints{1, 2, 2, 1, 100}
	c = intsDummy{1, 2, 2, 1, 100}

	correct := make(bs, len(a))
	correctSame := make(ints, len(a))
	for i, v := range a {
		correct[i] = v >= b[i]

		if v >= b[i] {
			correctSame[i] = int(1)
		} else {
			correctSame[i] = int(0)
		}
	}

	// return bools
	if res, err = a.Gte(b, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("Gte is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = ints{0, 1, 2, 3, 4}
	if res, err = a.Gte(b, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(ints) {
		if v != correctSame[i] {
			t.Errorf("Gte is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// now for compatible types

	// return bools
	a = ints{0, 1, 2, 3, 4}
	if res, err = a.Gte(c, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("Gte is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = ints{0, 1, 2, 3, 4}
	if res, err = a.Gte(c, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(ints) {
		if v != correctSame[i] {
			t.Errorf("Gte is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// stupids # 1 : differing length
	if _, err := a.Gte(b[:3], true); err == nil {
		t.Errorf("Expected an error when performing Gte on differing lengths")
	}

	// stupids #2 : different types (which are checked before lengths)
	if _, err := a.Gte(f64s{}, true); err == nil {
		t.Errorf("Expected an error when performing Gte on a non-compatible type")
	}
}

func Test_i64s_Gte(t *testing.T) {
	var a, b i64s
	var c i64sDummy
	var res Array
	var err error

	a = i64s{0, 1, 2, 3, 4}
	b = i64s{1, 2, 2, 1, 100}
	c = i64sDummy{1, 2, 2, 1, 100}

	correct := make(bs, len(a))
	correctSame := make(i64s, len(a))
	for i, v := range a {
		correct[i] = v >= b[i]

		if v >= b[i] {
			correctSame[i] = int64(1)
		} else {
			correctSame[i] = int64(0)
		}
	}

	// return bools
	if res, err = a.Gte(b, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("Gte is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = i64s{0, 1, 2, 3, 4}
	if res, err = a.Gte(b, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(i64s) {
		if v != correctSame[i] {
			t.Errorf("Gte is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// now for compatible types

	// return bools
	a = i64s{0, 1, 2, 3, 4}
	if res, err = a.Gte(c, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("Gte is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = i64s{0, 1, 2, 3, 4}
	if res, err = a.Gte(c, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(i64s) {
		if v != correctSame[i] {
			t.Errorf("Gte is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// stupids # 1 : differing length
	if _, err := a.Gte(b[:3], true); err == nil {
		t.Errorf("Expected an error when performing Gte on differing lengths")
	}

	// stupids #2 : different types (which are checked before lengths)
	if _, err := a.Gte(f64s{}, true); err == nil {
		t.Errorf("Expected an error when performing Gte on a non-compatible type")
	}
}

func Test_i32s_Gte(t *testing.T) {
	var a, b i32s
	var c i32sDummy
	var res Array
	var err error

	a = i32s{0, 1, 2, 3, 4}
	b = i32s{1, 2, 2, 1, 100}
	c = i32sDummy{1, 2, 2, 1, 100}

	correct := make(bs, len(a))
	correctSame := make(i32s, len(a))
	for i, v := range a {
		correct[i] = v >= b[i]

		if v >= b[i] {
			correctSame[i] = int32(1)
		} else {
			correctSame[i] = int32(0)
		}
	}

	// return bools
	if res, err = a.Gte(b, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("Gte is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = i32s{0, 1, 2, 3, 4}
	if res, err = a.Gte(b, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(i32s) {
		if v != correctSame[i] {
			t.Errorf("Gte is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// now for compatible types

	// return bools
	a = i32s{0, 1, 2, 3, 4}
	if res, err = a.Gte(c, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("Gte is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = i32s{0, 1, 2, 3, 4}
	if res, err = a.Gte(c, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(i32s) {
		if v != correctSame[i] {
			t.Errorf("Gte is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// stupids # 1 : differing length
	if _, err := a.Gte(b[:3], true); err == nil {
		t.Errorf("Expected an error when performing Gte on differing lengths")
	}

	// stupids #2 : different types (which are checked before lengths)
	if _, err := a.Gte(f64s{}, true); err == nil {
		t.Errorf("Expected an error when performing Gte on a non-compatible type")
	}
}

func Test_u8s_Gte(t *testing.T) {
	var a, b u8s
	var c u8sDummy
	var res Array
	var err error

	a = u8s{0, 1, 2, 3, 4}
	b = u8s{1, 2, 2, 1, 100}
	c = u8sDummy{1, 2, 2, 1, 100}

	correct := make(bs, len(a))
	correctSame := make(u8s, len(a))
	for i, v := range a {
		correct[i] = v >= b[i]

		if v >= b[i] {
			correctSame[i] = byte(1)
		} else {
			correctSame[i] = byte(0)
		}
	}

	// return bools
	if res, err = a.Gte(b, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("Gte is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = u8s{0, 1, 2, 3, 4}
	if res, err = a.Gte(b, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(u8s) {
		if v != correctSame[i] {
			t.Errorf("Gte is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// now for compatible types

	// return bools
	a = u8s{0, 1, 2, 3, 4}
	if res, err = a.Gte(c, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("Gte is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = u8s{0, 1, 2, 3, 4}
	if res, err = a.Gte(c, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(u8s) {
		if v != correctSame[i] {
			t.Errorf("Gte is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// stupids # 1 : differing length
	if _, err := a.Gte(b[:3], true); err == nil {
		t.Errorf("Expected an error when performing Gte on differing lengths")
	}

	// stupids #2 : different types (which are checked before lengths)
	if _, err := a.Gte(f64s{}, true); err == nil {
		t.Errorf("Expected an error when performing Gte on a non-compatible type")
	}
}

/* Lt */

func Test_f64s_Lt(t *testing.T) {
	var a, b f64s
	var c f64sDummy
	var res Array
	var err error

	a = f64s{0, 1, 2, 3, 4}
	b = f64s{1, 2, 2, 1, 100}
	c = f64sDummy{1, 2, 2, 1, 100}

	correct := make(bs, len(a))
	correctSame := make(f64s, len(a))
	for i, v := range a {
		correct[i] = v < b[i]

		if v < b[i] {
			correctSame[i] = float64(1)
		} else {
			correctSame[i] = float64(0)
		}
	}

	// return bools
	if res, err = a.Lt(b, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("Lt is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = f64s{0, 1, 2, 3, 4}
	if res, err = a.Lt(b, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(f64s) {
		if v != correctSame[i] {
			t.Errorf("Lt is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// now for compatible types

	// return bools
	a = f64s{0, 1, 2, 3, 4}
	if res, err = a.Lt(c, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("Lt is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = f64s{0, 1, 2, 3, 4}
	if res, err = a.Lt(c, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(f64s) {
		if v != correctSame[i] {
			t.Errorf("Lt is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// stupids # 1 : differing length
	if _, err := a.Lt(b[:3], true); err == nil {
		t.Errorf("Expected an error when performing Lt on differing lengths")
	}

	// stupids #2 : different types (which are checked before lengths)
	if _, err := a.Lt(f32s{}, true); err == nil {
		t.Errorf("Expected an error when performing Lt on a non-compatible type")
	}
}

func Test_f32s_Lt(t *testing.T) {
	var a, b f32s
	var c f32sDummy
	var res Array
	var err error

	a = f32s{0, 1, 2, 3, 4}
	b = f32s{1, 2, 2, 1, 100}
	c = f32sDummy{1, 2, 2, 1, 100}

	correct := make(bs, len(a))
	correctSame := make(f32s, len(a))
	for i, v := range a {
		correct[i] = v < b[i]

		if v < b[i] {
			correctSame[i] = float32(1)
		} else {
			correctSame[i] = float32(0)
		}
	}

	// return bools
	if res, err = a.Lt(b, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("Lt is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = f32s{0, 1, 2, 3, 4}
	if res, err = a.Lt(b, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(f32s) {
		if v != correctSame[i] {
			t.Errorf("Lt is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// now for compatible types

	// return bools
	a = f32s{0, 1, 2, 3, 4}
	if res, err = a.Lt(c, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("Lt is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = f32s{0, 1, 2, 3, 4}
	if res, err = a.Lt(c, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(f32s) {
		if v != correctSame[i] {
			t.Errorf("Lt is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// stupids # 1 : differing length
	if _, err := a.Lt(b[:3], true); err == nil {
		t.Errorf("Expected an error when performing Lt on differing lengths")
	}

	// stupids #2 : different types (which are checked before lengths)
	if _, err := a.Lt(f64s{}, true); err == nil {
		t.Errorf("Expected an error when performing Lt on a non-compatible type")
	}
}

func Test_ints_Lt(t *testing.T) {
	var a, b ints
	var c intsDummy
	var res Array
	var err error

	a = ints{0, 1, 2, 3, 4}
	b = ints{1, 2, 2, 1, 100}
	c = intsDummy{1, 2, 2, 1, 100}

	correct := make(bs, len(a))
	correctSame := make(ints, len(a))
	for i, v := range a {
		correct[i] = v < b[i]

		if v < b[i] {
			correctSame[i] = int(1)
		} else {
			correctSame[i] = int(0)
		}
	}

	// return bools
	if res, err = a.Lt(b, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("Lt is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = ints{0, 1, 2, 3, 4}
	if res, err = a.Lt(b, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(ints) {
		if v != correctSame[i] {
			t.Errorf("Lt is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// now for compatible types

	// return bools
	a = ints{0, 1, 2, 3, 4}
	if res, err = a.Lt(c, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("Lt is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = ints{0, 1, 2, 3, 4}
	if res, err = a.Lt(c, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(ints) {
		if v != correctSame[i] {
			t.Errorf("Lt is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// stupids # 1 : differing length
	if _, err := a.Lt(b[:3], true); err == nil {
		t.Errorf("Expected an error when performing Lt on differing lengths")
	}

	// stupids #2 : different types (which are checked before lengths)
	if _, err := a.Lt(f64s{}, true); err == nil {
		t.Errorf("Expected an error when performing Lt on a non-compatible type")
	}
}

func Test_i64s_Lt(t *testing.T) {
	var a, b i64s
	var c i64sDummy
	var res Array
	var err error

	a = i64s{0, 1, 2, 3, 4}
	b = i64s{1, 2, 2, 1, 100}
	c = i64sDummy{1, 2, 2, 1, 100}

	correct := make(bs, len(a))
	correctSame := make(i64s, len(a))
	for i, v := range a {
		correct[i] = v < b[i]

		if v < b[i] {
			correctSame[i] = int64(1)
		} else {
			correctSame[i] = int64(0)
		}
	}

	// return bools
	if res, err = a.Lt(b, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("Lt is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = i64s{0, 1, 2, 3, 4}
	if res, err = a.Lt(b, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(i64s) {
		if v != correctSame[i] {
			t.Errorf("Lt is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// now for compatible types

	// return bools
	a = i64s{0, 1, 2, 3, 4}
	if res, err = a.Lt(c, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("Lt is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = i64s{0, 1, 2, 3, 4}
	if res, err = a.Lt(c, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(i64s) {
		if v != correctSame[i] {
			t.Errorf("Lt is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// stupids # 1 : differing length
	if _, err := a.Lt(b[:3], true); err == nil {
		t.Errorf("Expected an error when performing Lt on differing lengths")
	}

	// stupids #2 : different types (which are checked before lengths)
	if _, err := a.Lt(f64s{}, true); err == nil {
		t.Errorf("Expected an error when performing Lt on a non-compatible type")
	}
}

func Test_i32s_Lt(t *testing.T) {
	var a, b i32s
	var c i32sDummy
	var res Array
	var err error

	a = i32s{0, 1, 2, 3, 4}
	b = i32s{1, 2, 2, 1, 100}
	c = i32sDummy{1, 2, 2, 1, 100}

	correct := make(bs, len(a))
	correctSame := make(i32s, len(a))
	for i, v := range a {
		correct[i] = v < b[i]

		if v < b[i] {
			correctSame[i] = int32(1)
		} else {
			correctSame[i] = int32(0)
		}
	}

	// return bools
	if res, err = a.Lt(b, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("Lt is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = i32s{0, 1, 2, 3, 4}
	if res, err = a.Lt(b, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(i32s) {
		if v != correctSame[i] {
			t.Errorf("Lt is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// now for compatible types

	// return bools
	a = i32s{0, 1, 2, 3, 4}
	if res, err = a.Lt(c, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("Lt is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = i32s{0, 1, 2, 3, 4}
	if res, err = a.Lt(c, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(i32s) {
		if v != correctSame[i] {
			t.Errorf("Lt is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// stupids # 1 : differing length
	if _, err := a.Lt(b[:3], true); err == nil {
		t.Errorf("Expected an error when performing Lt on differing lengths")
	}

	// stupids #2 : different types (which are checked before lengths)
	if _, err := a.Lt(f64s{}, true); err == nil {
		t.Errorf("Expected an error when performing Lt on a non-compatible type")
	}
}

func Test_u8s_Lt(t *testing.T) {
	var a, b u8s
	var c u8sDummy
	var res Array
	var err error

	a = u8s{0, 1, 2, 3, 4}
	b = u8s{1, 2, 2, 1, 100}
	c = u8sDummy{1, 2, 2, 1, 100}

	correct := make(bs, len(a))
	correctSame := make(u8s, len(a))
	for i, v := range a {
		correct[i] = v < b[i]

		if v < b[i] {
			correctSame[i] = byte(1)
		} else {
			correctSame[i] = byte(0)
		}
	}

	// return bools
	if res, err = a.Lt(b, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("Lt is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = u8s{0, 1, 2, 3, 4}
	if res, err = a.Lt(b, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(u8s) {
		if v != correctSame[i] {
			t.Errorf("Lt is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// now for compatible types

	// return bools
	a = u8s{0, 1, 2, 3, 4}
	if res, err = a.Lt(c, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("Lt is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = u8s{0, 1, 2, 3, 4}
	if res, err = a.Lt(c, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(u8s) {
		if v != correctSame[i] {
			t.Errorf("Lt is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// stupids # 1 : differing length
	if _, err := a.Lt(b[:3], true); err == nil {
		t.Errorf("Expected an error when performing Lt on differing lengths")
	}

	// stupids #2 : different types (which are checked before lengths)
	if _, err := a.Lt(f64s{}, true); err == nil {
		t.Errorf("Expected an error when performing Lt on a non-compatible type")
	}
}

/* Lte */

func Test_f64s_Lte(t *testing.T) {
	var a, b f64s
	var c f64sDummy
	var res Array
	var err error

	a = f64s{0, 1, 2, 3, 4}
	b = f64s{1, 2, 2, 1, 100}
	c = f64sDummy{1, 2, 2, 1, 100}

	correct := make(bs, len(a))
	correctSame := make(f64s, len(a))
	for i, v := range a {
		correct[i] = v <= b[i]

		if v <= b[i] {
			correctSame[i] = float64(1)
		} else {
			correctSame[i] = float64(0)
		}
	}

	// return bools
	if res, err = a.Lte(b, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("Lte is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = f64s{0, 1, 2, 3, 4}
	if res, err = a.Lte(b, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(f64s) {
		if v != correctSame[i] {
			t.Errorf("Lte is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// now for compatible types

	// return bools
	a = f64s{0, 1, 2, 3, 4}
	if res, err = a.Lte(c, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("Lte is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = f64s{0, 1, 2, 3, 4}
	if res, err = a.Lte(c, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(f64s) {
		if v != correctSame[i] {
			t.Errorf("Lte is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// stupids # 1 : differing length
	if _, err := a.Lte(b[:3], true); err == nil {
		t.Errorf("Expected an error when performing Lte on differing lengths")
	}

	// stupids #2 : different types (which are checked before lengths)
	if _, err := a.Lte(f32s{}, true); err == nil {
		t.Errorf("Expected an error when performing Lte on a non-compatible type")
	}
}

func Test_f32s_Lte(t *testing.T) {
	var a, b f32s
	var c f32sDummy
	var res Array
	var err error

	a = f32s{0, 1, 2, 3, 4}
	b = f32s{1, 2, 2, 1, 100}
	c = f32sDummy{1, 2, 2, 1, 100}

	correct := make(bs, len(a))
	correctSame := make(f32s, len(a))
	for i, v := range a {
		correct[i] = v <= b[i]

		if v <= b[i] {
			correctSame[i] = float32(1)
		} else {
			correctSame[i] = float32(0)
		}
	}

	// return bools
	if res, err = a.Lte(b, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("Lte is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = f32s{0, 1, 2, 3, 4}
	if res, err = a.Lte(b, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(f32s) {
		if v != correctSame[i] {
			t.Errorf("Lte is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// now for compatible types

	// return bools
	a = f32s{0, 1, 2, 3, 4}
	if res, err = a.Lte(c, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("Lte is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = f32s{0, 1, 2, 3, 4}
	if res, err = a.Lte(c, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(f32s) {
		if v != correctSame[i] {
			t.Errorf("Lte is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// stupids # 1 : differing length
	if _, err := a.Lte(b[:3], true); err == nil {
		t.Errorf("Expected an error when performing Lte on differing lengths")
	}

	// stupids #2 : different types (which are checked before lengths)
	if _, err := a.Lte(f64s{}, true); err == nil {
		t.Errorf("Expected an error when performing Lte on a non-compatible type")
	}
}

func Test_ints_Lte(t *testing.T) {
	var a, b ints
	var c intsDummy
	var res Array
	var err error

	a = ints{0, 1, 2, 3, 4}
	b = ints{1, 2, 2, 1, 100}
	c = intsDummy{1, 2, 2, 1, 100}

	correct := make(bs, len(a))
	correctSame := make(ints, len(a))
	for i, v := range a {
		correct[i] = v <= b[i]

		if v <= b[i] {
			correctSame[i] = int(1)
		} else {
			correctSame[i] = int(0)
		}
	}

	// return bools
	if res, err = a.Lte(b, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("Lte is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = ints{0, 1, 2, 3, 4}
	if res, err = a.Lte(b, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(ints) {
		if v != correctSame[i] {
			t.Errorf("Lte is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// now for compatible types

	// return bools
	a = ints{0, 1, 2, 3, 4}
	if res, err = a.Lte(c, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("Lte is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = ints{0, 1, 2, 3, 4}
	if res, err = a.Lte(c, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(ints) {
		if v != correctSame[i] {
			t.Errorf("Lte is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// stupids # 1 : differing length
	if _, err := a.Lte(b[:3], true); err == nil {
		t.Errorf("Expected an error when performing Lte on differing lengths")
	}

	// stupids #2 : different types (which are checked before lengths)
	if _, err := a.Lte(f64s{}, true); err == nil {
		t.Errorf("Expected an error when performing Lte on a non-compatible type")
	}
}

func Test_i64s_Lte(t *testing.T) {
	var a, b i64s
	var c i64sDummy
	var res Array
	var err error

	a = i64s{0, 1, 2, 3, 4}
	b = i64s{1, 2, 2, 1, 100}
	c = i64sDummy{1, 2, 2, 1, 100}

	correct := make(bs, len(a))
	correctSame := make(i64s, len(a))
	for i, v := range a {
		correct[i] = v <= b[i]

		if v <= b[i] {
			correctSame[i] = int64(1)
		} else {
			correctSame[i] = int64(0)
		}
	}

	// return bools
	if res, err = a.Lte(b, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("Lte is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = i64s{0, 1, 2, 3, 4}
	if res, err = a.Lte(b, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(i64s) {
		if v != correctSame[i] {
			t.Errorf("Lte is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// now for compatible types

	// return bools
	a = i64s{0, 1, 2, 3, 4}
	if res, err = a.Lte(c, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("Lte is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = i64s{0, 1, 2, 3, 4}
	if res, err = a.Lte(c, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(i64s) {
		if v != correctSame[i] {
			t.Errorf("Lte is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// stupids # 1 : differing length
	if _, err := a.Lte(b[:3], true); err == nil {
		t.Errorf("Expected an error when performing Lte on differing lengths")
	}

	// stupids #2 : different types (which are checked before lengths)
	if _, err := a.Lte(f64s{}, true); err == nil {
		t.Errorf("Expected an error when performing Lte on a non-compatible type")
	}
}

func Test_i32s_Lte(t *testing.T) {
	var a, b i32s
	var c i32sDummy
	var res Array
	var err error

	a = i32s{0, 1, 2, 3, 4}
	b = i32s{1, 2, 2, 1, 100}
	c = i32sDummy{1, 2, 2, 1, 100}

	correct := make(bs, len(a))
	correctSame := make(i32s, len(a))
	for i, v := range a {
		correct[i] = v <= b[i]

		if v <= b[i] {
			correctSame[i] = int32(1)
		} else {
			correctSame[i] = int32(0)
		}
	}

	// return bools
	if res, err = a.Lte(b, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("Lte is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = i32s{0, 1, 2, 3, 4}
	if res, err = a.Lte(b, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(i32s) {
		if v != correctSame[i] {
			t.Errorf("Lte is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// now for compatible types

	// return bools
	a = i32s{0, 1, 2, 3, 4}
	if res, err = a.Lte(c, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("Lte is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = i32s{0, 1, 2, 3, 4}
	if res, err = a.Lte(c, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(i32s) {
		if v != correctSame[i] {
			t.Errorf("Lte is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// stupids # 1 : differing length
	if _, err := a.Lte(b[:3], true); err == nil {
		t.Errorf("Expected an error when performing Lte on differing lengths")
	}

	// stupids #2 : different types (which are checked before lengths)
	if _, err := a.Lte(f64s{}, true); err == nil {
		t.Errorf("Expected an error when performing Lte on a non-compatible type")
	}
}

func Test_u8s_Lte(t *testing.T) {
	var a, b u8s
	var c u8sDummy
	var res Array
	var err error

	a = u8s{0, 1, 2, 3, 4}
	b = u8s{1, 2, 2, 1, 100}
	c = u8sDummy{1, 2, 2, 1, 100}

	correct := make(bs, len(a))
	correctSame := make(u8s, len(a))
	for i, v := range a {
		correct[i] = v <= b[i]

		if v <= b[i] {
			correctSame[i] = byte(1)
		} else {
			correctSame[i] = byte(0)
		}
	}

	// return bools
	if res, err = a.Lte(b, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("Lte is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = u8s{0, 1, 2, 3, 4}
	if res, err = a.Lte(b, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(u8s) {
		if v != correctSame[i] {
			t.Errorf("Lte is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// now for compatible types

	// return bools
	a = u8s{0, 1, 2, 3, 4}
	if res, err = a.Lte(c, false); err != nil {
		t.Error(err)
	}

	for i, v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("Lte is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = u8s{0, 1, 2, 3, 4}
	if res, err = a.Lte(c, true); err != nil {
		t.Error(err)
	}

	for i, v := range res.(u8s) {
		if v != correctSame[i] {
			t.Errorf("Lte is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// stupids # 1 : differing length
	if _, err := a.Lte(b[:3], true); err == nil {
		t.Errorf("Expected an error when performing Lte on differing lengths")
	}

	// stupids #2 : different types (which are checked before lengths)
	if _, err := a.Lte(f64s{}, true); err == nil {
		t.Errorf("Expected an error when performing Lte on a non-compatible type")
	}
}
