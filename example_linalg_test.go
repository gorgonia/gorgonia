package gorgonia

import (
	"fmt"
	"log"
	"testing"

	"gorgonia.org/tensor"
)

func ExampleBatchedMatMul() {
	g := NewGraph()
	a := NewTensor(g, Float64, 3, WithShape(2, 2, 3), WithInit(RangedFrom(1)), WithName("a"))
	b := NewTensor(g, Float64, 3, WithShape(2, 3, 2), WithInit(RangedFrom(13)), WithName("b"))
	c, err := BatchedMatMul(a, b)
	if err != nil {
		log.Fatal(err)
	}

	d := NewTensor(g, Float64, 3, WithShape(10, 1, 1), WithInit(RangedFrom(1)), WithName("d"))
	e := NewTensor(g, Float64, 3, WithShape(10, 1, 10), WithInit(RangedFrom(11)), WithName("e"))
	f, err := BatchedMatMul(d, e)
	if err != nil {
		log.Fatal(err)
	}
	h, err := BatchedMatMul(e, d, true, true)
	if err != nil {
		log.Fatal(err)
	}

	i := NewTensor(g, Float64, 4, WithShape(1, 3, 2, 4), WithInit(RangedFrom(1)), WithName("i"))
	j := NewTensor(g, Float64, 4, WithShape(1, 3, 4, 2), WithInit(ValuesOf(10.0)), WithName("j"))
	k, err := BatchedMatMul(i, j)
	if err != nil {
		log.Fatal(err)
	}

	x := NewTensor(g, Float64, 4, WithShape(3, 2, 2, 3), WithInit(RangedFrom(1)), WithName("x"))
	y := NewTensor(g, Float64, 4, WithShape(3, 2, 3, 2), WithInit(RangedFrom(37)), WithName("y"))
	z, err := BatchedMatMul(x, y)
	if err != nil {
		log.Fatal(err)
	}

	m := NewTapeMachine(g)
	if err := m.RunAll(); err != nil {
		log.Fatal(err)
	}

	fmt.Printf("a: %v\n%v\n", a.Value().Shape(), a.Value().Data())
	fmt.Printf("b: %v\n%v\n", b.Value().Shape(), b.Value().Data())
	fmt.Printf("c: %v\n%v\n", c.Value().Shape(), c.Value().Data())
	fmt.Printf("d: %v\n%v\n", d.Value().Shape(), d.Value().Data())
	fmt.Printf("e: %v\n%v\n", e.Value().Shape(), e.Value().Data())
	fmt.Printf("f: %v\n%v\n", f.Value().Shape(), f.Value().Data())
	fmt.Printf("h: %v\n%v\n", h.Value().Shape(), h.Value().Data())
	fmt.Printf("i: %v\n%v\n", i.Value().Shape(), i.Value().Data())
	fmt.Printf("j: %v\n%v\n", j.Value().Shape(), j.Value().Data())
	fmt.Printf("k: %v\n%v\n", k.Value().Shape(), k.Value().Data())
	fmt.Printf("x: %v\n%v\n", x.Value().Shape(), x.Value().Data())
	fmt.Printf("y: %v\n%v\n", y.Value().Shape(), y.Value().Data())
	fmt.Printf("z: %v\n%v\n", z.Value().Shape(), z.Value().Data())

	// Output:
	// a: (2, 2, 3)
	// [1 2 3 4 5 6 7 8 9 10 11 12]
	// b: (2, 3, 2)
	// [13 14 15 16 17 18 19 20 21 22 23 24]
	// c: (2, 2, 2)
	// [94 100 229 244 508 532 697 730]
	// d: (10, 1, 1)
	// [1 2 3 4 5 6 7 8 9 10]
	// e: (10, 1, 10)
	// [11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110]
	// f: (10, 1, 10)
	// [11 12 13 14 15 16 17 18 19 20 42 44 46 48 50 52 54 56 58 60 93 96 99 102 105 108 111 114 117 120 164 168 172 176 180 184 188 192 196 200 255 260 265 270 275 280 285 290 295 300 366 372 378 384 390 396 402 408 414 420 497 504 511 518 525 532 539 546 553 560 648 656 664 672 680 688 696 704 712 720 819 828 837 846 855 864 873 882 891 900 1010 1020 1030 1040 1050 1060 1070 1080 1090 1100]
	// h: (10, 10, 1)
	// [11 12 13 14 15 16 17 18 19 20 42 44 46 48 50 52 54 56 58 60 93 96 99 102 105 108 111 114 117 120 164 168 172 176 180 184 188 192 196 200 255 260 265 270 275 280 285 290 295 300 366 372 378 384 390 396 402 408 414 420 497 504 511 518 525 532 539 546 553 560 648 656 664 672 680 688 696 704 712 720 819 828 837 846 855 864 873 882 891 900 1010 1020 1030 1040 1050 1060 1070 1080 1090 1100]
	// i: (1, 3, 2, 4)
	// [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24]
	// j: (1, 3, 4, 2)
	// [10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10]
	// k: (1, 3, 2, 2)
	// [100 100 260 260 420 420 580 580 740 740 900 900]
	// x: (3, 2, 2, 3)
	// [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36]
	// y: (3, 2, 3, 2)
	// [37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72]
	// z: (3, 2, 2, 2)
	// [238 244 589 604 1084 1108 1489 1522 2146 2188 2605 2656 3424 3484 3937 4006 4918 4996 5485 5572 6628 6724 7249 7354]

}

func TestIncrSlices(t *testing.T) {
	// validSlices to see if the slice and shape matches
	validSlices := func(a []sli, shp tensor.Shape) bool {
		for i := range a {
			if a[i].start < shp[i] {
				return true
			}
		}
		return false
	}

	shp := tensor.Shape{2, 3, 4}
	slices := make([]sli, len(shp))
	for i := range slices {
		slices[i].end = 1
	}

	for halt := false; !halt; halt = incrSlices(slices, shp) {
		if !validSlices(slices, shp) {
			t.Errorf("Generated invalid slice %v", slices)
		}
	}
}

func ExampleBatchedMatMul_withBackprop() {
	g := NewGraph()
	a := NewTensor(g, Float64, 4, WithShape(2, 4, 3, 9), WithInit(RangedFrom(1)), WithName("a"))
	b := NewTensor(g, Float64, 4, WithShape(2, 4, 3, 9), WithInit(RangedFrom(13)), WithName("b"))
	c, err := BatchedMatMul(a, b, false, true)
	if err != nil {
		log.Fatal(err)
	}
	s, err := Sum(c)
	if err != nil {
		log.Fatal(err)
	}
	grads, err := Grad(s, a, b)
	if err != nil {
		log.Fatal(err)
	}

	m := NewTapeMachine(g)
	if err := m.RunAll(); err != nil {
		log.Fatal(err)
	}

	fmt.Printf("a: %v\n%v\n", a.Value().Shape(), a.Value().Data())
	fmt.Printf("b: %v\n%v\n", b.Value().Shape(), b.Value().Data())
	fmt.Printf("c: %v\n%v\n", c.Value().Shape(), c.Value().Data())
	fmt.Printf("grads[0]:%v\n%v\n", grads[0].Shape(), grads[0].Value().Data())
	// Output:
	// a: (2, 4, 3, 9)
	// [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216]
	// b: (2, 4, 3, 9)
	// [13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224 225 226 227 228]
	// c: (2, 4, 3, 3)
	// [825 1230 1635 2202 3336 4470 3579 5442 7305 12732 15324 17916 16296 19617 22938 19860 23910 27960 37761 42540 47319 43512 49020 54528 49263 55500 61737 75912 82878 89844 83850 91545 99240 91788 100212 108636 127185 136338 145491 137310 147192 157074 147435 158046 168657 191580 202920 214260 203892 215961 228030 216204 229002 241800 269097 282624 296151 283596 297852 312108 298095 313080 328065 359736 375450 391164 376422 392865 409308 393108 410280 427452]
	// grads[0]:(2, 4, 3, 9)
	// [66 69 72 75 78 81 84 87 90 66 69 72 75 78 81 84 87 90 66 69 72 75 78 81 84 87 90 147 150 153 156 159 162 165 168 171 147 150 153 156 159 162 165 168 171 147 150 153 156 159 162 165 168 171 228 231 234 237 240 243 246 249 252 228 231 234 237 240 243 246 249 252 228 231 234 237 240 243 246 249 252 309 312 315 318 321 324 327 330 333 309 312 315 318 321 324 327 330 333 309 312 315 318 321 324 327 330 333 390 393 396 399 402 405 408 411 414 390 393 396 399 402 405 408 411 414 390 393 396 399 402 405 408 411 414 471 474 477 480 483 486 489 492 495 471 474 477 480 483 486 489 492 495 471 474 477 480 483 486 489 492 495 552 555 558 561 564 567 570 573 576 552 555 558 561 564 567 570 573 576 552 555 558 561 564 567 570 573 576 633 636 639 642 645 648 651 654 657 633 636 639 642 645 648 651 654 657 633 636 639 642 645 648 651 654 657]
}
