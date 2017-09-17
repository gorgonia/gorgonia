package main

import (
	"encoding/csv"
	"io"
	"log"
	"math/rand"
	"os"
	"strconv"

	"gorgonia.org/tensor"
)

func loadStatic() (w, x, y []float64) {
	d0, err := os.Open("testdata/X_ds1.10.csv")
	if err == nil {
		r := csv.NewReader(d0)
		r.Comma = ','
		for {
			record, err := r.Read()
			if err == io.EOF {
				break
			}
			if err != nil {
				log.Fatal(err)
			}

			for _, f := range record {
				fl, _ := strconv.ParseFloat(f, 64)
				x = append(x, fl)
			}
		}
		if len(x) != N*feats {
			log.Fatalf("Expected %d*%d. Got %d instead", N, feats, len(x))
		}
	} else {
		log.Println("could not read from file")
		x = tensor.Random(Float, N*feats).([]float64)
	}

	w0, err := os.Open("testdata/W_ds1.10.csv")
	if err == nil {
		r := csv.NewReader(w0)
		r.Comma = ' '
		for {
			record, err := r.Read()
			if err == io.EOF {
				break
			}
			if err != nil {
				log.Fatal(err)
			}

			fl, _ := strconv.ParseFloat(record[0], 64)
			w = append(w, fl)
		}

		if len(w) != feats {
			log.Fatalf("Expected %d rows. Got %d instead", feats, len(w))
		}
	} else {
		w = tensor.Random(Float, feats).([]float64)
	}

	y0, err := os.Open("testdata/Y_ds1.10.csv")
	if err == nil {
		r := csv.NewReader(y0)
		r.Comma = ','
		for {
			record, err := r.Read()
			if err == io.EOF {
				break
			}
			if err != nil {
				log.Fatal(err)
			}

			fl, _ := strconv.ParseFloat(record[0], 64)
			y = append(y, fl)
		}
		if len(y) != N {
			log.Fatalf("Expected %d rows. Got %d instead", N, len(y))
		}
	} else {
		y = make([]float64, N)
		for i := range y {
			y[i] = float64(rand.Intn(2))
		}

	}
	return
}
