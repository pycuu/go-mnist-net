package dataset

import (
	"encoding/csv"
	"os"
	"strconv"
)

type Sample struct {
	Input  []float64
	Label  int
	Target []float64
}

// returns objects "sample" with normalised values
func ReadCsv(file_path string) ([]Sample, error) {
	f, err := os.Open(file_path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	csv_reader := csv.NewReader(f)
	records, err := csv_reader.ReadAll()
	if err != nil {
		return nil, err
	}

	var samples []Sample
	for _, record := range records {
		if len(record) < 2 {
			continue
		}
		label, err := strconv.Atoi(record[0])
		if err != nil {
			continue
		}
		input := make([]float64, len(record)-1)
		for i := 1; i < len(record); i++ {
			val, err := strconv.ParseFloat(record[i], 64)
			if err != nil {
				input[i-1] = 0.0
			} else {
				input[i-1] = val / 255.0
			}
		}
		target := make([]float64, 10)
		if label >= 0 && label < 10 {
			target[label] = 1.0
		}
		samples = append(samples, Sample{
			Input:  input,
			Label:  label,
			Target: target,
		})
	}
	return samples, nil
}
