package main

import (
	"fmt"
	"handwritten-nn/dataset"
	"log"
)

func main() {
	data, err := dataset.ReadCsv("data/mnist_train.csv")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(data[0].Target)

}
