package main

import (
	"log"

	"github.com/TuSKan/geocolor"

	_ "github.com/gogpu/gg-svg" // Register SVG backend
	"github.com/gogpu/gg/recording"
)

func main() {
	// 1. Define High-Resolution Configuration
	grid := &geocolor.GridConfig{
		ResL:        100,
		ResA:        101,
		ResB:        101,
		MinBounds:   [3]float64{0.0, -100.0, -100.0},
		MaxBounds:   [3]float64{100.0, 100.0, 100.0},
		OriginIndex: [3]int{0, 50, 50}, // Points exactly to L=0, a=0, b=0
	}

	cParam := float32(0.05) // Relaxed dampening curve

	// 2. Initialize and Compute
	log.Println("Compiling and executing XLA tensor graph...")
	cs := geocolor.NewColorSpace(grid, cParam)

	if err := cs.Compute(); err != nil {
		log.Fatal(err)
	}
	targetIndex := [3]int{90, 80, 50} // Start at L=90, a=60, b=0

	log.Println("Tracing Geodesic from Saturated Target...")
	L, C := cs.TraceGeodesic(targetIndex)
	smoothedL, smoothedC := SmoothPath(L, C, 5)
	log.Printf("Geodesic path calculated! It took %d steps.", len(L))

	// 3. SVG Canvas Configuration
	width := float64(800)
	height := float64(600)
	margin := float64(60)

	log.Println("Rendering SVG...")
	r := PlotAxis(width, height, margin, smoothedL, smoothedC)

	// Create SVG backend
	backend, err := recording.NewBackend("svg")
	if err != nil {
		log.Fatal(err)
	}

	// Playback to SVG
	r.Playback(backend)

	// Save to file
	if fb, ok := backend.(recording.FileBackend); ok {
		fb.SaveToFile("output.svg")
		log.Println("Successfully saved to output.svg")
	}
}

func PlotAxis(width, height, margin float64, L []float64, C []float64) *recording.Recording {
	rec := recording.NewRecorder(int(width), int(height))

	// White background
	rec.SetFillRGBA(1, 1, 1, 1)
	rec.DrawRectangle(0, 0, width, height)
	rec.Fill()

	// 1. Dynamic bounds calculation for Chroma (X-axis)
	minL, maxL := 0.0, 100.0
	minC := 0.0
	maxC := 1.0 // fallback minimum to prevent division by zero

	// Find the actual maximum C value dynamically so it never flies off-screen
	for _, cVal := range C {
		if cVal > maxC {
			maxC = cVal
		}
	}

	// Add 10% padding to the right so the curve doesn't touch the edge
	maxC = maxC * 1.10

	scaleX := (width - 2*margin) / (maxC - minC)
	scaleY := (height - 2*margin) / (maxL - minL)

	// Project: X-axis = Chroma (C), Y-axis = Lightness (L)
	project := func(l, c float64) (float64, float64) {
		x := margin + (c-minC)*scaleX
		y := height - margin - (l-minL)*scaleY // SVG Y goes down, so we subtract
		return x, y
	}

	// Draw Axes
	rec.SetStrokeRGBA(0, 0, 0, 1)
	rec.SetLineWidth(2)

	// X-axis (Chroma)
	rec.MoveTo(margin, height-margin)
	rec.LineTo(width-margin, height-margin)
	rec.Stroke()

	// Y-axis (Lightness)
	rec.MoveTo(margin, margin)
	rec.LineTo(margin, height-margin)
	rec.Stroke()

	// Draw theoretical straight neutral axis (Perfect Grey, C=0)
	rec.SetStrokeRGBA(0.5, 0.5, 0.5, 1) // Gray
	rec.SetLineWidth(1)

	// Dashed effect for the baseline
	for lStep := 0.0; lStep <= 100.0; lStep += 5.0 {
		x, y1 := project(lStep, 0)
		_, y2 := project(lStep+2.5, 0)
		rec.MoveTo(x, y1)
		rec.LineTo(x, y2)
		rec.Stroke()
	}

	// Draw the calculated curve safely
	if len(L) > 0 && len(C) > 0 {
		rec.SetStrokeRGBA(0.2, 0.4, 0.8, 1) // Blue
		rec.SetLineWidth(2.5)

		x0, y0 := project(L[0], C[0])
		rec.MoveTo(x0, y0)

		for i := 1; i < len(L); i++ {
			x, y := project(L[i], C[i])
			rec.LineTo(x, y)
		}

		// Fix the "Black Singularity" - if the trace hit the floor but didn't reach C=0,
		// mathematically draw the final step to the true absolute origin.
		if L[len(L)-1] == 0 && C[len(C)-1] > 0 {
			finalX, finalY := project(0, 0)
			rec.LineTo(finalX, finalY)
		}

		rec.Stroke()
	}

	return rec.FinishRecording()
}

// SmoothPath applies a moving average window to eliminate discrete grid-stepping artifacts
// and recover the true continuous geodesic curve.
func SmoothPath(L, C []float64, window int) ([]float64, []float64) {
	if len(L) < window {
		return L, C
	}

	smoothL := make([]float64, len(L))
	smoothC := make([]float64, len(C))

	for i := 0; i < len(L); i++ {
		sumL, sumC, count := 0.0, 0.0, 0.0

		// Average the neighboring points within the window
		for j := -window; j <= window; j++ {
			idx := i + j
			if idx >= 0 && idx < len(L) {
				sumL += L[idx]
				sumC += C[idx]
				count++
			}
		}
		smoothL[i] = sumL / count
		smoothC[i] = sumC / count
	}

	// Force the final point to hit absolute Black to fix the vertical drop artifact
	smoothL[len(smoothL)-1] = 0.0
	smoothC[len(smoothC)-1] = 0.0

	return smoothL, smoothC
}
