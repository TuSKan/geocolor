package main

import (
	"bufio"
	"fmt"
	"log"
	"math"
	"os"

	"github.com/TuSKan/geocolor"
)

func main() {
	// 1. Define High-Resolution Configuration
	grid := &geocolor.GridConfig{
		ResL:        100,
		ResA:        301,
		ResB:        301,
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

	// Assuming your pipeline returns the actual RGB colors for each voxel
	// in a 4D tensor: [ResL][ResA][ResB][3]float32
	colors := cs.GetColors()
	if len(colors) > 0 {
		// Check the first non-black voxel's color values
		log.Printf("Debug - First Voxel Color: R:%f, G:%f, B:%f",
			colors[0][0][0][0], colors[0][0][0][1], colors[0][0][0][2])
	}

	// Pre-allocate the slice to avoid reallocation overhead (ResL * ResA * ResB * 6 floats)
	totalVoxels := grid.ResL * grid.ResA * grid.ResB
	vertexData := make([]float32, 0, totalVoxels*6)

	log.Println("Extracting 3D Point Cloud from tensor...")

	// Replace your extraction loop with this exact block
	for lIdx := 0; lIdx < grid.ResL; lIdx++ {
		for aIdx := 0; aIdx < grid.ResA; aIdx++ {
			for bIdx := 0; bIdx < grid.ResB; bIdx++ {
				// 1. Extract non-Riemannian Polar values from GoMLX
				L_val := float64(colors[lIdx][aIdx][bIdx][0])
				C_val := float64(colors[lIdx][aIdx][bIdx][1])
				H_val := float64(colors[lIdx][aIdx][bIdx][2])

				// 2. Convert Polar to Cartesian for spatial positioning
				a_pos := C_val * math.Cos(H_val)
				b_pos := C_val * math.Sin(H_val)

				// 3. Convert to Display-Ready sRGB
				r, g, bl := LabToRGB(L_val, a_pos, b_pos)

				// 4. Gamut Culling
				if r >= 0.0 && r <= 1.0 && g >= 0.0 && g <= 1.0 && bl >= 0.0 && bl <= 1.0 {
					vertexData = append(vertexData,
						float32(a_pos), float32(L_val), float32(b_pos),
						float32(r), float32(g), float32(bl),
					)
				}
			}
		}
	}

	log.Println("Writing to PLY file...")
	if err := ExportPointCloudToPLY("cloud.ply", vertexData); err != nil {
		log.Fatalf("Failed to export PLY: %v", err)
	}

	log.Println("Done! Drop geocolor_cloud.ply into MeshLab or Blender.")
}

// ExportPointCloudToPLY writes an interleaved [X,Y,Z, R,G,B] float32 slice to a PLY file.
// Expects XYZ as spatial coordinates and RGB as 0.0 - 1.0 normalized floats.
func ExportPointCloudToPLY(filename string, vertexData []float32) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	// Use buffered writer for massive performance gains on large point clouds
	writer := bufio.NewWriter(file)

	// A single vertex consists of 6 floats (X, Y, Z, R, G, B)
	numVertices := len(vertexData) / 6

	// 1. Write PLY Header
	writer.WriteString("ply\n")
	writer.WriteString("format ascii 1.0\n")
	writer.WriteString(fmt.Sprintf("element vertex %d\n", numVertices))
	writer.WriteString("property float x\n")
	writer.WriteString("property float y\n")
	writer.WriteString("property float z\n")
	writer.WriteString("property uchar red\n") // Colors in PLY are typically 0-255 ints
	writer.WriteString("property uchar green\n")
	writer.WriteString("property uchar blue\n")
	writer.WriteString("end_header\n")

	// 2. Write Vertices
	for i := 0; i < len(vertexData); i += 6 {
		x := vertexData[i]
		y := vertexData[i+1]
		z := vertexData[i+2]

		// Convert normalized 0.0-1.0 floats to 8-bit integers (0-255)
		r := uint8(vertexData[i+3] * 255.0)
		g := uint8(vertexData[i+4] * 255.0)
		b := uint8(vertexData[i+5] * 255.0)

		// Format: X Y Z R G B
		fmt.Fprintf(writer, "%f %f %f %d %d %d\n", x, y, z, r, g, b)
	}

	// 3. Flush the buffer to disk
	return writer.Flush()
}

func LabToRGB(l, a, b float64) (float64, float64, float64) {
	// 1. Lab to XYZ (D50)
	fy := (l + 16.0) / 116.0
	fx := a/500.0 + fy
	fz := fy - b/200.0

	invCurve := func(t float64) float64 {
		if t > 0.206893 {
			return t * t * t
		}
		return (t - 16.0/116.0) / 7.787
	}

	// Reference White D50
	x := invCurve(fx) * 0.96422
	y := invCurve(fy) * 1.00000
	z := invCurve(fz) * 0.82521

	// 2. Chromatic Adaptation: D50 -> D65 (Bradford Transform)
	// This ensures colors don't look "yellowed" on your monitor
	x65 := x*0.95557 + y*-0.02303 + z*0.06316
	y65 := x*-0.02828 + y*1.00994 + z*0.02101
	z65 := x*0.01229 + y*-0.02048 + z*1.32991

	// 3. XYZ (D65) to Linear sRGB
	rLinear := x65*3.2406 + y65*-1.5372 + z65*-0.4986
	gLinear := x65*-0.9689 + y65*1.8758 + z65*0.0415
	bLinear := x65*0.0557 + y65*-0.2040 + z65*1.0570

	// 4. Linear to sRGB Gamma
	srgb := func(c float64) float64 {
		if c <= 0.0031308 {
			return 12.92 * c
		}
		return 1.055*math.Pow(c, 1/2.4) - 0.055
	}

	return srgb(rLinear), srgb(gLinear), srgb(bLinear)
}
