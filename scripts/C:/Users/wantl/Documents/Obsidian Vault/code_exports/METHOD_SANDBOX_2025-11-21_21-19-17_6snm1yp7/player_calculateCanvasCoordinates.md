# player.js - `calculateCanvasCoordinates()`

**Source File**: `../my_app/static/player.js`
**Method Name**: `calculateCanvasCoordinates`
**Line Number**: 1588
**Size**: 818 characters
**Extracted**: 2025-11-21 21:19:17

---

```javascript

  calculateCanvasCoordinates(pdfCoords) {
    const viewport = this.state.pdf.viewport;
    const canvas = this.elements.pdfCanvas;

    if (!viewport || !canvas) {
      return null;
    }

    // Transform PDF bounding box corners to canvas space
    // viewport.transform handles scaling + Y-flip automatically
    const [x1, y1] = pdfjsLib.Util.applyTransform(
        [pdfCoords.x, pdfCoords.y],
        viewport.transform
    );

    const [x2, y2] = pdfjsLib.Util.applyTransform(
        [pdfCoords.x + pdfCoords.width, pdfCoords.y + pdfCoords.height],
        viewport.transform
    );

    // Use min/max/abs to ensure correct bounding box regardless of corner order
    return {
      x: Math.min(x1, x2),
      y: Math.min(y1, y2),
      width: Math.abs(x2 - x1),
      height: Math.abs(y2 - y1)
    };
  }
```
