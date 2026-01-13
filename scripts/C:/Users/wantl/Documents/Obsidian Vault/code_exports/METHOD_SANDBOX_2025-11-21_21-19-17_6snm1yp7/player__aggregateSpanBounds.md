# player.js - `_aggregateSpanBounds()`

**Source File**: `../my_app/static/player.js`
**Method Name**: `_aggregateSpanBounds`
**Line Number**: 1617
**Size**: 2,575 characters
**Extracted**: 2025-11-21 21:19:17

---

```javascript

  _aggregateSpanBounds(spans) {
    if (spans.length === 0) return [];

    const lineBoxes = [];
    let currentLine = null;

    // Calculate tolerance dynamically based on average font size
    const avgFontSize = spans.reduce((sum, s) => sum + s.font_size, 0) / spans.length;
    const yTolerance = Math.max(3, avgFontSize * 0.4);

    spans.forEach(span => {
      const bbox = span.bbox;

      // ✅ FIX: Use simple top/bottom variables (Y-down)
      const span_top = bbox.y;
      const span_bottom = bbox.y + bbox.height;

      if (!currentLine) {
        // Initialize line using Y-down coordinates
        currentLine = {
          x0: bbox.x,
          y0: span_top, // Start Y-coordinate (top edge)
          x1: bbox.x + bbox.width,
          y_top: span_top, // Track the highest (smallest Y) point
          y_bottom: span_bottom, // Track the lowest (largest Y) point
          count: 1
        };
      } else {
        // Check vertical proximity using the line's top edge (y_top) for stability
        if (Math.abs(span_top - currentLine.y_top) < yTolerance) {
          // Merge horizontally and vertically
          currentLine.x0 = Math.min(currentLine.x0, bbox.x);
          currentLine.x1 = Math.max(currentLine.x1, bbox.x + bbox.width);
          currentLine.y_top = Math.min(currentLine.y_top, span_top); // Shrink top (smaller Y value)
          currentLine.y_bottom = Math.max(currentLine.y_bottom, span_bottom); // Extend bottom (larger Y value)
          currentLine.count++;
        } else {
          // Span is on a new line, finalize the old one
          lineBoxes.push({
            x: currentLine.x0,
            y: currentLine.y_top, // Use the highest (smallest Y) point
            width: currentLine.x1 - currentLine.x0,
            height: currentLine.y_bottom - currentLine.y_top, // Calculate height accurately
          });

          // Start new line (Line 1636: Missing variable substitution here)
          currentLine = {
            x0: bbox.x,
            y0: span_top, // ✅ FIX: Use span_top instead of old bbox.y
            x1: bbox.x + bbox.width,
            y_top: span_top, // ✅ FIX: Use span_top instead of old y_top
            y_bottom: span_bottom, // ✅ FIX: Use span_bottom instead of old bbox.y
            count: 1
          };
        }
      }
    });

    if (currentLine) {
      lineBoxes.push({
        x: currentLine.x0,
        y: currentLine.y_top,
        width: currentLine.x1 - currentLine.x0,
        height: currentLine.y_bottom - currentLine.y_top,
      });
    }

    return lineBoxes;
  }
```
