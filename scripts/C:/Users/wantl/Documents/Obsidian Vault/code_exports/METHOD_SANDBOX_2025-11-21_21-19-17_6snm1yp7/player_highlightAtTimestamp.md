# player.js - `highlightAtTimestamp()`

**Source File**: `../my_app/static/player.js`
**Method Name**: `highlightAtTimestamp`
**Line Number**: 418
**Size**: 1,651 characters
**Extracted**: 2025-11-21 21:19:17

---

```javascript

  highlightAtTimestamp(citation) {
    // Defensive check: Highlighting requires a valid index pointer and coordinate data cache
    if (!citation || citation.span_start_index === -1 || !this.state.audiobook.coordinateData) {
      return;
    }

    const currentPage = citation.page;
    // Use the coordinateData cache (which is the content of the _raw.json file)
    const pageCoordData = this.state.audiobook.coordinateData[currentPage - 1];

    if (!pageCoordData || !pageCoordData.coordinate_blocks) {
      return;
    }

    const allSpans = pageCoordData.coordinate_blocks;
    const startIdx = citation.span_start_index;
    const endIdx = citation.span_end_index;

    // Slice the spans array using the indices provided by the backend
    const sentenceSpans = allSpans.slice(startIdx, endIdx + 1);

    if (sentenceSpans.length === 0) {
      return;
    }

    // Aggregate the span bounding boxes into rectangular lines
    const lineBoxes = this._aggregateSpanBounds(sentenceSpans);
    const container = this.elements.highlightContainer;

    // Draw rectangles for each line box
    lineBoxes.forEach((lineBox) => {
      const screenCoords = this.calculateCanvasCoordinates(lineBox);
      if (!screenCoords) return;

      const highlightDiv = document.createElement('div');
      highlightDiv.className = 'highlight-rect sentence current';

      highlightDiv.style.left = `${screenCoords.x}px`;
      highlightDiv.style.top = `${screenCoords.y}px`;
      highlightDiv.style.width = `${screenCoords.width}px`;
      highlightDiv.style.height = `${screenCoords.height}px`;

      container.appendChild(highlightDiv);
    });
  }
```
