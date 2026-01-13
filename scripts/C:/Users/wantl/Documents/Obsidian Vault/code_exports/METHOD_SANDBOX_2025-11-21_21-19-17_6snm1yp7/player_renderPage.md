# player.js - `renderPage()`

**Source File**: `../my_app/static/player.js`
**Method Name**: `renderPage`
**Line Number**: 1408
**Size**: 3,896 characters
**Extracted**: 2025-11-21 21:19:17

---

```javascript
    async renderPage(pageNum) {
      if (!this.state.pdf.pdfDocument) {
        console.warn('No PDF loaded');
        return;
      }
      if (pageNum < 1 || pageNum > this.state.pdf.totalPages) {
        console.warn(`Invalid page number: ${pageNum}`);
        return;
      }

      // ACTION 2.4: Clear highlights immediately after validation
      this.clearHighlights();

      // Set render lock (prevents simultaneous renders)
      if (this.state.pdf.isPageRendering) {
        this.state.pdf.pendingPageNum = pageNum;
        return;
      }

      this.state.pdf.isPageRendering = true;
      this.state.pdf.currentPageNum = pageNum;

      // Disable buttons during render
      if (this.elements.prevPageButton) this.elements.prevPageButton.disabled = true;
      if (this.elements.nextPageButton) this.elements.nextPageButton.disabled = true;

      try {
        const page = await this.state.pdf.pdfDocument.getPage(pageNum);
        const outputScale = window.devicePixelRatio || 1;

        // ZOOM LOGIC
        let scale = this.state.pdf.scale;

        if (scale === null || scale === undefined) {
          const fitMode = this.state.pdf.fitMode || 'width';
          const viewportDefault = page.getViewport({scale: 1.0});

          if (fitMode === 'height') {
            // Removed padding compensation (CSS fix handles padding externally)
            const desiredHeight = this.elements.pdfViewerContainer.clientHeight;
            scale = desiredHeight / viewportDefault.height;
          } else {
            // Removed padding compensation (CSS fix handles padding externally)
            const desiredWidth = this.elements.pdfViewerContainer.clientWidth;
            scale = desiredWidth / viewportDefault.width;
          }

          this.state.pdf.scale = scale;
        }

        const viewport = page.getViewport({scale: scale});
        this.state.pdf.viewport = viewport;

        const canvas = this.elements.pdfCanvas;
        const context = canvas.getContext('2d');

        // Set internal resolution (quality)
        canvas.width = Math.floor(viewport.width * outputScale);
        canvas.height = Math.floor(viewport.height * outputScale);

        // Set display size (layout)
        canvas.style.width = Math.floor(viewport.width) + 'px';
        canvas.style.height = Math.floor(viewport.height) + 'px';

        // 4.3 Highlight Scroll Fix: Set container size to match canvas
        this.elements.highlightContainer.style.width = canvas.style.width;
        this.elements.highlightContainer.style.height = canvas.style.height;


        // Render with transform for HiDPI
        const transform = outputScale !== 1
            ? [outputScale, 0, 0, outputScale, 0, 0]
            : null;

        const renderContext = {
          canvasContext: context,
          viewport: viewport,
          transform: transform
        };

        await page.render(renderContext).promise;

        // Update UI
        this.elements.pageIndicator.textContent =
            `Page: ${pageNum} / ${this.state.pdf.totalPages}`;

        if (this.elements.zoomLevel) {
          this.updateZoomDisplay();
        }

      } catch (error) {
        console.error('Page render error:', error);
        this.logError('Failed to render page: ' + error.message);
      } finally {
        this.state.pdf.isPageRendering = false;

        // Re-enable buttons based on page number
        if (this.elements.prevPageButton) this.elements.prevPageButton.disabled = (pageNum === 1);
        if (this.elements.nextPageButton) this.elements.nextPageButton.disabled = (pageNum === this.state.pdf.totalPages);

        // Process pending page request
        if (this.state.pdf.pendingPageNum !== null) {
          const pending = this.state.pdf.pendingPageNum;
          this.state.pdf.pendingPageNum = null;
          this.renderPage(pending);
        }
      }
    }
```
