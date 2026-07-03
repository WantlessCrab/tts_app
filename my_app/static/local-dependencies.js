/*
 * local-dependencies.js
 * Loads optional browser-side runtime libraries from local static assets only.
 * No CDN or network fallback is attempted. Audio playback remains available through
 * NativeAudioBackend when optional visualization/PDF dependencies are absent.
 */
(function () {
    const state = {
        pdfjs: false,
        wavesurfer: false,
        errors: [],
        assets: {
            pdfjs: '/static/vendor/pdfjs/pdf.min.js',
            pdfWorker: '/static/vendor/pdfjs/pdf.worker.min.js',
            wavesurfer: '/static/vendor/wavesurfer/wavesurfer.min.js'
        },
        ready: null
    };

    function loadScript(src, label) {
        return new Promise((resolve) => {
            const script = document.createElement('script');
            script.src = src;
            script.async = false;
            script.onload = () => resolve(true);
            script.onerror = () => {
                state.errors.push({label, src, message: 'local asset not found or failed to load'});
                resolve(false);
            };
            document.head.appendChild(script);
        });
    }

    async function boot() {
        const pdfLoaded = await loadScript(state.assets.pdfjs, 'pdfjs');
        state.pdfjs = Boolean(pdfLoaded && window.pdfjsLib);
        if (state.pdfjs) {
            window.pdfjsLib.GlobalWorkerOptions.workerSrc = state.assets.pdfWorker;
        }

        const waveLoaded = await loadScript(state.assets.wavesurfer, 'wavesurfer');
        state.wavesurfer = Boolean(waveLoaded && window.WaveSurfer);

        return state;
    }

    state.ready = boot();
    window.TTSLocalDependencies = state;
})();