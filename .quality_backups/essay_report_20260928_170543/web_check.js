
        // DOM Elements
        const tempSlider = document.getElementById('tempSlider');
        const tempValue = document.getElementById('tempValue');
        const maxTokensSlider = document.getElementById('maxTokensSlider');
        const maxTokensValue = document.getElementById('maxTokensValue');
        const topkSlider = document.getElementById('topkSlider');
        const topkValue = document.getElementById('topkValue');
        const greedyToggle = document.getElementById('greedyToggle');
        const promptInput = document.getElementById('promptInput');
        const responseMode = document.getElementById('responseMode');
        const outputBody = document.getElementById('outputBody');
        const btnGenerate = document.getElementById('btnGenerate');
        const btnText = document.getElementById('btnText');
        const genStats = document.getElementById('genStats');

        // Sliders Listeners
        tempSlider.addEventListener('input', (e) => tempValue.textContent = parseFloat(e.target.value).toFixed(2));
        maxTokensSlider.addEventListener('input', (e) => maxTokensValue.textContent = e.target.value);
        topkSlider.addEventListener('input', (e) => topkValue.textContent = e.target.value);

        function usePreset(text) {
            promptInput.value = text;
            promptInput.focus();
        }

        async function fetchInfo() {
            try {
                const res = await fetch('/api/info');
                if (res.ok) {
                    const data = await res.json();
                    document.getElementById('gpuStatus').textContent = `${data.gpu_name} (${data.device.toUpperCase()})`;
                    document.getElementById('infoModelName').textContent = data.model_name;
                    document.getElementById('modelSummary').textContent = data.parameters_m + 'M parameter local model';
                    document.getElementById('infoParams').textContent = `${data.parameters_m}M`;
                    document.getElementById('infoVocab').textContent = `${data.vocab_size} Tokens`;
                    document.getElementById('infoValLoss').textContent = data.val_loss ? data.val_loss.toFixed(4) : 'N/A';
                }
            } catch (err) {
                console.error("Failed to fetch info:", err);
            }
        }

        async function generateText() {
            const prompt = promptInput.value;
            if (!prompt) return;

            btnGenerate.disabled = true;
            btnText.textContent = "Generating...";
            genStats.textContent = "Preparing your reply...";

            const startTime = performance.now();

            try {
                const response = await fetch('/api/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        prompt: prompt,
                        mode: responseMode.value,
                        max_new_tokens: parseInt(maxTokensSlider.value),
                        temperature: parseFloat(tempSlider.value),
                        top_k: parseInt(topkSlider.value),
                        greedy: greedyToggle.checked,
                        use_tools: document.getElementById('useTools').checked
                    })
                });

                const data = await response.json();
                const duration = ((performance.now() - startTime) / 1000).toFixed(2);

                if (data.status === 'success') {
                    document.getElementById('replySource').textContent = data.source === 'local_calculator' ? 'Local calculator' : 'CrachoLM output';
                    if (data.mode === 'chat') {
                        outputBody.textContent = data.reply;
                    } else {
                        outputBody.innerHTML = `<span class="prompt-highlight">${escapeHtml(data.prompt)}</span>${escapeHtml(data.generated_text.slice(data.prompt.length))}`;
                    }
                    genStats.textContent = 'Completed in ' + duration + 's';
                } else {
                    outputBody.textContent = `Error: ${data.message}`;
                    genStats.textContent = "Generation failed";
                }
            } catch (err) {
                outputBody.textContent = `Network Error: ${err.message}`;
                genStats.textContent = "Error communicating with server";
            } finally {
                btnGenerate.disabled = false;
                btnText.textContent = "Generate Response";
            }
        }

        function copyOutput() {
            const text = outputBody.innerText;
            navigator.clipboard.writeText(text);
            alert("Output text copied to clipboard!");
        }

        function escapeHtml(text) {
            return text
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#039;");
        }

        // Initialize info on page load
        fetchInfo();
    