// Global Variables for Model, Tokenizer, Chart, and Control
let model = null;
let tokenizer = null;
let lossChart = null;
let trainingActive = false;
let maxSeqLength = 3; // Loaded dynamically from sliders
let defaultSampleText = `artificial intelligence and deep learning are transforming our modern world. recurrent neural networks are designed to analyze sequential data such as text paragraphs or time series. an lstm or long short term memory network is a special type of recurrent neural network that is capable of learning long term dependencies. next word prediction is a fascinating natural language processing task that powers systems like autocomplete search engines and smart keyboards. we will train our neural network directly in the web browser using tensorflow.js. with enough epochs and training data, the lstm model will learn to write sentences word by word. keep learning and creating projects to master machine learning.`;

// 1. Simple Tokenizer Class (Mirrors Keras Tokenizer)
class SimpleTokenizer {
    constructor() {
        this.wordIndex = {};
        this.indexWord = {};
        this.vocabSize = 0;
    }

    // Clean text by lowercasing, removing punctuation, and trimming extra spaces
    cleanText(text) {
        return text.toLowerCase()
            .replace(/[^\w\s']/g, ' ')  // Remove punctuation (keep alphanumeric, space, apostrophe)
            .replace(/\s+/g, ' ')      // Replace multiple spaces with a single space
            .trim();
    }

    // Fit tokenizer on text to create word-to-index mapping
    fitOnText(text) {
        const cleaned = this.cleanText(text);
        const words = cleaned.split(' ').filter(w => w.length > 0);
        
        this.wordIndex = {};
        this.indexWord = {};
        let index = 1; // Index 0 is reserved for padding in deep learning
        
        words.forEach(word => {
            if (!this.wordIndex[word]) {
                this.wordIndex[word] = index;
                this.indexWord[index] = word;
                index++;
            }
        });
        
        this.vocabSize = index; // Total vocabulary size (unique words + 1)
    }

    // Convert text lines into sequence of numbers
    textsToSequences(lines) {
        return lines.map(line => {
            const cleaned = this.cleanText(line);
            const words = cleaned.split(' ').filter(w => w.length > 0);
            return words.map(word => this.wordIndex[word] || 0).filter(idx => idx !== 0);
        });
    }
}

// 2. DOM Elements Selection
const textInput = document.getElementById('text-input');
const fileUpload = document.getElementById('file-upload');
const loadSampleBtn = document.getElementById('load-sample-btn');
const trainBtn = document.getElementById('train-btn');
const stopTrainBtn = document.getElementById('stop-train-btn');
const gpuStatus = document.getElementById('gpu-status');

// Sliders and Value Displays
const paramSeqLen = document.getElementById('param-seq-len');
const paramEmbedDim = document.getElementById('param-embed-dim');
const paramLstmUnits = document.getElementById('param-lstm-units');
const paramEpochs = document.getElementById('param-epochs');
const paramLr = document.getElementById('param-lr');

const valSeqLen = document.getElementById('val-seq-len');
const valEmbedDim = document.getElementById('val-embed-dim');
const valLstmUnits = document.getElementById('val-lstm-units');
const valEpochs = document.getElementById('val-epochs');
const valLr = document.getElementById('val-lr');

// Metric Panels
const metricVocab = document.getElementById('metric-vocab');
const metricSequences = document.getElementById('metric-sequences');
const metricEpoch = document.getElementById('metric-epoch');
const metricLoss = document.getElementById('metric-loss');
const progressFill = document.getElementById('progress-bar-fill');
const progressPercent = document.getElementById('progress-percent');
const consoleOutput = document.getElementById('console-output');
const trainStatus = document.getElementById('train-status');

// Sandbox Controls
const playgroundCard = document.getElementById('playground-card');
const modelReadyBadge = document.getElementById('model-ready-badge');
const seedInput = document.getElementById('seed-input');
const predictBtn = document.getElementById('predict-btn');
const generateWordsCount = document.getElementById('generate-words-count');
const valGenerateWords = document.getElementById('val-generate-words');
const generatedTextBox = document.getElementById('generated-text-box');
const distributionBars = document.getElementById('distribution-bars');

// 3. Setup Interactive UI Listeners & Inits
window.addEventListener('DOMContentLoaded', () => {
    // Check GPU Status via TF.js Backend
    tf.ready().then(() => {
        const backend = tf.getBackend();
        if (backend === 'webgl') {
            gpuStatus.innerHTML = '<i class="fa-solid fa-circle-check" style="color: var(--success)"></i> GPU Active (WebGL)';
            logConsole('TensorFlow.js loaded successfully. GPU acceleration (WebGL) is active!', 'success');
        } else {
            gpuStatus.innerHTML = `<i class="fa-solid fa-circle-exclamation" style="color: var(--warning)"></i> CPU Mode (${backend})`;
            logConsole(`TensorFlow.js loaded in CPU Mode. GPU was not detected.`, 'system');
        }
    });

    // Populate Default Sample Text
    textInput.value = defaultSampleText;
    initChart();
});

// Update slider labels dynamically
paramSeqLen.addEventListener('input', (e) => { valSeqLen.innerText = e.target.value; });
paramEmbedDim.addEventListener('input', (e) => { valEmbedDim.innerText = e.target.value; });
paramLstmUnits.addEventListener('input', (e) => { valLstmUnits.innerText = e.target.value; });
paramEpochs.addEventListener('input', (e) => { valEpochs.innerText = e.target.value; });
paramLr.addEventListener('input', (e) => { valLr.innerText = e.target.value; });
generateWordsCount.addEventListener('input', (e) => { valGenerateWords.innerText = e.target.value; });

// Load sample text button
loadSampleBtn.addEventListener('click', () => {
    textInput.value = defaultSampleText;
    logConsole('Loaded default sample text corpus.', 'system');
});

// File Upload Handler
fileUpload.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (evt) => {
        textInput.value = evt.target.result;
        logConsole(`Successfully loaded file: "${file.name}" (${file.size} bytes).`, 'success');
    };
    reader.onerror = () => {
        logConsole(`Error reading file: "${file.name}".`, 'error');
    };
    reader.readAsText(file);
});

// Write to the terminal console
function logConsole(message, type = 'system') {
    const timestamp = new Date().toLocaleTimeString();
    const div = document.createElement('div');
    div.className = `log-line ${type}`;
    div.innerHTML = `<span style="color: var(--text-dim)">[${timestamp}]</span> ${message}`;
    consoleOutput.appendChild(div);
    consoleOutput.scrollTop = consoleOutput.scrollHeight;
}

// 4. Initialize Chart.js
function initChart() {
    const ctx = document.getElementById('lossChart').getContext('2d');
    
    // Set custom grid color
    Chart.defaults.color = 'rgba(62, 39, 35, 0.6)'; // Cocoa brown text labels
    Chart.defaults.font.family = 'Inter';
    
    lossChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Training Loss',
                data: [],
                borderColor: '#795548', // Cocoa brown for loss line
                backgroundColor: 'rgba(121, 85, 72, 0.08)',
                borderWidth: 2,
                fill: true,
                tension: 0.3,
                yAxisID: 'y'
            }, {
                label: 'Training Accuracy',
                data: [],
                borderColor: '#cc9a52', // Amber gold for accuracy line
                backgroundColor: 'rgba(198, 138, 76, 0.04)',
                borderWidth: 2,
                fill: false,
                tension: 0.3,
                yAxisID: 'y1'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: { boxWidth: 12, font: { size: 10 } }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(121, 85, 72, 0.08)' }, // Warm sepia gridlines
                    title: { display: true, text: 'Epoch', font: { size: 10 } }
                },
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    grid: { color: 'rgba(121, 85, 72, 0.08)' }, // Warm sepia gridlines
                    title: { display: true, text: 'Loss', font: { size: 10 } }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    grid: { drawOnChartArea: false },
                    title: { display: true, text: 'Accuracy', font: { size: 10 } },
                    min: 0,
                    max: 1
                }
            }
        }
    });
}

// Clear Chart Data
function resetChart() {
    if (lossChart) {
        lossChart.data.labels = [];
        lossChart.data.datasets[0].data = [];
        lossChart.data.datasets[1].data = [];
        lossChart.update();
    }
}

// 5. Deep Learning Pipeline: Training Loop
trainBtn.addEventListener('click', async () => {
    const rawText = textInput.value.trim();
    if (!rawText) {
        alert("Please paste some paragraph text or upload a dataset first!");
        return;
    }

    // Set UI to training state
    trainingActive = true;
    trainBtn.classList.add('hidden');
    stopTrainBtn.classList.remove('hidden');
    toggleSettingsInputs(true);
    resetChart();
    
    // Clear playground state
    playgroundCard.classList.add('disabled');
    modelReadyBadge.classList.add('hidden');
    seedInput.disabled = true;
    predictBtn.disabled = true;
    generateWordsCount.disabled = true;
    generatedTextBox.innerHTML = '<span class="placeholder">Training the model will unlock this playground...</span>';
    distributionBars.innerHTML = '<div class="bar-placeholder"><i class="fa-solid fa-chart-simple"></i><span>Softmax distribution loads here during prediction</span></div>';

    trainStatus.className = "badge badge-success animate-pulse";
    trainStatus.innerText = "Training...";
    logConsole("Initializing Tokenizer and cleaning text...", "system");

    // Retrieve parameter values
    maxSeqLength = parseInt(paramSeqLen.value);
    const embeddingDim = parseInt(paramEmbedDim.value);
    const lstmUnits = parseInt(paramLstmUnits.value);
    const epochsCount = parseInt(paramEpochs.value);
    const batchSize = parseInt(document.getElementById('val-epochs').innerText) > 100 ? 32 : 16; // Autocalc batch size
    const learningRate = parseFloat(paramLr.value);

    // Run Asynchronously to prevent UI freezing
    setTimeout(async () => {
        try {
            // 1. Tokenize Text
            tokenizer = new SimpleTokenizer();
            tokenizer.fitOnText(rawText);
            
            // Split corpus by sentences/lines
            const lines = rawText.split(/[.\n!?]/)
                .map(line => line.trim())
                .filter(line => line.length > 5);

            if (lines.length === 0) {
                throw new Error("No readable sentences found! Make sure you use punctuation (periods) or line breaks.");
            }

            const sequences = tokenizer.textsToSequences(lines);
            
            // Create incremental n-gram sequences
            // E.g., sequence: [1, 2, 3, 4] -> n-grams: [1, 2], [1, 2, 3], [1, 2, 3, 4]
            const trainingSequences = [];
            sequences.forEach(seq => {
                for (let i = 1; i < seq.length; i++) {
                    trainingSequences.push(seq.slice(0, i + 1));
                }
            });

            if (trainingSequences.length === 0) {
                throw new Error("Text is too short to generate training sequences. Please provide more words.");
            }

            // Update stats
            metricVocab.innerText = tokenizer.vocabSize;
            metricSequences.innerText = trainingSequences.length;
            logConsole(`Tokenization complete! Unique vocabulary words: ${tokenizer.vocabSize}`, "success");
            logConsole(`Generated ${trainingSequences.length} training sequences (N-grams).`, "system");

            // 2. Prep Tensors (Features & Labels)
            logConsole("Converting sequences to multi-dimensional tensors...", "system");
            const { X, y } = prepareTrainingData(trainingSequences, tokenizer.vocabSize, maxSeqLength);
            
            // 3. Build LSTM Sequential Network
            logConsole("Compiling LSTM Neural Network in WebGL context...", "system");
            model = buildModel(tokenizer.vocabSize, maxSeqLength, embeddingDim, lstmUnits, learningRate);
            logConsole("Model Architecture compiled successfully.", "success");

            // 4. Train Model
            logConsole("Starting training loop...", "system");
            
            await model.fit(X, y, {
                epochs: epochsCount,
                batchSize: batchSize,
                callbacks: {
                    onEpochEnd: async (epoch, logs) => {
                        // Check if user pressed stop training
                        if (!trainingActive) {
                            model.stopTraining = true;
                            return;
                        }

                        const currentEpoch = epoch + 1;
                        const loss = logs.loss.toFixed(4);
                        const acc = logs.acc.toFixed(4);

                        // Update metrics UI
                        metricEpoch.innerText = `${currentEpoch}/${epochsCount}`;
                        metricLoss.innerText = loss;

                        // Update progress bar
                        const pct = Math.round((currentEpoch / epochsCount) * 100);
                        progressFill.style.width = `${pct}%`;
                        progressPercent.innerText = `${pct}%`;

                        // Update chart
                        lossChart.data.labels.push(currentEpoch);
                        lossChart.data.datasets[0].data.push(logs.loss);
                        lossChart.data.datasets[1].data.push(logs.acc);
                        lossChart.update();

                        // Log epoch result occasionally to avoid console spamming
                        if (currentEpoch === 1 || currentEpoch % 5 === 0 || currentEpoch === epochsCount) {
                            logConsole(`Epoch ${currentEpoch}/${epochsCount} - Loss: ${loss} - Accuracy: ${acc}`, "epoch");
                        }
                    }
                }
            });

            // Cleanup training tensors from GPU memory to prevent memory leaks
            X.dispose();
            y.dispose();

            if (trainingActive) {
                // Training completed successfully
                logConsole("LSTM model training completed successfully!", "success");
                trainStatus.className = "badge badge-success";
                trainStatus.innerText = "Completed";
                
                // Unlock Prediction playground
                playgroundCard.classList.remove('disabled');
                modelReadyBadge.classList.remove('hidden');
                seedInput.disabled = false;
                predictBtn.disabled = false;
                generateWordsCount.disabled = false;
                seedInput.placeholder = "Enter seed word(s) here...";
                
                // Pre-fill seed input with first 2 words of vocab to guide them
                const vocabWords = Object.keys(tokenizer.wordIndex);
                if (vocabWords.length >= 2) {
                    seedInput.value = vocabWords[0] + " " + vocabWords[1];
                }

                generatedTextBox.innerHTML = '<span class="placeholder">Type words above and click "Predict"!</span>';
            } else {
                // Training was aborted
                logConsole("Training aborted by the user.", "error");
                trainStatus.className = "badge";
                trainStatus.innerText = "Stopped";
                metricEpoch.innerText = "-";
                metricLoss.innerText = "-";
                progressFill.style.width = `0%`;
                progressPercent.innerText = `0%`;
            }

        } catch (error) {
            logConsole(`Error: ${error.message}`, "error");
            alert(`An error occurred: ${error.message}`);
            trainStatus.className = "badge";
            trainStatus.innerText = "Error";
            trainingActive = false;
        } finally {
            // Restore UI buttons
            trainBtn.classList.remove('hidden');
            stopTrainBtn.classList.add('hidden');
            toggleSettingsInputs(false);
            trainingActive = false;
        }
    }, 100);
});

// Stop Training Trigger
stopTrainBtn.addEventListener('click', () => {
    trainingActive = false;
    logConsole("Stopping training at end of current epoch...", "warning");
});

// Helper to disable configuration fields during training
function toggleSettingsInputs(disable) {
    textInput.disabled = disable;
    fileUpload.disabled = disable;
    loadSampleBtn.disabled = disable;
    paramSeqLen.disabled = disable;
    paramEmbedDim.disabled = disable;
    paramLstmUnits.disabled = disable;
    paramEpochs.disabled = disable;
    paramLr.disabled = disable;
}

// 6. Prepares Input Tensors from Raw sequence array
function prepareTrainingData(sequences, vocabSize, seqLength) {
    const X_data = [];
    const y_data = [];

    sequences.forEach(seq => {
        let padded = [...seq];
        if (padded.length < seqLength + 1) {
            // Pad at the start (pre-padding)
            const padCount = (seqLength + 1) - padded.length;
            padded = Array(padCount).fill(0).concat(padded);
        } else if (padded.length > seqLength + 1) {
            // Slice if sequence exceeds maximum sequence capacity
            padded = padded.slice(padded.length - (seqLength + 1));
        }

        const inputX = padded.slice(0, seqLength);
        const labelY = padded[padded.length - 1];

        X_data.push(inputX);
        y_data.push(labelY);
    });

    const X = tf.tensor2d(X_data, [X_data.length, seqLength], 'int32');
    const y = tf.oneHot(tf.tensor1d(y_data, 'int32'), vocabSize);

    return { X, y };
}

// 7. Builds TF.js Layers Model
function buildModel(vocabSize, seqLength, embeddingDim, lstmUnits, learningRate) {
    const model = tf.sequential();
    
    // 1. Embedding Layer - maps word integers to continuous semantic vectors
    model.add(tf.layers.embedding({
        inputDim: vocabSize,
        outputDim: embeddingDim,
        inputLength: seqLength
    }));
    
    // 2. LSTM Layer - learns the contextual sequence features
    model.add(tf.layers.lstm({
        units: lstmUnits,
        returnSequences: false
    }));
    
    // 3. Dropout - prevents overfitting
    model.add(tf.layers.dropout({ rate: 0.2 }));
    
    // 4. Output Layer - Softmax returns probability distributions for all vocab words
    model.add(tf.layers.dense({
        units: vocabSize,
        activation: 'softmax'
    }));

    model.compile({
        optimizer: tf.train.adam(learningRate),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    return model;
}

// 8. Next-Word Prediction and Generation Sandbox logic
predictBtn.addEventListener('click', async () => {
    if (!model || !tokenizer) return;

    const rawSeed = seedInput.value.trim();
    if (!rawSeed) {
        alert("Please enter a starting phrase!");
        return;
    }

    const wordsToGenCount = parseInt(generateWordsCount.value);
    
    // Initialize Generated Text Display with the seed words
    generatedTextBox.innerHTML = `<span class="seed">${rawSeed}</span>`;
    
    let currentSentence = rawSeed;
    
    for (let w = 0; w < wordsToGenCount; w++) {
        // Clean and tokenize current sentence
        const cleanedText = tokenizer.cleanText(currentSentence);
        const tokens = cleanedText.split(' ').map(word => tokenizer.wordIndex[word] || 0).filter(idx => idx !== 0);

        if (tokens.length === 0) break;

        // Pad sequence to maxSeqLength
        let padded = [...tokens];
        if (padded.length < maxSeqLength) {
            const padCount = maxSeqLength - padded.length;
            padded = Array(padCount).fill(0).concat(padded);
        } else if (padded.length > maxSeqLength) {
            padded = padded.slice(padded.length - maxSeqLength);
        }

        // Run Model Prediction
        const inputTensor = tf.tensor2d([padded], [1, maxSeqLength], 'int32');
        const predictionTensor = model.predict(inputTensor);
        
        // Retrieve Output probability distribution values
        const probs = await predictionTensor.data();
        
        // Clean up input & output tensors to prevent WebGL leaks
        inputTensor.dispose();
        predictionTensor.dispose();

        // Map probabilities array to indexes
        const indexProbs = Array.from(probs).map((p, idx) => ({
            index: idx,
            prob: p
        }));

        // Sort descending by probability
        indexProbs.sort((a, b) => b.prob - a.prob);

        // Word index 0 is padding, if indexProbs[0].index is 0, we look at the second item
        let bestCandidate = indexProbs[0];
        if (bestCandidate.index === 0 && indexProbs.length > 1) {
            bestCandidate = indexProbs[1];
        }

        // Retrieve mapped word
        const predictedWord = tokenizer.indexWord[bestCandidate.index];
        
        if (!predictedWord) {
            break; // Stop if no word is matched
        }

        // Append to current sentence string
        currentSentence += " " + predictedWord;

        // Render predicted word in UI with nice pop animation
        const wordSpan = document.createElement('span');
        wordSpan.className = 'predicted-word';
        wordSpan.innerText = predictedWord;
        
        // Add a space before appending word
        generatedTextBox.appendChild(document.createTextNode(' '));
        generatedTextBox.appendChild(wordSpan);

        // Update the probability bars for the FIRST word predicted in this batch
        if (w === 0) {
            renderSoftmaxBars(indexProbs);
        }

        // Add a tiny delay between letters to make text generation feel "alive"
        await new Promise(resolve => setTimeout(resolve, 150));
    }
});

// Render the Top 3 Word Probability bars
function renderSoftmaxBars(indexProbs) {
    distributionBars.innerHTML = '';
    
    // Select top 3 candidate indexes
    let candidates = indexProbs.filter(cp => cp.index !== 0).slice(0, 3);
    
    if (candidates.length === 0) {
        distributionBars.innerHTML = '<div class="bar-placeholder"><span>No predictions generated</span></div>';
        return;
    }

    candidates.forEach(cand => {
        const word = tokenizer.indexWord[cand.index] || `<INDEX:${cand.index}>`;
        const percentage = (cand.prob * 100).toFixed(1);
        
        const container = document.createElement('div');
        container.className = 'prob-bar-container';
        
        container.innerHTML = `
            <div class="prob-labels">
                <span class="prob-word">${word}</span>
                <span class="prob-pct">${percentage}%</span>
            </div>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill" style="width: 0%"></div>
            </div>
        `;
        
        distributionBars.appendChild(container);
        
        // Trigger CSS animation of width after a small delay
        setTimeout(() => {
            const fill = container.querySelector('.prob-bar-fill');
            if (fill) fill.style.width = `${percentage}%`;
        }, 50);
    });
}
