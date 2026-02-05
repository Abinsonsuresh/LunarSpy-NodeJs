const recorder = require('node-record-lpcm16');
const Meyda = require('meyda');
const fs = require('fs');
const { exec } = require('child_process');

// --- CONFIGURATION ---
const TRAINING_FILE = 'kashmiri_model.json';

// --- DANGER WORDS LIST ---
// If the detected word matches one of these, it adds "Danger" to the speech.
const DANGER_WORDS = ["danger", "gun", "bomb", "kill", "attack", "fire"];

// --- DEBUG SETTINGS ---
const SHOW_VOLUME_METER = true; 

// --- TUNING ---
const MIN_VOLUME_THRESHOLD = 0.0002; 
const MATCH_THRESHOLD = 12; 

// Load Brain
if (!fs.existsSync(TRAINING_FILE)) {
    console.error("❌ Error: Model file not found. Run train.js first!");
    process.exit(1);
}
const knownWords = JSON.parse(fs.readFileSync(TRAINING_FILE));

console.log("------------------------------------------------");
console.log("🎙️  LISTENING... (Detecting ALL words from vocab.js)");
console.log("------------------------------------------------");

let liveFeatures = [];  
const MAX_HISTORY = 100; 

// --- SPEAKING FUNCTION ---
function speak(text) {
    const platform = process.platform;
    if (platform === 'darwin') { exec(`say "${text}"`); } 
    else if (platform === 'win32') { 
        const command = `PowerShell -Command "Add-Type –AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('${text}');"`;
        exec(command);
    } 
    else { exec(`espeak "${text}"`); }
}

const recording = recorder.record({
    sampleRate: 16000,
    threshold: 0,
    recordProgram: 'sox', 
});

const stream = recording.stream();
let rawAudioBuffer = [];
const BUFFER_SIZE = 512;

stream.on('data', (chunk) => {
    for (let i = 0; i < chunk.length; i += 2) {
        if (i + 1 < chunk.length) {
            const int16 = chunk.readInt16LE(i);
            rawAudioBuffer.push(int16 / 32768);
        }
    }

    while (rawAudioBuffer.length >= BUFFER_SIZE) {
        const signal = rawAudioBuffer.slice(0, BUFFER_SIZE);
        rawAudioBuffer = rawAudioBuffer.slice(BUFFER_SIZE);
        Meyda.bufferSize = BUFFER_SIZE;
        const features = Meyda.extract(['mfcc', 'rms'], signal);
        
        if (features) {
            // DEBUG: Volume Meter
            if (SHOW_VOLUME_METER && Math.random() < 0.1) { 
                 const bars = "I".repeat(Math.ceil(features.rms * 1000));
                 if (features.rms > MIN_VOLUME_THRESHOLD) {
                     console.log(`🔊 Hearing: ${features.rms.toFixed(5)} ${bars}`);
                 } 
            }

            if (features.rms > MIN_VOLUME_THRESHOLD) {
                liveFeatures.push(features.mfcc);
            } else {
                if (liveFeatures.length > 0) liveFeatures.shift();
            }
            
            if (liveFeatures.length > MAX_HISTORY) liveFeatures.shift();
            if (liveFeatures.length > 10 && liveFeatures.length % 5 === 0) checkForMatch();
        }
    }
});

function checkForMatch() {
    let bestMatch = null;
    let lowestDistance = Infinity;

    for (const [word, data] of Object.entries(knownWords)) {
        const trainedFeatures = data.features;
        if (liveFeatures.length < trainedFeatures.length * 0.5) continue;

        const compareLength = Math.min(liveFeatures.length, trainedFeatures.length);
        const liveSegment = liveFeatures.slice(liveFeatures.length - compareLength);
        const trainedSegment = trainedFeatures.slice(0, compareLength);

        const cost = computeDTW(liveSegment, trainedSegment);
        const normalizedCost = cost / compareLength;

        if (normalizedCost < lowestDistance) {
            lowestDistance = normalizedCost;
            bestMatch = word;
        }
    }

    if (lowestDistance < MATCH_THRESHOLD) {
        console.log(`\n🔎 Best Guess: ${bestMatch} (Score: ${lowestDistance.toFixed(2)})`);
        
        if (lowestDistance < MATCH_THRESHOLD) {
             const translation = knownWords[bestMatch].translation;
             triggerAlert(bestMatch, translation, lowestDistance);
             liveFeatures = []; 
        }
    }
}

// --- UPDATED ALERT FUNCTION ---
function triggerAlert(kashmiriWord, englishTranslation, confidence) {
    // 1. Log the Exact Match
    console.log(`\n✅✅✅ MATCH CONFIRMED: "${kashmiriWord.toUpperCase()}"`);
    console.log(`📢 DETECTED VOCAB WORD: [${englishTranslation.toUpperCase()}]`);

    // 2. Prepare text to speak
    let textToSpeak = englishTranslation;

    // 3. Check if it is a Danger Word
    if (DANGER_WORDS.includes(englishTranslation.toLowerCase())) {
        console.log(`🚨 THREAT DETECTED! Adding warning...`);
        // Append "Danger" to the speech
        textToSpeak = `${englishTranslation}... Danger Danger Danger Danger !`;
    }

    // 4. Speak
    speak(textToSpeak);
    
    console.log("------------------------------------------------");
}

function vectorDistance(v1, v2) {
    let sum = 0;
    for (let i = 0; i < v1.length; i++) sum += Math.pow(v1[i] - v2[i], 2);
    return Math.sqrt(sum);
}

function computeDTW(seq1, seq2) {
    const n = seq1.length;
    const m = seq2.length;
    const dtw = Array(n + 1).fill(null).map(() => Array(m + 1).fill(Infinity));
    dtw[0][0] = 0;
    for (let i = 1; i <= n; i++) {
        for (let j = 1; j <= m; j++) {
            const cost = vectorDistance(seq1[i - 1], seq2[j - 1]);
            dtw[i][j] = cost + Math.min(dtw[i - 1][j], dtw[i][j - 1], dtw[i - 1][j - 1]);
        }
    }
    return dtw[n][m];
}