const fs = require('fs');
const recorder = require('node-record-lpcm16');
const Meyda = require('meyda');

const TRAINING_FILE = 'kashmiri_model.json';
// Duration: How long to listen for each word (in milliseconds)
const RECORDING_DURATION = 2000; 

const wordsToTrain = [
    { kashmiri: "khatara", english: "danger" }, 
    { kashmiri: "bandook", english: "gun" },
    { kashmiri: "salam", english: "peace" }
];

let trainingData = {};
if (fs.existsSync(TRAINING_FILE)) {
    try {
        trainingData = JSON.parse(fs.readFileSync(TRAINING_FILE));
    } catch (e) {
        trainingData = {};
    }
}

// Helper: Pauses execution for X milliseconds
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

async function trainWord(wordObj) {
    console.log(`\n🎙️  Get ready to say: "${wordObj.kashmiri.toUpperCase()}"`);
    await sleep(1000); // Give user a second to breathe
    console.log("🔴 RECORDING... (Speak now!)");

    return new Promise((resolve) => {
        let audioBuffer = [];
        let collectedFeatures = [];
        const BUFFER_SIZE = 512;

        const recording = recorder.record({
            sampleRate: 16000,
            threshold: 0, // Record everything, even silence
            recordProgram: 'sox',
        });

        const stream = recording.stream();

        // 1. Listen to the microphone stream directly
        stream.on('data', (chunk) => {
            // Convert raw bytes to float samples
            for (let i = 0; i < chunk.length; i += 2) {
                if (i + 1 < chunk.length) {
                    const int16 = chunk.readInt16LE(i);
                    audioBuffer.push(int16 / 32768);
                }
            }

            // Extract MFCCs as data comes in
            while (audioBuffer.length >= BUFFER_SIZE) {
                const signal = audioBuffer.slice(0, BUFFER_SIZE);
                audioBuffer = audioBuffer.slice(BUFFER_SIZE);

                Meyda.bufferSize = BUFFER_SIZE;
                const mfcc = Meyda.extract('mfcc', signal);
                if (mfcc) {
                    collectedFeatures.push(mfcc);
                }
            }
        });

        // 2. Stop after 2 seconds
        setTimeout(() => {
            recording.stop();
            console.log(`✅ Stopped. Captured ${collectedFeatures.length} feature frames.`);
            resolve(collectedFeatures);
        }, RECORDING_DURATION);
    });
}

async function startTraining() {
    for (const item of wordsToTrain) {
        // Clear previous data for this word if it exists
        const features = await trainWord(item);
        
        if (features.length < 10) {
            console.warn(`⚠️  WARNING: Very little audio detected for ${item.kashmiri}. Check your mic!`);
        }

        trainingData[item.kashmiri] = {
            features: features,
            translation: item.english
        };
        
        console.log(`💾 Saved "${item.kashmiri}" to memory.`);
        await sleep(1000); // Pause before next word
    }
    
    fs.writeFileSync(TRAINING_FILE, JSON.stringify(trainingData));
    console.log(`\n🎉 Training Complete! Model saved to ${TRAINING_FILE}`);
}

startTraining();