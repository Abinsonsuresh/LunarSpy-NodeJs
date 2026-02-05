const fs = require('fs');
const recorder = require('node-record-lpcm16');
const Meyda = require('meyda');
const wordsToTrain = require('./vocab'); // <--- IMPORTS YOUR LIST

const TRAINING_FILE = 'kashmiri_model.json';
const RECORDING_DURATION = 2000; 

let trainingData = {};
if (fs.existsSync(TRAINING_FILE)) {
    try {
        trainingData = JSON.parse(fs.readFileSync(TRAINING_FILE));
    } catch (e) {
        trainingData = {};
    }
}

const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

async function trainWord(wordObj) {
    console.log(`\n🎙️  Get ready to say: "${wordObj.kashmiri.toUpperCase()}"`);
    await sleep(1000); 
    console.log("🔴 RECORDING... (Speak now!)");

    return new Promise((resolve) => {
        let audioBuffer = [];
        let collectedFeatures = [];
        const BUFFER_SIZE = 512;

        const recording = recorder.record({
            sampleRate: 16000,
            threshold: 0,
            recordProgram: 'sox',
        });

        const stream = recording.stream();

        stream.on('data', (chunk) => {
            for (let i = 0; i < chunk.length; i += 2) {
                if (i + 1 < chunk.length) {
                    const int16 = chunk.readInt16LE(i);
                    audioBuffer.push(int16 / 32768);
                }
            }

            while (audioBuffer.length >= BUFFER_SIZE) {
                const signal = audioBuffer.slice(0, BUFFER_SIZE);
                audioBuffer = audioBuffer.slice(BUFFER_SIZE);
                Meyda.bufferSize = BUFFER_SIZE;
                const mfcc = Meyda.extract('mfcc', signal);
                if (mfcc) collectedFeatures.push(mfcc);
            }
        });

        setTimeout(() => {
            recording.stop();
            console.log(`✅ Stopped. Captured ${collectedFeatures.length} frames.`);
            resolve(collectedFeatures);
        }, RECORDING_DURATION);
    });
}

async function startTraining() {
    for (const item of wordsToTrain) {
        const features = await trainWord(item);
        trainingData[item.kashmiri] = {
            features: features,
            translation: item.english
        };
        console.log(`💾 Saved "${item.kashmiri}" to memory.`);
        await sleep(1000);
    }
    
    fs.writeFileSync(TRAINING_FILE, JSON.stringify(trainingData));
    console.log(`\n🎉 Training Complete! Model saved to ${TRAINING_FILE}`);
}

startTraining();