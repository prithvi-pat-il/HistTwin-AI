import { AvatarController } from './avatar.js?v=60';

// DOM Elements
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const messagesContainer = document.getElementById('messages-container');
const fileInput = document.getElementById('file-input');
const uploadBtn = document.getElementById('upload-btn');
const cameraBtn = document.getElementById('camera-btn');
const previewContainer = document.getElementById('attachment-preview');
const previewImg = document.getElementById('preview-img');
const removeAttachmentBtn = document.getElementById('remove-attachment');
const languageSelect = document.getElementById('language-select');
const voiceSelect = document.getElementById('voice-select');
const ttsEngineSelect = document.getElementById('tts-engine-select');
const clearChatBtn = document.getElementById('clear-chat');
const micBtn = document.getElementById('mic-btn');

// Virtual Guide Master Toggle & Panel Elements
const toggleGuideBtn = document.getElementById('toggle-guide-btn');
const placeholderEnableBtn = document.getElementById('placeholder-enable-btn');
const characterPanel = document.getElementById('character-panel');
const guideOffPlaceholder = document.getElementById('guide-off-placeholder');
const guideStatusBadge = document.getElementById('guide-status-badge');
const avatarContainer = document.getElementById('avatar-container');
const togglePanelBtn = document.getElementById('toggle-panel-btn');
const avatarSelect = document.getElementById('avatar-select');
const expressionSelect = document.getElementById('expression-select');

// Camera Elements
const cameraModal = document.getElementById('camera-modal');
const cameraStream = document.getElementById('camera-stream');
const captureBtn = document.getElementById('capture-btn');
const closeCameraBtn = document.getElementById('close-camera');
const cameraCanvas = document.getElementById('camera-canvas');

// State
let currentFile = null;
let currentContextPlace = null;
let isVirtualGuideEnabled = false; // OFF BY DEFAULT AS PER REQUIREMENT
let avatarController = null;
let voices = [];

// Sync Language & Voice dropdowns
if (languageSelect && voiceSelect) {
    languageSelect.addEventListener('change', () => {
        voiceSelect.value = languageSelect.value;
    });
    voiceSelect.addEventListener('change', () => {
        languageSelect.value = voiceSelect.value;
    });
}

// Populate Speech Synthesis Voices
function populateVoices() {
    if (window.speechSynthesis) {
        voices = window.speechSynthesis.getVoices();
    }
}
populateVoices();
if (window.speechSynthesis && window.speechSynthesis.onvoiceschanged !== undefined) {
    window.speechSynthesis.onvoiceschanged = populateVoices;
}

// --- Virtual Guide Master Toggle Functionality ---
function setVirtualGuideState(enabled) {
    isVirtualGuideEnabled = enabled;
    
    if (enabled) {
        // Activate 3D Guide Mode
        characterPanel.classList.remove('hidden');
        guideOffPlaceholder.classList.add('hidden');
        
        toggleGuideBtn.classList.add('active');
        toggleGuideBtn.querySelector('.guide-btn-text').textContent = "Virtual Guide Active";
        guideStatusBadge.innerHTML = `Mode: <span class="font-semibold text-emerald-400">3D Avatar Active</span>`;
        
        // Initialize Avatar Controller lazily when enabled
        if (!avatarController && avatarContainer) {
            avatarController = new AvatarController(avatarContainer);
        }
    } else {
        // Disable 3D Guide Mode (Default Mode)
        characterPanel.classList.add('hidden');
        guideOffPlaceholder.classList.remove('hidden');
        
        toggleGuideBtn.classList.remove('active');
        toggleGuideBtn.querySelector('.guide-btn-text').textContent = "Enable Virtual Guide";
        guideStatusBadge.innerHTML = `Mode: <span class="font-semibold text-slate-300">Standard Text</span>`;
        
        if (avatarController) {
            avatarController.stopSpeaking();
        }
    }
}

// Virtual Guide Event Listeners
if (toggleGuideBtn) {
    toggleGuideBtn.addEventListener('click', () => {
        setVirtualGuideState(!isVirtualGuideEnabled);
    });
}
if (placeholderEnableBtn) {
    placeholderEnableBtn.addEventListener('click', () => {
        setVirtualGuideState(true);
    });
}
if (togglePanelBtn) {
    togglePanelBtn.addEventListener('click', () => {
        setVirtualGuideState(false);
    });
}

// Avatar Model Selector
if (avatarSelect) {
    avatarSelect.addEventListener('change', (e) => {
        if (avatarController && isVirtualGuideEnabled) {
            avatarController.loadModel(e.target.value);
        }
    });
}

// Facial Expression Selector
if (expressionSelect) {
    expressionSelect.addEventListener('change', (e) => {
        if (avatarController) {
            avatarController.setExpression(e.target.value);
        }
    });
}

// --- Speech Recognition (STT) ---
let recognition = null;
if ('webkitSpeechRecognition' in window) {
    recognition = new webkitSpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;

    recognition.onstart = () => {
        micBtn.classList.add('text-amber-400');
        chatInput.placeholder = "Listening...";
    };

    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        chatInput.value += (chatInput.value ? ' ' : '') + transcript;
    };

    recognition.onerror = () => {
        micBtn.classList.remove('text-amber-400');
        chatInput.placeholder = "Ask HistTwin AI about Raigad...";
    };

    recognition.onend = () => {
        micBtn.classList.remove('text-amber-400');
        chatInput.placeholder = "Ask HistTwin AI about Raigad...";
    };
} else {
    micBtn.style.display = 'none';
}

micBtn.addEventListener('click', () => {
    if (recognition) {
        if (micBtn.classList.contains('text-amber-400')) {
            recognition.stop();
        } else {
            const langMap = { 'English': 'en-US', 'Hindi': 'hi-IN', 'Marathi': 'mr-IN' };
            const currentVoice = voiceSelect ? voiceSelect.value : 'English';
            recognition.lang = langMap[currentVoice] || 'en-US';
            recognition.start();
        }
    }
});

// File Attachment Handler
uploadBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) handleFile(file);
});

function handleFile(file) {
    currentFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        previewContainer.classList.remove('hidden');
    };
    reader.readAsDataURL(file);
}

removeAttachmentBtn.addEventListener('click', () => {
    currentFile = null;
    fileInput.value = '';
    previewContainer.classList.add('hidden');
});

// Camera Controls
let stream = null;
cameraBtn.addEventListener('click', async () => {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        cameraStream.srcObject = stream;
        cameraModal.classList.remove('hidden');
    } catch (err) {
        alert("Unable to access camera: " + err.message);
    }
});

closeCameraBtn.addEventListener('click', stopCamera);

function stopCamera() {
    if (stream) stream.getTracks().forEach(track => track.stop());
    cameraModal.classList.add('hidden');
}

captureBtn.addEventListener('click', () => {
    const context = cameraCanvas.getContext('2d');
    cameraCanvas.width = cameraStream.videoWidth;
    cameraCanvas.height = cameraStream.videoHeight;
    context.drawImage(cameraStream, 0, 0, cameraCanvas.width, cameraCanvas.height);
    
    cameraCanvas.toBlob((blob) => {
        const file = new File([blob], "camera_capture.jpg", { type: "image/jpeg" });
        handleFile(file);
        stopCamera();
    }, 'image/jpeg');
});

// Clear Chat
// Clear Chat
clearChatBtn.addEventListener('click', () => {
    messagesContainer.innerHTML = `
        <div class="message ai flex items-start gap-4">
            <img src="logo.png" alt="HistTwin AI" class="w-10 h-10 rounded-2xl object-cover shrink-0 shadow-lg border border-amber-500/40 mt-0.5">
            <div class="content ios-glass-card px-6 py-5 rounded-2xl max-w-2xl text-zinc-100 leading-relaxed text-sm shadow-xl">
                New conversation started! I am <b>HistTwin AI</b>. How can I assist you with Raigad district history today?
            </div>
        </div>
    `;
    currentContextPlace = null;
});

// Send Message Implementation
sendBtn.addEventListener('click', sendMessage);
chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});

async function sendMessage() {
    const text = chatInput.value.trim();
    if (!text && !currentFile) return;

    // Display user message
    let imgSrc = currentFile ? previewImg.src : null;
    addMessage('user', text, imgSrc);

    // Automatically summon the 3D character when user asks a question or uploads an image
    if (!isVirtualGuideEnabled) {
        setVirtualGuideState(true);
    }

    // Reset inputs
    chatInput.value = '';
    const fileToSend = currentFile;
    currentFile = null;
    fileInput.value = '';
    previewContainer.classList.add('hidden');

    // Display loading message
    const loadingId = addLoadingMessage();

    try {
        const formData = new FormData();
        if (fileToSend) formData.append('file', fileToSend);
        if (text) formData.append('text', text);
        formData.append('language', languageSelect.value);
        if (ttsEngineSelect) formData.append('tts_engine', ttsEngineSelect.value);
        if (currentContextPlace) formData.append('context_place', currentContextPlace);

        const response = await fetch('/chat', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        removeMessage(loadingId);

        if (data.error) {
            addMessage('ai', "Sorry: " + data.error);
        } else {
            if (data.type === 'prediction' || data.prediction) {
                currentContextPlace = data.prediction;
            }

            let textLines = data.text_lines || [data.response];
            let videoUrl = data.video_url || null;
            let audioUrl = data.audio_url || null;
            let motionUrl = data.motion_url || data.motion_data || null;
            let ttsEngineUsed = data.tts_engine || (ttsEngineSelect ? ttsEngineSelect.value : 'elevenlabs');

            addMessage('ai', textLines, null, audioUrl, motionUrl, null, videoUrl, ttsEngineUsed);
        }

    } catch (err) {
        removeMessage(loadingId);
        addMessage('ai', ["Error connecting to server."]);
        console.error(err);
    }
}

// Line-by-Line Typing & Narration Synced Render
function addMessage(role, linesOrText, imgSrc = null, audioUrl = null, motionUrl = null, visemeUrl = null, videoUrl = null, ttsEngine = 'elevenlabs') {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role} flex items-start gap-4`;

    const isAI = role === 'ai';
    const avatarBadge = isAI 
        ? `<img src="logo.png" alt="HistTwin AI" class="w-10 h-10 rounded-2xl object-cover shrink-0 shadow-lg border border-amber-500/40 mt-0.5">` 
        : `<div class="avatar w-10 h-10 rounded-2xl bg-zinc-800 border border-white/15 text-zinc-200 flex items-center justify-center font-bold text-xs shrink-0 mt-0.5 shadow-md">You</div>`;

    let lines = Array.isArray(linesOrText) ? linesOrText : [linesOrText];
    let contentHtml = '';

    if (imgSrc) {
        contentHtml += `<img src="${imgSrc}" alt="User Image" class="max-w-xs rounded-xl mb-3 border border-white/15 shadow-lg"><br>`;
    }

    // Line-by-line container
    contentHtml += `<div class="story-lines-wrapper space-y-2">`;
    lines.forEach((line, idx) => {
        let formattedLine = line.replace(/\*\*(.*?)\*\*/g, '<b>$1</b>');
        contentHtml += `<div class="story-line" style="animation-delay: ${idx * 0.15}s">${formattedLine}</div>`;
    });
    contentHtml += `</div>`;

    if (videoUrl) {
        contentHtml += `<div class="mt-4"><video src="${videoUrl}" autoplay loop controls class="w-full max-w-md rounded-xl border border-white/15 shadow-xl"></video></div>`;
    }

    // Single Play/Pause Button for AI responses
    if (isAI) {
        const langName = (voiceSelect ? voiceSelect.value : 'English');
        const engineLabel = (ttsEngine === 'huggingface' || ttsEngine === 'hf') ? '🤗 Hugging Face' : '⚡ ElevenLabs';
        contentHtml += `
            <div class="msg-audio-controls mt-4 pt-3 border-t border-white/10 flex items-center gap-3 flex-wrap">
                <span class="lang-badge bg-zinc-900/80 border border-white/15 text-zinc-400 px-2.5 py-1 rounded-md text-[11px] font-medium">${langName}</span>
                <span class="engine-badge bg-amber-500/10 border border-amber-500/30 text-amber-300 px-2.5 py-1 rounded-md text-[11px] font-semibold">${engineLabel}</span>
                <button class="play-pause-btn bg-emerald-500 hover:bg-emerald-600 text-white px-3.5 py-1.5 rounded-xl text-xs font-semibold flex items-center gap-1.5 transition-all shadow-lg">
                    <span class="btn-icon">▶</span> <span class="btn-text">Play Narration</span>
                </button>
            </div>
        `;
    }

    msgDiv.innerHTML = `
        ${avatarBadge}
        <div class="content ios-glass-card px-6 py-5 rounded-2xl max-w-2xl text-zinc-100 leading-relaxed text-sm shadow-xl w-full">
            ${contentHtml}
        </div>
    `;

    messagesContainer.appendChild(msgDiv);
    scrollToBottom();

    // Bind Play/Pause & Virtual Guide Sync Logic
    if (isAI) {
        const btn = msgDiv.querySelector('.play-pause-btn');
        if (btn) {
            let isPlaying = false;
            let audio = audioUrl ? new Audio(audioUrl) : null;
            let utterance = null;
            let motionData = null;
            let assetsLoaded = false;

            const loadAssets = async () => {
                if (assetsLoaded) return;
                try {
                    if (motionUrl) {
                        motionData = await fetch(motionUrl).then(res => res.json()).catch(() => null);
                    }
                    assetsLoaded = true;
                } catch (e) {
                    console.error("Failed to load motion assets:", e);
                }
            };

            const updateButtonState = (playing) => {
                isPlaying = playing;
                const icon = btn.querySelector('.btn-icon');
                const textEl = btn.querySelector('.btn-text');
                if (playing) {
                    icon.textContent = "⏸";
                    textEl.textContent = "Pause";
                    btn.className = "play-pause-btn bg-red-500 hover:bg-red-600 text-white px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all shadow-md";
                } else {
                    icon.textContent = "▶";
                    textEl.textContent = "Play Narration";
                    btn.className = "play-pause-btn bg-emerald-500 hover:bg-emerald-600 text-white px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all shadow-md";
                }
            };

            const stopAllGlobalSpeech = () => {
                if (window.currentPlayingAudio) window.currentPlayingAudio.pause();
                if (window.speechSynthesis) window.speechSynthesis.cancel();
                document.querySelectorAll('.play-pause-btn').forEach(otherBtn => {
                    if (otherBtn !== btn) {
                        const icon = otherBtn.querySelector('.btn-icon');
                        const textEl = otherBtn.querySelector('.btn-text');
                        if (icon && textEl) {
                            icon.textContent = "▶";
                            textEl.textContent = "Play Narration";
                            otherBtn.className = "play-pause-btn bg-emerald-500 hover:bg-emerald-600 text-white px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all shadow-md";
                        }
                    }
                });
            };

            const startPlayback = async () => {
                stopAllGlobalSpeech();

                if (audio) {
                    window.currentPlayingAudio = audio;
                    await loadAssets();

                    if (isVirtualGuideEnabled && avatarController) {
                        avatarController.playMotion(audio, motionData);
                        avatarController.startSpeaking(audio);
                    }

                    audio.play().then(() => updateButtonState(true)).catch(err => console.warn(err));
                } else if (window.speechSynthesis) {
                    // Browser Web Speech Fallback
                    const fullText = lines.join(' ').replace(/\*\*(.*?)\*\*/g, '$1');
                    utterance = new SpeechSynthesisUtterance(fullText);
                    const langMap = { 'English': 'en-US', 'Hindi': 'hi-IN', 'Marathi': 'mr-IN' };
                    const targetLang = langMap[voiceSelect.value || languageSelect.value || 'English'];
                    if (targetLang && voices.length > 0) {
                        const matchedVoice = voices.find(v => v.lang.startsWith(targetLang.substring(0, 2)));
                        if (matchedVoice) utterance.voice = matchedVoice;
                    }

                    utterance.onstart = () => {
                        updateButtonState(true);
                        if (isVirtualGuideEnabled && avatarController) {
                            avatarController.startSpeaking();
                        }
                    };

                    const stopSpeechAnim = () => {
                        updateButtonState(false);
                        if (avatarController) avatarController.stopSpeaking();
                    };

                    utterance.onend = stopSpeechAnim;
                    utterance.onerror = stopSpeechAnim;

                    window.speechSynthesis.speak(utterance);
                }
            };

            btn.addEventListener('click', async () => {
                if (isPlaying) {
                    if (audio) audio.pause();
                    else if (window.speechSynthesis) window.speechSynthesis.cancel();
                    if (avatarController) avatarController.stopSpeaking();
                    updateButtonState(false);
                } else {
                    await startPlayback();
                }
            });

            if (audio) {
                audio.addEventListener('ended', () => {
                    updateButtonState(false);
                    if (avatarController) avatarController.stopSpeaking();
                    if (window.currentPlayingAudio === audio) window.currentPlayingAudio = null;
                });
                audio.addEventListener('pause', () => updateButtonState(false));
            }

            // AUTO-PLAY ONLY WHEN VIRTUAL GUIDE IS ACTIVELY ENABLED
            if (isVirtualGuideEnabled) {
                startPlayback();
            }
        }
    }
}

function addLoadingMessage() {
    const id = 'loading-' + Date.now();
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message ai flex items-start gap-4';
    msgDiv.id = id;
    msgDiv.innerHTML = `
        <img src="logo.png" alt="HistTwin AI" class="w-10 h-10 rounded-2xl object-cover shrink-0 shadow-lg border border-amber-500/40 mt-0.5">
        <div class="content ios-glass-card px-6 py-4 rounded-2xl text-zinc-200 text-sm shadow-xl flex items-center gap-3">
            <div class="loader"></div>
            <span class="text-xs text-zinc-400 font-medium">HistTwin AI is generating your historical guide...</span>
        </div>
    `;
    messagesContainer.appendChild(msgDiv);
    scrollToBottom();
    return id;
}

function removeMessage(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function scrollToBottom() {
    requestAnimationFrame(() => {
        messagesContainer.scrollTo({
            top: messagesContainer.scrollHeight,
            behavior: 'smooth'
        });
    });
}
