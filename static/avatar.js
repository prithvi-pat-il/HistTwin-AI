/**
 * AvatarController - High-Performance 60FPS 3D Character Controller
 * Pure Character Actions, Natural Gestures & Skeletal Motion (No Lip Sync)
 */
export class AvatarController {
    constructor(containerId) {
        if (typeof containerId === 'string') {
            this.container = document.getElementById(containerId);
        } else {
            this.container = containerId;
        }
        if (!this.container) {
            console.error(`Container element not found.`);
            return;
        }

        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.model = null;
        this.mixer = null;
        this.clock = new THREE.Clock();

        // Animation Actions
        this.actions = {};
        this.activeAction = null;
        this.idleActionName = 'Idle';
        this.talkingActionName = null;

        // Bone references for skeletal poses and smooth procedural gestures
        this.bones = {};
        this.headNode = null;
        this.neckNode = null;

        // Morph Target Meshes for natural expressions and blinking
        this.morphMeshes = [];

        // Speech & State
        this.isSpeaking = false;
        this.speechTime = 0;
        this.gestureTime = 0;
        this.blinkTimer = 0;
        this.blinkDuration = 0.16;
        this.isBlinking = false;

        // Motion Data Sync
        this.currentMotion = null;
        this.currentAudio = null;

        // Mouse & Head Tracking
        this.mouse = { x: 0, y: 0 };
        this.targetRotation = { x: 0, y: 0 };
        this.currentHeadRot = { x: 0, y: 0 };

        // Pre-allocated math objects for Zero GC in render loop
        this._tempVec1 = new THREE.Vector3();
        this._tempQuat = new THREE.Quaternion();

        this.init();
    }

    init() {
        // 1. Create Scene
        this.scene = new THREE.Scene();

        // 2. Create Camera
        const aspect = (this.container.clientWidth || 384) / (this.container.clientHeight || 500);
        this.camera = new THREE.PerspectiveCamera(42, aspect, 0.1, 50);
        this.camera.position.set(0, 1.35, 2.2);

        // 3. Create GPU-Optimized WebGL Renderer
        this.renderer = new THREE.WebGLRenderer({
            antialias: true,
            alpha: true,
            powerPreference: "high-performance",
            precision: "mediump"
        });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
        this.renderer.shadowMap.enabled = false;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.05;

        // Clear previous canvas
        this.container.innerHTML = '';
        this.container.appendChild(this.renderer.domElement);

        // 4. Studio Lighting Setup
        const ambientLight = new THREE.AmbientLight(0xffffff, 1.2);
        this.scene.add(ambientLight);

        const keyLight = new THREE.DirectionalLight(0xfffaed, 1.4);
        keyLight.position.set(1.8, 3.5, 3.0);
        this.scene.add(keyLight);

        const fillLight = new THREE.DirectionalLight(0x7ac9ff, 0.7);
        fillLight.position.set(-2.0, 1.5, -1.5);
        this.scene.add(fillLight);

        // 5. Event Listeners
        this._onWindowResize = this.onWindowResize.bind(this);
        this._onMouseMove = this.onMouseMove.bind(this);

        window.addEventListener('resize', this._onWindowResize, { passive: true });
        window.addEventListener('mousemove', this._onMouseMove, { passive: true });

        // 6. Default Model Loading
        this.loadModel('girl_talking.glb');

        // 7. Start High-Performance Render Loop
        this.animate = this.animate.bind(this);
        requestAnimationFrame(this.animate);
    }

    showLoading(show) {
        let loaderEl = this.container.querySelector('.avatar-loader');
        if (show) {
            if (!loaderEl) {
                loaderEl = document.createElement('div');
                loaderEl.className = 'avatar-loader';
                loaderEl.innerHTML = '<div class="loader"></div><p style="margin-top:10px; font-size:13px; color: #ececf1; font-weight:500;">Loading 3D Guide...</p>';
                loaderEl.style.cssText = "position:absolute; top:50%; left:50%; transform:translate(-50%, -50%); text-align:center; z-index:10; pointer-events:none;";
                this.container.appendChild(loaderEl);
            }
        } else if (loaderEl) {
            loaderEl.remove();
        }
    }

    async loadModel(url) {
        this.showLoading(true);

        const oldErr = this.container.querySelector('.avatar-error-msg');
        if (oldErr) oldErr.remove();

        const guideCard = document.getElementById('cultural-guide-card');
        if (guideCard) guideCard.style.display = 'none';
        if (this.renderer && this.renderer.domElement) {
            this.renderer.domElement.style.display = 'block';
        }

        // Clean up previous model
        if (this.model) {
            this.scene.remove(this.model);
            if (this.mixer) {
                this.mixer.stopAllAction();
                this.mixer.uncacheRoot(this.model);
            }
            this.model = null;
            this.mixer = null;
            this.actions = {};
            this.morphMeshes = [];
            this.bones = {};
            this.headNode = null;
            this.neckNode = null;
        }

        let modelUrl = url;
        if (!url.startsWith('http') && !url.startsWith('/')) {
            modelUrl = url;
        }

        const loader = new THREE.GLTFLoader();

        try {
            const gltf = await new Promise((resolve, reject) => {
                loader.load(modelUrl, resolve, undefined, () => {
                    loader.load(url, resolve, undefined, reject);
                });
            });

            this.model = gltf.scene;
            this.scene.add(this.model);

            // Compute standard bounding box and normalize scale
            this.model.updateMatrixWorld(true);
            const box = new THREE.Box3().setFromObject(this.model);
            const size = box.getSize(new THREE.Vector3());

            if (size.y > 0) {
                const targetHeight = 1.6;
                const scale = targetHeight / size.y;
                this.model.scale.set(scale, scale, scale);
                this.baseModelScaleY = scale;

                this.model.updateMatrixWorld(true);
                box.setFromObject(this.model);
                const minY = box.min.y;
                this.baseModelPosY = -minY;
                this.model.position.set(0, this.baseModelPosY, 0);
            } else {
                this.model.position.set(0, 0, 0);
                this.model.scale.set(1, 1, 1);
                this.baseModelPosY = 0;
            }

            // Ideal camera framing
            this.camera.position.set(0, 1.35, 2.2);
            this.camera.lookAt(new THREE.Vector3(0, 1.25, 0));

            // Traverse and map meshes and bones
            this.model.traverse((child) => {
                if (child.isMesh) {
                    child.frustumCulled = true;
                    if (child.morphTargetDictionary && child.morphTargetInfluences) {
                        this.morphMeshes.push(child);
                    }
                }

                if (child.isBone) {
                    const n = child.name.toLowerCase();
                    if (n.includes('hips') || n.includes('pelvis')) this.bones[0] = child;
                    if (n.includes('spine') && !n.includes('spine1') && !n.includes('spine2')) this.bones[3] = child;
                    if (n.includes('spine1')) this.bones[6] = child;
                    if (n.includes('spine2') || n.includes('chest')) this.bones[9] = child;

                    if (n.includes('neck')) {
                        this.bones[12] = child;
                        this.neckNode = child;
                    }
                    if (n.includes('head') && !n.includes('top')) {
                        this.bones[15] = child;
                        this.headNode = child;
                    }

                    if (n.includes('leftshoulder') || n.includes('left_shoulder') || n.includes('shoulder.l')) this.bones[13] = child;
                    if (n.includes('rightshoulder') || n.includes('right_shoulder') || n.includes('shoulder.r')) this.bones[14] = child;
                    if (n.includes('leftarm') || n.includes('left_arm') || n.includes('upperarm.l') || (n.includes('arm') && n.includes('l') && !n.includes('fore'))) this.bones[16] = child;
                    if (n.includes('rightarm') || n.includes('right_arm') || n.includes('upperarm.r') || (n.includes('arm') && n.includes('r') && !n.includes('fore'))) this.bones[17] = child;
                    if (n.includes('leftforearm') || n.includes('left_forearm') || n.includes('lowerarm.l') || n.includes('forearm.l')) this.bones[18] = child;
                    if (n.includes('rightforearm') || n.includes('right_forearm') || n.includes('lowerarm.r') || n.includes('forearm.r')) this.bones[19] = child;
                    if (n.includes('lefthand') || n.includes('left_hand') || n.includes('hand.l')) this.bones[20] = child;
                    if (n.includes('righthand') || n.includes('right_hand') || n.includes('hand.r')) this.bones[21] = child;
                }
            });

            // Set up Animation Mixer
            if (gltf.animations && gltf.animations.length > 0) {
                this.mixer = new THREE.AnimationMixer(this.model);

                gltf.animations.forEach((clip) => {
                    const action = this.mixer.clipAction(clip);
                    this.actions[clip.name] = action;
                });

                const allNames = Object.keys(this.actions);
                
                // Identify Idle clip
                const idleMatches = ['idle', 'Idle', 'IDLE', 'standing', 'Standing', 'breathe', 'Breathe'];
                const matchedIdle = allNames.find(name => idleMatches.some(m => name.toLowerCase().includes(m.toLowerCase())));
                this.idleActionName = matchedIdle || allNames[0];

                // Identify Talking/Gesture clip
                const talkMatches = ['talk', 'talking', 'speak', 'gesture', 'wave', 'explain'];
                const matchedTalk = allNames.find(name => talkMatches.some(m => name.toLowerCase().includes(m.toLowerCase())));
                this.talkingActionName = matchedTalk || null;

                if (this.actions[this.idleActionName]) {
                    this.playAction(this.idleActionName);
                }
            }

            // Natural default arm posture if not overridden by clip
            if (this.bones[16] && this.bones[17]) {
                this.bones[16].rotation.z = -0.32;
                this.bones[16].rotation.x = 0.38;
                this.bones[17].rotation.z = 0.32;
                this.bones[17].rotation.x = 0.38;
                if (this.bones[18]) this.bones[18].rotation.y = 0.45;
                if (this.bones[19]) this.bones[19].rotation.y = -0.45;
            }

        } catch (error) {
            console.error("Failed to load 3D GLTF model:", error);
            const errEl = document.createElement('div');
            errEl.className = 'avatar-error-msg';
            errEl.style.cssText = "padding: 18px; color:#ff5252; text-align:center; position:absolute; top:50%; left:50%; transform:translate(-50%, -50%); background:rgba(20,20,25,0.9); border-radius:12px; border:1px solid rgba(255,80,80,0.3);";
            errEl.innerHTML = `Unable to load 3D model.<br><small style="color:#aaa;">(${error.message || 'Check connection'})</small>`;
            this.container.appendChild(errEl);
        } finally {
            this.showLoading(false);
        }
    }

    playAction(name, duration = 0.4) {
        const action = this.actions[name];
        if (!action) return;

        if (this.activeAction === action) return;

        if (this.activeAction) {
            this.activeAction.fadeOut(duration);
        }

        action.reset().fadeIn(duration).play();
        this.activeAction = action;
    }

    setMorphInfluence(targetName, value) {
        for (let i = 0; i < this.morphMeshes.length; i++) {
            const mesh = this.morphMeshes[i];
            const dict = mesh.morphTargetDictionary;
            const inf = mesh.morphTargetInfluences;
            if (!dict || !inf) continue;

            let idx = dict[targetName];
            if (idx === undefined) {
                const keys = Object.keys(dict);
                for (let k = 0; k < keys.length; k++) {
                    if (keys[k].toLowerCase() === targetName.toLowerCase()) {
                        idx = dict[keys[k]];
                        break;
                    }
                }
            }

            if (idx !== undefined) {
                inf[idx] = value;
            }
        }
    }

    setExpression(expressionName) {
        const shapes = ['Surprised', 'Sad', 'Angry', 'mouthSmile', 'Smile'];
        shapes.forEach(s => this.setMorphInfluence(s, 0));

        switch (expressionName) {
            case 'happy':
                this.setMorphInfluence('mouthSmile', 0.75);
                this.setMorphInfluence('Smile', 0.75);
                break;
            case 'surprised':
                this.setMorphInfluence('Surprised', 0.85);
                break;
            case 'sad':
                this.setMorphInfluence('Sad', 0.65);
                break;
            default:
                break;
        }
    }

    playMotion(audio, motionData) {
        this.currentMotion = motionData;
        this.currentAudio = audio;
        this.isSpeaking = true;
        this.speechTime = 0;

        if (this.talkingActionName && this.actions[this.talkingActionName]) {
            this.playAction(this.talkingActionName, 0.4);
        }

        const stopMotion = () => {
            this.stopSpeaking();
        };

        if (audio) {
            audio.addEventListener('ended', stopMotion, { once: true });
            audio.addEventListener('error', stopMotion, { once: true });
        }
    }

    startSpeaking(audioElement) {
        this.isSpeaking = true;
        this.speechTime = 0;
        this.gestureTime = 0;

        if (audioElement) {
            this.currentAudio = audioElement;
        }

        const guideCard = document.getElementById('cultural-guide-card');
        if (guideCard) guideCard.classList.add('is-speaking');

        // Play talking action or gesture if available
        if (this.talkingActionName && this.actions[this.talkingActionName]) {
            this.playAction(this.talkingActionName, 0.35);
        } else {
            const gestureClips = ['Wave', 'Yes', 'ThumbsUp', 'Standing'];
            const found = gestureClips.find(g => this.actions[g]);
            if (found) {
                this.playAction(found, 0.35);
            }
        }
    }

    stopSpeaking() {
        this.isSpeaking = false;
        this.currentAudio = null;
        this.currentMotion = null;

        const guideCard = document.getElementById('cultural-guide-card');
        if (guideCard) guideCard.classList.remove('is-speaking');

        if (this.actions && this.actions[this.idleActionName]) {
            this.playAction(this.idleActionName, 0.4);
        }
    }

    onMouseMove(event) {
        this.mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
        this.mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

        this.targetRotation.y = this.mouse.x * 0.32;
        this.targetRotation.x = this.mouse.y * 0.18;
    }

    onWindowResize() {
        if (!this.container || !this.renderer || !this.camera) return;

        const width = this.container.clientWidth || 384;
        const height = this.container.clientHeight || 500;

        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();

        this.renderer.setSize(width, height);
    }

    animate() {
        requestAnimationFrame(this.animate);

        // Skip rendering if container is hidden or off-screen to save 100% GPU/CPU
        if (!this.container || this.container.clientWidth === 0 || this.container.offsetParent === null) {
            return;
        }

        // Cap delta time to prevent stutter jumps
        const delta = Math.min(this.clock.getDelta(), 0.05);
        const time = this.clock.getElapsedTime();

        // 1. Update Animation Mixer
        if (this.mixer) {
            this.mixer.update(delta);
        }

        // 2. Skeletal Motion Stream or Procedural Conversational Actions
        if (this.isSpeaking) {
            this.speechTime += delta;
            this.gestureTime += delta;

            // Mode A: PantoMatrix Motion Stream (If active)
            if (this.currentMotion && this.currentAudio && !this.currentAudio.paused) {
                const currentTime = this.currentAudio.currentTime;
                const frameIndex = Math.floor(currentTime * 30);

                if (frameIndex < this.currentMotion.frameCount) {
                    const pose = this.currentMotion.poses[frameIndex];

                    for (const jointIndex in this.bones) {
                        const bone = this.bones[jointIndex];
                        const i = parseInt(jointIndex) * 3;
                        if (i + 2 < pose.length) {
                            const rx = pose[i];
                            const ry = pose[i + 1];
                            const rz = pose[i + 2];

                            const angle = Math.sqrt(rx * rx + ry * ry + rz * rz);
                            if (angle > 0.0001) {
                                this._tempVec1.set(rx, ry, rz).normalize();
                                this._tempQuat.setFromAxisAngle(this._tempVec1, angle);
                                bone.quaternion.slerp(this._tempQuat, 0.22);
                            }
                        }
                    }
                }
            } else if (this.model && !this.activeAction) {
                // Mode B: Procedural Conversational Gestures & Natural Head Articulation
                const nod = Math.sin(time * 3.5) * 0.04;
                const sway = Math.cos(time * 2.2) * 0.03;

                if (this.headNode) {
                    this.headNode.rotation.x = nod;
                    this.headNode.rotation.y = sway;
                }

                const leftArm = this.bones[16];
                const rightArm = this.bones[17];
                const leftForearm = this.bones[18];
                const rightForearm = this.bones[19];

                if (leftArm && rightArm) {
                    const armG = Math.sin(this.gestureTime * 2.8) * 0.12;
                    const armS = Math.cos(this.gestureTime * 2.0) * 0.08;

                    leftArm.rotation.z = -0.32 + armG;
                    leftArm.rotation.x = 0.38 + armS;

                    rightArm.rotation.z = 0.32 - armG;
                    rightArm.rotation.x = 0.38 - armS;

                    if (leftForearm) leftForearm.rotation.y = 0.45 + Math.sin(this.gestureTime * 3.2) * 0.15;
                    if (rightForearm) rightForearm.rotation.y = -0.45 - Math.cos(this.gestureTime * 3.2) * 0.15;
                }
            }
        }

        // 3. Smooth Damped Mouse / Head Gaze Tracking
        if (this.model) {
            this.currentHeadRot.x = THREE.MathUtils.damp(this.currentHeadRot.x, this.targetRotation.x, 4.0, delta);
            this.currentHeadRot.y = THREE.MathUtils.damp(this.currentHeadRot.y, this.targetRotation.y, 4.0, delta);

            if (this.headNode) {
                this.headNode.rotation.y = this.currentHeadRot.y * 0.8;
                this.headNode.rotation.x = this.currentHeadRot.x * 0.6;
            } else {
                this.model.rotation.y = this.currentHeadRot.y * 0.4;
            }

            // Natural Subtle Idle Breathing
            if (!this.isSpeaking) {
                const breathing = Math.sin(time * 2.0) * 0.0025;
                const baseY = (this.baseModelPosY !== undefined) ? this.baseModelPosY : 0;
                this.model.position.y = baseY + breathing;
            }
        }

        // 4. Procedural Eye Blinking (Morph Target based)
        this.blinkTimer += delta;
        if (!this.isBlinking && this.blinkTimer > 3.5) {
            this.isBlinking = true;
            this.blinkTimer = 0;
        }

        if (this.isBlinking) {
            if (this.blinkTimer < this.blinkDuration) {
                const progress = this.blinkTimer / this.blinkDuration;
                const weight = Math.sin(progress * Math.PI);
                this.setMorphInfluence('eyesClosed', weight);
                this.setMorphInfluence('eyeBlinkLeft', weight);
                this.setMorphInfluence('eyeBlinkRight', weight);
            } else {
                this.isBlinking = false;
                this.setMorphInfluence('eyesClosed', 0);
                this.setMorphInfluence('eyeBlinkLeft', 0);
                this.setMorphInfluence('eyeBlinkRight', 0);
                this.blinkTimer = 0;
            }
        }

        // 5. Render Scene
        if (this.renderer && this.scene && this.camera) {
            this.renderer.render(this.scene, this.camera);
        }
    }
}
