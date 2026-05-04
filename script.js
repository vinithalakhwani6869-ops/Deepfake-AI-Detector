/*
   DeepFake Detector AI - Three.js Scene
   Premium wireframe polyhedron
*/

(function () {
  const canvas = document.getElementById('threeCanvas');
  if (!canvas || typeof THREE === 'undefined') return;

  const renderer = new THREE.WebGLRenderer({
    canvas,
    alpha: true,
    antialias: true,
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  function getSize() {
    const rect = canvas.parentElement.getBoundingClientRect();
    return { w: rect.width, h: rect.height };
  }

  const { w, h } = getSize();
  renderer.setSize(w, h);
  renderer.setClearColor(0x000000, 0);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(42, w / h, 0.1, 100);
  camera.position.set(0, 0, 5.5);

  const geoOuter = new THREE.IcosahedronGeometry(1.9, 1);
  const matOuter = new THREE.MeshBasicMaterial({
    color: 0x7B61FF,
    wireframe: true,
    transparent: true,
    opacity: 0.25,
  });
  const meshOuter = new THREE.Mesh(geoOuter, matOuter);
  scene.add(meshOuter);

  const geoInner = new THREE.IcosahedronGeometry(1.25, 1);
  const matInner = new THREE.MeshBasicMaterial({
    color: 0x00E6FF,
    wireframe: true,
    transparent: true,
    opacity: 0.18,
  });
  const meshInner = new THREE.Mesh(geoInner, matInner);
  scene.add(meshInner);

  const geoCore = new THREE.IcosahedronGeometry(0.62, 0);
  const matCore = new THREE.MeshStandardMaterial({
    color: 0x0d0b14,
    emissive: 0x7B61FF,
    emissiveIntensity: 0.7,
    metalness: 0.8,
    roughness: 0.2,
    transparent: true,
    opacity: 0.9,
  });
  const meshCore = new THREE.Mesh(geoCore, matCore);
  scene.add(meshCore);

  scene.add(new THREE.AmbientLight(0xffffff, 0.15));

  const light1 = new THREE.PointLight(0x7B61FF, 2.2, 14);
  light1.position.set(4, 3, 3);
  scene.add(light1);

  const light2 = new THREE.PointLight(0x00E6FF, 1.6, 14);
  light2.position.set(-4, -2, 2);
  scene.add(light2);

  const clock = new THREE.Clock();

  function animate() {
    requestAnimationFrame(animate);
    const t = clock.getElapsedTime();

    meshOuter.rotation.x = t * 0.09;
    meshOuter.rotation.y = t * 0.14;

    meshInner.rotation.x = -t * 0.07;
    meshInner.rotation.y = t * 0.18;

    meshCore.rotation.x = t * 0.05;
    meshCore.rotation.y = t * 0.08;

    const floatY = Math.sin(t * 0.6) * 0.09;
    meshOuter.position.y = floatY;
    meshInner.position.y = floatY;
    meshCore.position.y = floatY;

    matCore.emissiveIntensity = 0.55 + Math.sin(t * 1.1) * 0.15;

    light1.position.x = Math.sin(t * 0.35) * 5;
    light1.position.z = Math.cos(t * 0.35) * 5;
    light2.position.x = Math.cos(t * 0.28) * 5;
    light2.position.z = Math.sin(t * 0.28) * 5;

    renderer.render(scene, camera);
  }

  animate();

  let resizeTimer;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      const { w: nw, h: nh } = getSize();
      renderer.setSize(nw, nh);
      camera.aspect = nw / nh;
      camera.updateProjectionMatrix();
    }, 80);
  });
})();

const hamburger = document.getElementById('hamburger');
const mobileMenu = document.getElementById('mobileMenu');
const tryNowBtn = document.getElementById('tryNowBtn');
const mobileTryNowBtn = document.getElementById('mobileTryNowBtn');
const analysisEngine = document.getElementById('analysis-engine');
const heroUploadBtn = document.getElementById('heroUploadBtn');
const viewDemoBtn = document.getElementById('viewDemoBtn');
const demoModal = document.getElementById('demoModal');
const closeModalBtn = document.getElementById('closeModalBtn');
const demoVideo = document.getElementById('demoVideo');
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const dropTitle = document.getElementById('dropTitle');
const analysisStatus = document.getElementById('analysisStatus');
const navAnchors = document.querySelectorAll('.nav-links a, .mobile-menu a');
const DEMO_VIDEO_URL = 'https://www.youtube.com/embed/9No-FiEInLA?autoplay=1&rel=0';

if (hamburger && mobileMenu) {
  hamburger.addEventListener('click', () => {
    mobileMenu.classList.toggle('open');
  });
}

function scrollToAnalysis() {
  if (analysisEngine) {
    analysisEngine.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}

[tryNowBtn, mobileTryNowBtn].forEach((button) => {
  if (!button) return;
  button.addEventListener('click', () => {
    if (mobileMenu) {
      mobileMenu.classList.remove('open');
    }
    scrollToAnalysis();
  });
});

if (heroUploadBtn) {
  heroUploadBtn.addEventListener('click', () => {
    scrollToAnalysis();
    if (fileInput) {
      fileInput.click();
    }
  });
}

navAnchors.forEach((anchor) => {
  anchor.addEventListener('click', (event) => {
    const href = anchor.getAttribute('href');
    if (!href || !href.startsWith('#')) return;

    const target = document.querySelector(href);
    if (!target) return;

    event.preventDefault();
    if (mobileMenu) {
      mobileMenu.classList.remove('open');
    }
    target.scrollIntoView({ behavior: 'smooth', block: 'start' });
  });
});

function openModal() {
  if (!demoModal || !demoVideo) return;
  demoModal.classList.add('is-open');
  demoModal.setAttribute('aria-hidden', 'false');
  demoVideo.src = DEMO_VIDEO_URL;
  document.body.style.overflow = 'hidden';
}

function closeModal() {
  if (!demoModal || !demoVideo) return;
  demoModal.classList.remove('is-open');
  demoModal.setAttribute('aria-hidden', 'true');
  demoVideo.src = '';
  document.body.style.overflow = '';
}

if (viewDemoBtn) {
  viewDemoBtn.addEventListener('click', openModal);
}

if (closeModalBtn) {
  closeModalBtn.addEventListener('click', closeModal);
}

if (demoModal) {
  demoModal.addEventListener('click', (event) => {
    if (event.target === demoModal) {
      closeModal();
    }
  });
}

document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape' && demoModal && demoModal.classList.contains('is-open')) {
    closeModal();
  }
});

function setStatus(message, state) {
  if (!analysisStatus) return;
  analysisStatus.classList.remove('is-loading', 'is-success', 'is-error');
  if (state) {
    analysisStatus.classList.add(state);
  }

  const statusLabel = analysisStatus.querySelector('.status-label');
  if (statusLabel) {
    statusLabel.textContent = message;
  }
}

async function uploadFile(file) {
  if (!file) return;

  if (dropTitle) {
    dropTitle.textContent = `Selected: ${file.name}`;
  }

  setStatus('Analyzing...', 'is-loading');

  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch('http://localhost:3000/upload', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error('Upload failed');
    }

    const data = await response.json();
    setStatus(`Result: ${data.result}`, 'is-success');
  } catch (error) {
    setStatus('Unable to analyze right now. Please make sure the backend server is running.', 'is-error');
  }
}

if (dropZone && fileInput) {
  dropZone.addEventListener('click', () => {
    fileInput.click();
  });

  fileInput.addEventListener('change', (event) => {
    const file = event.target.files && event.target.files[0];
    uploadFile(file);
  });

  dropZone.addEventListener('dragover', (event) => {
    event.preventDefault();
    dropZone.classList.add('is-active');
  });

  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('is-active');
  });

  dropZone.addEventListener('drop', (event) => {
    event.preventDefault();
    dropZone.classList.remove('is-active');
    const file = event.dataTransfer.files && event.dataTransfer.files[0];
    uploadFile(file);
  });
}
