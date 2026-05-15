/* ============================================================
   DeepFake Detector AI — script.js
   ============================================================
   TWO INDEPENDENT CONCERNS IN THIS FILE:

   ① FORM 1 — Image Detection (uploadFile)
      • No <form> element involved — drop zone is a plain <div>
      • Page refresh is impossible by design (no native submission)
      • fetch() sends FormData to FastAPI POST /detect
      • Result panel updated in-place with verdict / confidence / filename

   ② FORM 2 — Contact Form (contactForm submit handler)
      • event.preventDefault() called first → stops any reload
      • emailjs.sendForm() sends email directly from the browser
      • Inline feedback shown on success or failure — no page change
   ============================================================ */


/* ─────────────────────────────────────────────────────────────
   THREE.JS SCENE — completely unchanged
   ───────────────────────────────────────────────────────────── */
(function () {
  const canvas = document.getElementById('threeCanvas');
  if (!canvas || typeof THREE === 'undefined') return;

  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  function getSize() {
    const rect = canvas.parentElement.getBoundingClientRect();
    return { w: rect.width, h: rect.height };
  }

  const { w, h } = getSize();
  renderer.setSize(w, h);
  renderer.setClearColor(0x000000, 0);

  const scene  = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(42, w / h, 0.1, 100);
  camera.position.set(0, 0, 5.5);

  const geoOuter = new THREE.IcosahedronGeometry(1.9, 1);
  const matOuter = new THREE.MeshBasicMaterial({ color: 0x7B61FF, wireframe: true, transparent: true, opacity: 0.25 });
  const meshOuter = new THREE.Mesh(geoOuter, matOuter);
  scene.add(meshOuter);

  const geoInner = new THREE.IcosahedronGeometry(1.25, 1);
  const matInner = new THREE.MeshBasicMaterial({ color: 0x00E6FF, wireframe: true, transparent: true, opacity: 0.18 });
  const meshInner = new THREE.Mesh(geoInner, matInner);
  scene.add(meshInner);

  const geoCore = new THREE.IcosahedronGeometry(0.62, 0);
  const matCore = new THREE.MeshStandardMaterial({
    color: 0x0d0b14, emissive: 0x7B61FF, emissiveIntensity: 0.7,
    metalness: 0.8, roughness: 0.2, transparent: true, opacity: 0.9,
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
    meshInner.rotation.y =  t * 0.18;
    meshCore.rotation.x  =  t * 0.05;
    meshCore.rotation.y  =  t * 0.08;

    const floatY = Math.sin(t * 0.6) * 0.09;
    meshOuter.position.y = floatY;
    meshInner.position.y = floatY;
    meshCore.position.y  = floatY;

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


/* ─────────────────────────────────────────────────────────────
   EMAILJS INITIALISATION
   ─────────────────────────────────────────────────────────────
   emailjs.init() must be called once before any sendForm() call.
   The public key identifies your EmailJS account.
   It is safe to expose in frontend code — it cannot send email
   on its own without a valid Service ID + Template ID pair.
   ───────────────────────────────────────────────────────────── */
(function initEmailJS() {
  if (typeof emailjs === 'undefined') {
    console.warn('[EmailJS] SDK not loaded. Contact form will not send email.');
    return;
  }
  emailjs.init('mJyzC0Km38LNWzoDQ');  // ← your Public Key
})();


/* ─────────────────────────────────────────────────────────────
   ELEMENT REFERENCES — unchanged from original
   ───────────────────────────────────────────────────────────── */
const hamburger       = document.getElementById('hamburger');
const mobileMenu      = document.getElementById('mobileMenu');
const tryNowBtn       = document.getElementById('tryNowBtn');
const mobileTryNowBtn = document.getElementById('mobileTryNowBtn');
const analysisEngine  = document.getElementById('analysis-engine');
const heroUploadBtn   = document.getElementById('heroUploadBtn');
const viewDemoBtn     = document.getElementById('viewDemoBtn');
const demoModal       = document.getElementById('demoModal');
const closeModalBtn   = document.getElementById('closeModalBtn');
const demoVideo       = document.getElementById('demoVideo');
const dropZone        = document.getElementById('dropZone');
const fileInput       = document.getElementById('fileInput');
const dropTitle       = document.getElementById('dropTitle');
const analysisStatus  = document.getElementById('analysisStatus');
const navAnchors      = document.querySelectorAll('.nav-links a, .mobile-menu a');

/* ── Result panel elements (Form 1) */
const resultPanel      = document.getElementById('resultPanel');
const resultLoading    = document.getElementById('resultLoading');
const resultSuccess    = document.getElementById('resultSuccess');
const resultError      = document.getElementById('resultError');
const resultVerdict    = document.getElementById('resultVerdict');
const resultConfidence = document.getElementById('resultConfidence');
const resultFilename   = document.getElementById('resultFilename');
const resultErrorText  = document.getElementById('resultErrorText');

/* ── Contact form elements (Form 2) */
const contactForm      = document.getElementById('contactForm');
const contactName      = document.getElementById('contactName');
const contactEmail     = document.getElementById('contactEmail');
const contactMessage   = document.getElementById('contactMessage');
const contactSubmitBtn = document.getElementById('contactSubmitBtn');
const contactFeedback  = document.getElementById('contactFeedback');

const DEMO_VIDEO_URL   = 'https://www.youtube.com/embed/9No-FiEInLA?autoplay=1&rel=0';

/* ── File validation constants (Form 1) */
const ACCEPTED_EXTS  = ['.jpg', '.jpeg', '.png', '.webp'];
const ACCEPTED_TYPES = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
const MAX_BYTES      = 20 * 1024 * 1024; // 20 MB


/* ─────────────────────────────────────────────────────────────
   NAVIGATION — unchanged
   ───────────────────────────────────────────────────────────── */
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

[tryNowBtn, mobileTryNowBtn].forEach((btn) => {
  if (!btn) return;
  btn.addEventListener('click', () => {
    if (mobileMenu) mobileMenu.classList.remove('open');
    scrollToAnalysis();
  });
});

if (heroUploadBtn) {
  heroUploadBtn.addEventListener('click', () => {
    scrollToAnalysis();
    if (fileInput) fileInput.click();
  });
}

navAnchors.forEach((anchor) => {
  anchor.addEventListener('click', (e) => {
    const href = anchor.getAttribute('href');
    if (!href || !href.startsWith('#')) return;
    const target = document.querySelector(href);
    if (!target) return;
    e.preventDefault();
    if (mobileMenu) mobileMenu.classList.remove('open');
    target.scrollIntoView({ behavior: 'smooth', block: 'start' });
  });
});


/* ─────────────────────────────────────────────────────────────
   DEMO MODAL — unchanged
   ───────────────────────────────────────────────────────────── */
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

if (viewDemoBtn)   viewDemoBtn.addEventListener('click', openModal);
if (closeModalBtn) closeModalBtn.addEventListener('click', closeModal);

if (demoModal) {
  demoModal.addEventListener('click', (e) => {
    if (e.target === demoModal) closeModal();
  });
}

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && demoModal && demoModal.classList.contains('is-open')) {
    closeModal();
  }
});


/* ─────────────────────────────────────────────────────────────
   LEGACY STATUS HELPER — kept so existing CSS states still work
   ───────────────────────────────────────────────────────────── */
function setStatus(message, state) {
  if (!analysisStatus) return;
  analysisStatus.classList.remove('is-loading', 'is-success', 'is-error');
  if (state) analysisStatus.classList.add(state);
  const lbl = analysisStatus.querySelector('.status-label');
  if (lbl) lbl.textContent = message;
}


/* ═══════════════════════════════════════════════════════════════
   FORM 1 — IMAGE DETECTION HELPERS
   ═══════════════════════════════════════════════════════════════

   showLoadingState()
     Reveals the result panel and shows only the spinner.

   showResult(result, confidence, filename)
     Reveals the result panel with verdict badge, confidence, filename.
     Adds colour modifier class depending on "Real" vs "Fake".

   showDetectionError(message)
     Reveals the result panel with the error state and given message.
═══════════════════════════════════════════════════════════════ */

/** Reset panel to blank slate, then make it visible. */
function _openResultPanel() {
  if (!resultPanel) return;
  // Hide all three inner states
  [resultLoading, resultSuccess, resultError].forEach((el) => {
    if (el) el.style.display = 'none';
  });
  // Remove any colour modifier left over from a previous detection
  resultPanel.classList.remove('result-panel--fake', 'result-panel--real', 'result-panel--error');
  resultPanel.style.display = 'block';
}

function showLoadingState() {
  _openResultPanel();
  if (resultLoading) resultLoading.style.display = 'flex';
}

/**
 * @param {string} result      "Real" or "Fake"
 * @param {number} confidence  0–100
 * @param {string} filename
 */
function showResult(result, confidence, filename) {
  _openResultPanel();

  const isFake = result.toLowerCase() === 'fake';

  // Colour-code the panel border
  resultPanel.classList.add(isFake ? 'result-panel--fake' : 'result-panel--real');

  // Verdict badge
  if (resultVerdict) {
    resultVerdict.textContent = result.toUpperCase();
    resultVerdict.className   = 'result-verdict ' + (isFake ? 'result-verdict--fake' : 'result-verdict--real');
  }

  if (resultConfidence) resultConfidence.textContent = confidence.toFixed(2) + '%';
  if (resultFilename)   resultFilename.textContent   = filename || '—';

  if (resultSuccess) resultSuccess.style.display = 'flex';

  // Bring the panel into view without jarring the user
  resultPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/** @param {string} message Human-readable error description. */
function showDetectionError(message) {
  _openResultPanel();
  resultPanel.classList.add('result-panel--error');
  if (resultErrorText) resultErrorText.textContent = message;
  if (resultError)     resultError.style.display   = 'flex';
  resultPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}


/* ═══════════════════════════════════════════════════════════════
   FORM 1 — uploadFile()
   ═══════════════════════════════════════════════════════════════

   WHY THERE IS NO PAGE REFRESH:
     The drop zone is a plain <div>, not a <form>.
     The hidden <input type="file"> is triggered only by fileInput.click()
     inside JS event handlers — it never participates in a form submission.
     fetch() handles the network call asynchronously.
     Nothing here can cause a browser navigation event.

   FLOW:
     validate type → validate size → show spinner →
     POST FormData → parse response → showResult or showDetectionError
═══════════════════════════════════════════════════════════════ */
async function uploadFile(file) {
  if (!file) return;

  /* 1. Validate file extension -------------------------------------------- */
  const ext  = '.' + file.name.split('.').pop().toLowerCase();
  const mime = (file.type || '').toLowerCase();

  if (!ACCEPTED_EXTS.includes(ext)) {
    showDetectionError(
      `"${ext}" is not a supported format. Please upload a JPG, PNG, or WebP image.`
    );
    setStatus('Unsupported file type.', 'is-error');
    return;
  }

  /* 2. Validate file size -------------------------------------------------- */
  if (file.size > MAX_BYTES) {
    showDetectionError(
      `File is ${(file.size / 1024 / 1024).toFixed(1)} MB — the maximum is 20 MB. Please use a smaller image.`
    );
    setStatus('File too large.', 'is-error');
    return;
  }

  /* 3. Update drop zone title ---------------------------------------------- */
  if (dropTitle) dropTitle.textContent = `Selected: ${file.name}`;

  /* 4. Show loading state immediately ------------------------------------- */
  showLoadingState();
  setStatus('Analysing…', 'is-loading');

  /* 5. POST to FastAPI backend --------------------------------------------- */
  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch('http://127.0.0.1:8000/detect', {
      method: 'POST',
      body: formData,
    });

    /* 6. Handle non-2xx HTTP status codes ---------------------------------- */
    if (!response.ok) {
      let detail = `Server returned HTTP ${response.status}.`;
      try {
        const body = await response.json();
        if (body && body.detail) detail = body.detail;
      } catch (_) { /* response body was not JSON */ }

      if (response.status === 415) {
        showDetectionError('The server rejected this file type. Please upload a JPG, PNG, or WebP image.');
      } else if (response.status === 422) {
        showDetectionError('The image could not be processed. It may be corrupt or not a valid image file.');
      } else if (response.status === 413) {
        showDetectionError('The file is too large for the server. Please try a smaller image.');
      } else if (response.status === 500) {
        showDetectionError('The server encountered an internal error. Check that the FastAPI server is running correctly.');
      } else {
        showDetectionError(detail);
      }

      setStatus(detail, 'is-error');
      return;
    }

    /* 7. Parse the JSON response ------------------------------------------ */
    const data = await response.json();

    if (!data || typeof data.result === 'undefined') {
      showDetectionError('Received an unexpected response from the server. Please try again.');
      setStatus('Unexpected server response.', 'is-error');
      return;
    }

    /* 8. Display the result ----------------------------------------------- */
    showResult(data.result, data.confidence ?? 0, data.filename ?? file.name);
    setStatus(
      `Result: ${data.result} — ${(data.confidence ?? 0).toFixed(2)}% confidence`,
      'is-success'
    );

  } catch (err) {
    /*
      fetch() itself threw — the backend is offline, unreachable,
      or the browser blocked the request due to CORS.
    */
    const msg = err instanceof TypeError
      ? 'Cannot reach the backend. Make sure the FastAPI server is running at http://127.0.0.1:8000 and try again.'
      : `Request failed: ${err.message}`;

    showDetectionError(msg);
    setStatus(msg, 'is-error');
  } finally {
    /*
      Reset the file input value so the user can re-upload the same
      file if needed. Without this, the "change" event won't fire
      a second time for the same filename.
    */
    if (fileInput) fileInput.value = '';
  }
}


/* ─────────────────────────────────────────────────────────────
   FORM 1 — Drag-and-drop + click handlers — unchanged structure
   ───────────────────────────────────────────────────────────── */
if (dropZone && fileInput) {

  // Click anywhere on the drop zone to open the file picker
  dropZone.addEventListener('click', () => {
    fileInput.click();
  });

  // File picker selection
  fileInput.addEventListener('change', (e) => {
    const file = e.target.files && e.target.files[0];
    if (file) uploadFile(file);
  });

  // Drag over — highlight the zone
  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('is-active');
  });

  // Drag leave — remove highlight
  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('is-active');
  });

  // Drop — extract file and upload
  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('is-active');
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) uploadFile(file);
  });
}


/* ═══════════════════════════════════════════════════════════════
   FORM 2 — CONTACT FORM via EmailJS
   ═══════════════════════════════════════════════════════════════

   HOW PAGE REFRESH IS PREVENTED:
     The submit event listener calls event.preventDefault() as its
     very first action. This cancels the browser's default form
     submission (which would cause a GET/POST navigation + reload).
     After that, JS drives everything.

   HOW EMAILJS WORKS:
     emailjs.sendForm(SERVICE_ID, TEMPLATE_ID, formElement)
       • Reads all <input> and <textarea> elements inside the form
       • Maps their name="" attributes to {{variables}} in your template
       • POSTs to EmailJS servers over HTTPS (no backend needed)
       • Returns a Promise that resolves on success or rejects on failure

     Your template variables:
       {{from_name}}   ← <input name="from_name">
       {{from_email}}  ← <input name="from_email">
       {{message}}     ← <textarea name="message">

   VALIDATION RULES:
     • Name    — required, minimum 2 characters
     • Email   — required, must match basic email pattern
     • Message — required, minimum 10 characters

   STATES:
     Sending  — button disabled + text "Sending…"
     Success  — green feedback message, fields cleared, button re-enabled
     Error    — orange feedback message, button re-enabled for retry
═══════════════════════════════════════════════════════════════ */

/* ── EmailJS credentials ── */
const EMAILJS_SERVICE_ID  = 'service_8c9ygmc';
const EMAILJS_TEMPLATE_ID = 'template_8dn52rc';

/** Show the inline feedback div with a message and colour state. */
function setContactFeedback(message, type /* 'success' | 'error' */) {
  if (!contactFeedback) return;
  contactFeedback.textContent = message;
  contactFeedback.className   = 'contact-feedback contact-feedback--' + type;
  contactFeedback.style.display = 'block';
}

/** Hide the inline feedback div. */
function clearContactFeedback() {
  if (!contactFeedback) return;
  contactFeedback.style.display = 'none';
  contactFeedback.className     = 'contact-feedback';
  contactFeedback.textContent   = '';
}

/**
 * Validate the contact form fields.
 * Returns { valid: true } or { valid: false, message: '...' }.
 */
function validateContactForm() {
  const name    = (contactName    ? contactName.value.trim()    : '');
  const email   = (contactEmail   ? contactEmail.value.trim()   : '');
  const message = (contactMessage ? contactMessage.value.trim() : '');

  if (!name || name.length < 2) {
    return { valid: false, field: contactName, message: 'Please enter your name (at least 2 characters).' };
  }

  // Basic email pattern — good enough for frontend validation
  const emailPattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  if (!email || !emailPattern.test(email)) {
    return { valid: false, field: contactEmail, message: 'Please enter a valid email address.' };
  }

  if (!message || message.length < 10) {
    return { valid: false, field: contactMessage, message: 'Please enter a message (at least 10 characters).' };
  }

  return { valid: true };
}

if (contactForm) {
  contactForm.addEventListener('submit', async (e) => {

    /*
      ── CRITICAL: prevent page refresh ────────────────────────────────────────
      This is the first thing that runs. The browser's default action
      for a form submit event is to navigate to the action URL (or reload
      the current page if no action is set). preventDefault() stops that
      entirely before any other code runs.
    */
    e.preventDefault();

    // Clear any previous feedback
    clearContactFeedback();

    /* 1. Validate fields ---------------------------------------------------- */
    const validation = validateContactForm();
    if (!validation.valid) {
      setContactFeedback(validation.message, 'error');
      // Focus the invalid field so the user knows where to look
      if (validation.field) validation.field.focus();
      return;
    }

    /* 2. Guard: make sure EmailJS SDK loaded -------------------------------- */
    if (typeof emailjs === 'undefined') {
      setContactFeedback(
        'Email service is unavailable. Please email us directly at support@deepfakeai.com',
        'error'
      );
      return;
    }

    /* 3. Show loading state on button --------------------------------------- */
    if (contactSubmitBtn) {
      contactSubmitBtn.disabled   = true;
      contactSubmitBtn.textContent = 'Sending…';
    }

    /* 4. Send via EmailJS --------------------------------------------------- */
    try {
      /*
        emailjs.sendForm() reads the form's input/textarea name attributes
        and maps them to template variables automatically.
        It returns a Promise — we await it.
      */
      await emailjs.sendForm(
        EMAILJS_SERVICE_ID,   // 'service_8c9ygmc'
        EMAILJS_TEMPLATE_ID,  // 'template_8dn52rc'
        contactForm           // the actual <form> DOM element
      );

      /* 5a. SUCCESS --------------------------------------------------------- */
      setContactFeedback(
        '✓ Message sent! We\'ll get back to you as soon as possible.',
        'success'
      );

      // Clear all fields so the form looks fresh
      contactForm.reset();

    } catch (err) {
      /* 5b. FAILURE --------------------------------------------------------- */
      console.error('[EmailJS] Send failed:', err);

      setContactFeedback(
        'Failed to send your message. Please try again, or email us directly at support@deepfakeai.com',
        'error'
      );

    } finally {
      /* 6. Always re-enable the submit button ------------------------------ */
      if (contactSubmitBtn) {
        contactSubmitBtn.disabled    = false;
        contactSubmitBtn.textContent = 'Submit';
      }
    }
  });
}