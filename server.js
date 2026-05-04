const express = require('express');
const multer = require('multer');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;

const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 200 * 1024 * 1024,
  },
});

app.use(express.static(path.join(__dirname)));

app.post('/upload', upload.single('file'), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded' });
  }

  const result = Math.random() > 0.5 ? 'Real' : 'Fake';
  return res.json({ result });
});

app.use((error, req, res, next) => {
  if (error instanceof multer.MulterError && error.code === 'LIMIT_FILE_SIZE') {
    return res.status(400).json({ error: 'File exceeds 200MB limit' });
  }

  if (error) {
    return res.status(500).json({ error: 'Upload processing failed' });
  }

  return next();
});

app.listen(PORT, () => {
  console.log(`DeepFake Detector AI server running at http://localhost:${PORT}`);
});
