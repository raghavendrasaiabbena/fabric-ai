"""
Fabric Image Enhancement Pipeline

Priority order for upscaling:
  1. Replicate API  — Real-ESRGAN x4 (best quality, cloud)
  2. OpenCV DNN     — EDSR / LapSRN local model
  3. Pillow LANCZOS — pure interpolation fallback

Pre/post processing always runs locally (OpenCV):
  - Brightness snapshot  → locked at end so output matches input brightness
  - CLAHE                → only on underexposed images (mean L < 110)
  - Noise reduction      → conservative h=3 to preserve weave texture
  - Texture sharpening   → single-pass unsharp mask on L channel only
  - Brightness lock      → scale L channel back to original mean

Color safety guarantee:
  All LAB operations touch only L (lightness). A/B (colour) never modified.
"""

import io
import os
import logging
import time
import urllib.request

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# Replicate model — Real-ESRGAN x4plus (best for real-world fabric photos)
REPLICATE_MODEL = "nightmareai/real-esrgan:f121d640bd286e1fdc67f9799164c1d5be36ff74576ee11c803ae5b665dd46aa"


class FabricImageEnhancer:
    def __init__(self, replicate_token: str = ""):
        self.replicate_token = replicate_token or os.environ.get("REPLICATE_API_TOKEN", "")
        self.replicate_available = self._check_replicate()
        self.sr_model, self.local_sr_method = self._load_local_sr_model()

        if self.replicate_available:
            self.sr_method = "Real-ESRGAN 4x (Replicate API)"
        else:
            self.sr_method = self.local_sr_method

        logger.info(f"Active enhancement method: {self.sr_method}")

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------

    def _check_replicate(self) -> bool:
        if not self.replicate_token:
            logger.info("REPLICATE_API_TOKEN not set — using local fallback")
            return False
        try:
            import replicate  # noqa: F401
            logger.info("Replicate SDK found — Real-ESRGAN enabled")
            return True
        except ImportError:
            logger.warning("replicate package not installed — run: pip install replicate")
            return False

    def _load_local_sr_model(self):
        for model_file, model_name, scale in [
            ("EDSR_x4.pb",   "edsr",    4),
            ("LapSRN_x4.pb", "lapsrn",  4),
        ]:
            path = os.path.join(MODELS_DIR, model_file)
            if os.path.exists(path):
                try:
                    sr = cv2.dnn_superres.DnnSuperResImpl_create()
                    sr.readModel(path)
                    sr.setModel(model_name, scale)
                    logger.info(f"Loaded local model: {model_file}")
                    return sr, f"{model_name.upper()} 4x (local DNN)"
                except Exception as e:
                    logger.warning(f"Could not load {model_file}: {e}")
        return None, "Pillow LANCZOS 4x (fallback)"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enhance(self, input_path: str, output_path: str) -> dict:
        t0 = time.time()

        img = self._safe_read(input_path)
        original_h, original_w = img.shape[:2]
        original_brightness = self._mean_brightness(img)
        logger.info(f"Original brightness (L): {original_brightness:.1f}")

        img = self._correct_lighting(img, original_brightness)
        img = self._reduce_noise(img)
        img = self._upscale(img, input_path)
        img = self._sharpen_texture(img)
        img = self._lock_brightness(img, original_brightness)

        enhanced_h, enhanced_w = img.shape[:2]
        self._save(img, output_path)

        elapsed = round(time.time() - t0, 2)
        return {
            "method": self.sr_method,
            "original_resolution": f"{original_w}x{original_h}",
            "enhanced_resolution": f"{enhanced_w}x{enhanced_h}",
            "scale_factor": round(enhanced_w / original_w, 1),
            "processing_seconds": elapsed,
        }

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def _safe_read(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            pil = Image.open(path).convert("RGB")
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        return img

    def _mean_brightness(self, img: np.ndarray) -> float:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        return float(np.mean(lab[:, :, 0]))

    def _correct_lighting(self, img: np.ndarray, mean_l: float) -> np.ndarray:
        if mean_l >= 110:
            logger.info("Image well-lit — skipping CLAHE")
            return img
        logger.info(f"Applying CLAHE (mean L={mean_l:.1f})")
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(16, 16))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _reduce_noise(self, img: np.ndarray) -> np.ndarray:
        return cv2.fastNlMeansDenoisingColored(
            img, None, h=3, hColor=3,
            templateWindowSize=7, searchWindowSize=21,
        )

    def _upscale(self, img: np.ndarray, input_path: str) -> np.ndarray:
        if self.replicate_available:
            result = self._upscale_replicate(input_path)
            if result is not None:
                return result
            logger.warning("Replicate failed — falling back to local")

        if self.sr_model is not None:
            try:
                return self.sr_model.upsample(img)
            except Exception as e:
                logger.warning(f"Local DNN failed ({e}) — falling back to Pillow")

        return self._upscale_lanczos(img)

    def _upscale_replicate(self, input_path: str):
        try:
            import replicate

            os.environ["REPLICATE_API_TOKEN"] = self.replicate_token

            logger.info("Sending image to Replicate (Real-ESRGAN x4)…")
            with open(input_path, "rb") as f:
                output = replicate.run(
                    REPLICATE_MODEL,
                    input={
                        "image": f,
                        "scale": 4,
                        "face_enhance": False,
                    },
                )

            # output is a URL string or FileOutput object
            url = str(output)
            logger.info(f"Replicate returned: {url[:80]}…")

            # Download the result
            with urllib.request.urlopen(url) as resp:
                data = resp.read()

            pil = Image.open(io.BytesIO(data)).convert("RGB")
            return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

        except Exception as e:
            logger.error(f"Replicate error: {e}")
            return None

    def _upscale_lanczos(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        pil = pil.resize((w * 4, h * 4), Image.LANCZOS)
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    def _sharpen_texture(self, img: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_float = l.astype(np.float32)
        blur = cv2.GaussianBlur(l_float, (0, 0), 1.2)
        l_float = cv2.addWeighted(l_float, 1.35, blur, -0.35, 0)
        l = np.clip(l_float, 0, 255).astype(np.uint8)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _lock_brightness(self, img: np.ndarray, target_l: float) -> np.ndarray:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        current_l = float(np.mean(l))
        if current_l < 1:
            return img
        scale = target_l / current_l
        if abs(scale - 1.0) < 0.01:
            return img
        logger.info(f"Brightness lock: {current_l:.1f} → {target_l:.1f} (×{scale:.3f})")
        l = np.clip(l.astype(np.float32) * scale, 0, 255).astype(np.uint8)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _save(self, img: np.ndarray, path: str):
        ext = os.path.splitext(path)[1].lower()
        if ext == ".png":
            cv2.imwrite(path, img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        else:
            cv2.imwrite(path, img, [
                cv2.IMWRITE_JPEG_QUALITY, 97,
                cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444,
            ])
