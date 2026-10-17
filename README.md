# PromitoLipi — Bengali Character Detection and Transformer OCR

**Bengali script recognition | Object detection | Sequence-to-sequence OCR | Vision transformers | Low-resource NLP**

## Project overview

PromitoLipi explores two complementary ways of recognizing Bengali writing in images: **localizing individual characters with YOLOv8** and **transcribing word images with transformer-based OCR**. The work uses the PromitoLipi2 dataset and its Bengali character annotations, comprising **93 mapped classes** spanning vowels, consonants, numerals, vowel signs/modifiers, and conjunct forms.

The project covers annotation conversion and quality checks; supervised character detection; custom image-to-text transformer experiments; an EfficientNet-based visual encoder variant; and an exploratory fine-tuning pipeline for Microsoft TrOCR. These are **separate experimental pipelines** rather than one end-to-end benchmarked production system.

## Research questions

- How reliably can a YOLOv8 detector locate and classify the fine-grained symbols in Bengali word images?
- How accurately can an image-to-text transformer recover Bengali strings with complex character composition?
- How do different visual encoders affect character error rate (CER)?
- How should character annotations be processed for detection and sequence-recognition workflows?

## Dataset and annotations

**Primary dataset:** PromitoLipi2 (locally referenced in the notebooks as `PromitoLipi2/WordImages(bmp)` and `WordAnnotations(xml)`). Character-level boxes are encoded in **PASCAL VOC XML**; the project also uses an image-to-character-ID mapping text file for transcription experiments.

The file `imageannotationsid_train.txt` has rows similar to:

```text
20145.bmp: 4 31 11 37 25
20150.bmp: 4 31 33 30
```

Each row associates an image with a sequence of numerical character IDs, which are mapped back to Bengali symbols using the configured class dictionary. This is different from a bounding-box label file.

The notebooks include an OCR dataset split reported as **7,864 training / 983 validation / 983 test** samples (9,830 total). YOLO notebook logs separately show **1,474 validation images and 3,659 labeled instances** during evaluation. These counts relate to different pipelines and should not be treated as a single shared evaluation split.

## Methodology

### A. Dataset preparation and verification

1. Define and maintain the Bengali character-class mapping.
2. Parse PASCAL VOC XML objects and bounding-box coordinates.
3. Convert box coordinates to normalized YOLO format (`class_id x_center y_center width height`), using image dimensions.
4. Prepare image/label directories and dataset YAML configuration.
5. Verify boxes and Bengali category labels by rendering annotations over images with OpenCV and Matplotlib.
6. For word-level OCR, load the character-ID transcription file, map IDs to Unicode Bengali characters, tokenize sequences, and build training/validation/test datasets.

**Annotation-conversion note:** Some conversion and train/test preparation cells in `yolo-promitilipi.ipynb` are commented examples; inspect and run the preprocessing steps against your local dataset paths before assuming that conversion has already been executed in a fresh environment.

### B. YOLOv8 Bengali character detection

- **Model:** pretrained `yolov8m.pt` through Ultralytics.
- **Training configuration shown:** 5 epochs, batch size 16, image size 640, CUDA device 0.
- **Evaluation:** box precision, box recall, mAP@0.50, and mAP@0.50:0.95.
- **Qualitative validation:** infer on held-out `.bmp` files and draw predicted boxes and Bengali labels.

The trained detector predicts symbol identities and positions; it does not itself produce a fully ordered, linguistically postprocessed word transcription.

### C. Custom transformer OCR

The custom OCR notebooks implement an image-to-text sequence-learning workflow with:

- A `PromitoLipiDataset` class for image/transcription pairs.
- An explicit Bengali character vocabulary with special start/end, unknown, and padding tokens.
- Positional encoding and transformer sequence modeling.
- Loss functions for sequence learning, training and validation loops, scheduled learning rates, and checkpoint selection by validation CER/loss.
- Text decoding and qualitative inspection of ground-truth versus predicted strings.

`custom-transformer.ipynb` and `transformer-ocr (1).ipynb` explore this general pipeline. `custom-transformer-efficient.ipynb` modifies the visual feature extractor with **pretrained EfficientNet-B7**.

### D. TrOCR fine-tuning exploration

`trocr_finetune.ipynb` prepares a word-image/transcription dataset in an IAM-like CSV/image format and instantiates Hugging Face `TrOCRProcessor` and `VisionEncoderDecoderModel` (`microsoft/trocr-base-handwritten` processor and `microsoft/trocr-base-stage1` model). It defines an 80/20 split, dataloaders, an AdamW optimizer, a 20-epoch training loop, and CER evaluation code.

**Evidence status:** That notebook contains code but no saved execution outputs; it is documented as a *fine-tuning experiment/pipeline*, **not a verified trained benchmark**.

## Results from notebook outputs

### YOLOv8 detection

The YOLOv8 training output includes the following overall bounding-box metrics for a validation pass:

| Metric | Value |
|---|---:|
| Precision (boxes) | **0.862** |
| Recall (boxes) | **0.861** |
| mAP@0.50 | **0.916** |
| mAP@0.50:0.95 | **0.648** |
| Validation images | **1,474** |
| Annotated objects | **3,659** |

A nearby final logged evaluation reports mAP@0.50 = **0.915** and mAP@0.50:0.95 = **0.647**. The 0.916 value above refers to the logged validation pass, not a separately established held-out test result.

### Word-level transformer recognition

| Notebook | Logged overall test CER | Other recorded details |
|---|---:|---|
| `custom-transformer.ipynb` | **0.299** | A separate detailed logging call outputs 0.298 |
| `custom-transformer-efficient.ipynb` | **0.459** | A separate detailed logging call outputs 0.461; EfficientNet-B7 encoder |

**Interpretation:** Lower CER is better. In these particular saved runs, the custom transformer has a lower overall test CER than the EfficientNet variant. This is an observation, **not proof of architecture superiority**: the notebooks use different training durations and settings (150 versus 50 logged epochs), and should be rerun under matched conditions for a controlled comparison.

Additional notebook cells report CER for **short (length ≤ 2)** versus **longer (length > 2)** transcription subsets:

| Notebook | Length ≤ 2 subset CER | Length > 2 subset CER |
|---|---:|---:|
| Custom transformer | 0.091 | 0.557 |
| EfficientNet variant | 0.105 | 0.881 |

These subset values are **not** interchangeable with the full-test CER, and their weighting and filtering should be reviewed before aggregating results.

`transformer-ocr (1).ipynb` includes training logs but does not establish an equivalent completed held-out test metric in its saved output. The TrOCR fine-tuning notebook contains no saved result metrics.

### Qualitative findings

- YOLO visualizations demonstrate detections with class names and bounding boxes on selected test images.
- OCR notebooks log per-image ground-truth/predicted text and CER, including both exact matches and errors.
- Short strings generally perform better than longer strings in the logged length-based OCR analyses, emphasizing the difficulty of compositional Bengali script recognition.

## Evaluation measures

- **Detection precision/recall:** quality of predicted character bounding boxes and classes.
- **mAP@0.50:** average precision at intersection-over-union (IoU) threshold 0.50.
- **mAP@0.50:0.95:** average over multiple IoU thresholds.
- **CER:** character-level edit distance divided by the number of reference characters; **lower is better**.

Detection mAP and OCR CER measure different tasks and cannot be directly compared as if they were the same accuracy percentage.

## Repository files

```text
.
├── README.md
├── yolo-promitilipi.ipynb
├── custom-transformer.ipynb
├── custom-transformer-efficient.ipynb
├── transformer-ocr.ipynb
├── trocr_finetune.ipynb
└── imageannotationsid_train.txt
```

The proposed `transformer-ocr.ipynb` is a cleaner name for the uploaded `transformer-ocr (1).ipynb`. **Dataset images, XML labels, trained weights and configuration files are not included in this file list**; add them only where their licenses and sizes allow.

## Environment and reproduction

**Core tools:** Python, PyTorch, torchvision, OpenCV, Matplotlib, NumPy, Pillow, scikit-learn, Ultralytics YOLO, Hugging Face Transformers, `jiwer`, `evaluate`, `efficientnet-pytorch`.

Example environment setup (package versions should be pinned after validating an end-to-end run):

```bash
python -m venv .venv
source .venv/bin/activate
pip install jupyter torch torchvision numpy pandas pillow matplotlib opencv-python scikit-learn ultralytics transformers datasets evaluate jiwer efficientnet-pytorch tqdm
jupyter notebook
```

To reproduce the experiments:

1. Obtain PromitoLipi2 through an authorized source and preserve the `.bmp` images and annotation XML files.
2. Update the Kaggle-style absolute paths in the notebooks to your local image, label, YAML, and output directories.
3. Verify the class ID mapping and conversion from XML bounding boxes to YOLO labels.
4. Run the YOLO notebook for detection experiments and validation.
5. Run the transformer notebooks separately for transcription experiments, using the provided image-to-ID mapping file and corresponding images.
6. Compare metrics only on clearly identified, stable evaluation splits.

GPU acceleration is recommended for training. The code is presented as exploratory Jupyter experiments, not a packaged one-command application.

## Limitations and future directions

- Ensure fixed and publicly documented dataset splits and random seeds before making rigorous cross-model claims.
- Verify dataset class coverage and class-wise performance, especially for visually similar conjuncts and rare symbols.
- Report CER plus word error rate (WER), exact-match rate, and Unicode normalization policy for OCR.
- Test reading-order reconstruction that combines YOLO character detections into word-level text.
- Standardize epochs, optimizer, augmentation, and compute budgets across transformer variants.
- Add checkpoint provenance, an explicit trained TrOCR evaluation, and reproducible dependencies.
- Evaluate robustness to scanning noise, handwriting variation, and unusual character compositions.

## Related literature

The accompanying paper, **“Optical Text Recognition in Nepali and Bengali: A Transformer-based Approach”** (Hasan et al., 2024, [arXiv:2404.02375](https://arxiv.org/abs/2404.02375)), provides context on transformer-based OCR in low-resource South Asian languages. **Its Bengali/Nepali CER and WER values are results from the cited paper, not results of the notebooks in this repository.**

## Research context

This project explores Bengali OCR across **detection and recognition**, with an emphasis on dataset representation, low-resource scripts, custom transformer modeling, and empirical evaluation. It provides a basis for future work on multilingual OCR, document understanding, efficient vision-language architectures, and reliable text recognition for complex scripts.
