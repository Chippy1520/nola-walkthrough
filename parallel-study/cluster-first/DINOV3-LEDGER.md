# DINOv3 retrieval encoder comparison

Blocked before model loading: official Hugging Face `facebook/dinov3-vits16plus-pretrain-lvd1689m` weight endpoint returned HTTP 401 `GatedRepo`; no local Hugging Face login or matching official checkpoint found. Meta's official request form is an alternative source. No DINOv3 run, dependency installation, or substitution was performed.

Resume with official local checkpoint or approved Hugging Face authentication. Keep the existing PyTorch 2.5.1 CUDA environment intact, use isolated dependencies, and smoke-test actual GPU inference. Reuse the same EuroSAT and CIFAR-100 candidate pools, starting anchors, quotas, CLIP scores, and original DINO head features from the spectral comparison. Change only the frozen retrieval features. Compare against direct original DINO retrieval and CLIP confidence; audit quality, class recall, and subtype coverage, not only pooled accuracy. No CLIP visual prompt training or test inference.
