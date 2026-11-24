# Oxford-IIIT Pet — CLIP CDE template suffix ablation

Notebook: https://colab.research.google.com/

Frozen OpenAI CLIP ViT-B/32, full torchvision Oxford-IIIT Pet `test` split (3,669 images), same image embeddings in both arms. The 25 released NoLA descriptions per breed were kept fixed. Template: `a photo of a pet {}.`

| Prompt format | Correct | Top-1 |
|---|---:|---:|
| 25 descriptions plus one separate template | 3,141 / 3,669 | 85.61% |
| Same template appended to each of the 25 descriptions | 3,225 / 3,669 | 87.90% |

Paired difference: **+84 correct, +2.29 percentage points** (156 fixed, 72 lost). One normalized mean embedding per breed in both arms. No visual prompt training or DINO teacher ran in this comparison. The baseline is our one-template setup, not released Oxford Pets' two-template configuration.
